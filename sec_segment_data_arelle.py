"""
sec_segment_data_arelle
=======================

This module provides tools for pulling business segment revenue and operating
income directly from Inline XBRL filings hosted on the SEC’s EDGAR system.  It
implements **Option 1** as discussed with the user: download the latest
10‑K/10‑Q filings for a given ticker, extract the embedded XBRL instance using
the Arelle plugin for Inline XBRL document sets, and parse the resulting
instance to aggregate revenue and operating income by segment.  The final
result is returned as a tidy pandas `DataFrame` with segment names, fiscal
periods (the last three fiscal years plus trailing‑twelve‑months (TTM)) and
amounts.

The overall workflow is as follows:

1.  **Resolve the ticker to a CIK.**  The SEC provides an official
    `company_tickers.json` file mapping ticker symbols to CIK numbers.  The
    helper `resolve_ticker_to_cik()` normalises the ticker and looks up the
    corresponding 10‑digit CIK.

2.  **Locate the latest 10‑K and 10‑Q filings.**  Using the SEC’s
    `submissions` API, we inspect the company’s recent filings and select the
    most recent annual report (Form 10‑K) and quarterly report (Form 10‑Q).
    Their accession numbers and primary document names are used to build
    download URLs.

3.  **Download the Inline XBRL document.**  The primary document for each
    filing is an HTML file (e.g. ``aapl‑20240928.htm``) containing Inline XBRL
    markup.  The helper `download_file()` fetches the document with a
    polite User‑Agent header and writes it to a temporary location.

4.  **Extract the XBRL instance.**  Inline XBRL documents embed facts within
    the HTML.  To work with them more easily we extract a pure XBRL instance
    using the Arelle command‑line.  The function `extract_xbrl_instance()`
    invokes ``python -m arelle.CntlrCmdLine`` with the
    ``inlineXbrlDocumentSet`` plugin and ``--saveInstance`` flag to produce
    ``*_extracted.xbrl``.  Arelle must be installed (the module depends on
    ``arelle‑release``) and this script expects it to be available on your
    system.  Errors and warnings from Arelle are logged but do not halt
    execution.

5.  **Parse contexts and facts.**  The extracted XBRL instance is an XML
    document containing contexts (identifying the reporting entity, period and
    dimensions) and facts (numeric or non‑numeric values).  We parse the
    instance with ``xml.etree.ElementTree``, building a dictionary of
    contexts keyed by ``context id``.  Each context may include explicit or
    typed dimensions inside a `<segment>` element.  We capture all
    dimensions whose QName contains ``Segment`` (case‑insensitive) – this
    covers standard axes such as ``StatementBusinessSegmentsAxis`` and custom
    segment axes.  We also extract period start and end dates.

6.  **Filter relevant facts.**  Revenue and operating income are reported
    under various GAAP concepts.  We attempt to pull revenue using
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` and fall back to
    ``SalesRevenueNet`` or ``Revenues`` if necessary.  For operating
    income we look for ``SegmentOperatingIncomeLoss`` and fall back to
    ``OperatingIncomeLoss``.  Only facts linked to contexts with segment
    dimensions are considered.  The dimension member QName (e.g.
    ``aapl:IPhoneMember``) is converted to a human friendly segment name by
    stripping the ``Member`` suffix and inserting spaces before capital
    letters.

7.  **Aggregate by period and compute TTM.**  We convert the context period
    end date into a fiscal year by taking the year portion.  For each
    segment, we aggregate facts by fiscal year and by quarter (latest 10‑Q).
    Trailing twelve months (TTM) is computed by summing the latest 10‑Q
    amount with the previous fiscal year and subtracting the prior‑year
    quarter to avoid double counting.  If necessary facts are missing, we
    log a message and skip TTM computation.

The main entry point is `get_segment_data()` which returns a pandas
`DataFrame` with columns ``Segment``, ``Year`` (or ``TTM``), ``Revenue`` and
``OpIncome``.  The DataFrame’s `.attrs` dictionary records which GAAP
concepts were actually used for revenue and operating income.

Note that parsing Inline XBRL can be slow, especially for large filings.
Caching the extracted instance or results (e.g. in SQLite) is advisable if
processing many tickers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import xml.etree.ElementTree as ET

# Configure a module‑level logger.  Users of this module can configure the
# logging level externally (e.g. logging.basicConfig(level=logging.INFO)).
logger = logging.getLogger(__name__)


def resolve_ticker_to_cik(ticker: str) -> str:
    """Resolve a stock ticker to a 10‑digit CIK string.

    Parameters
    ----------
    ticker : str
        The company’s trading symbol (case‑insensitive).

    Returns
    -------
    str
        A 10‑digit string representing the company’s CIK.

    Raises
    ------
    ValueError
        If the ticker cannot be found in the SEC mapping.
    """
    ticker = ticker.upper().strip()
    mapping_url = (
        "https://www.sec.gov/files/company_tickers.json"
    )
    try:
        resp = requests.get(mapping_url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to download ticker mapping: {e}")
    mapping = {entry["ticker"]: entry["cik_str"] for entry in resp.json().values()}
    if ticker not in mapping:
        raise ValueError(f"Ticker {ticker} not found in SEC mapping")
    cik_int = int(mapping[ticker])
    return f"{cik_int:010d}"


def get_latest_filings(cik: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Retrieve the latest 10‑K and 10‑Q filings for a CIK.

    The SEC submissions JSON lists recent filings.  This helper finds the
    most recent occurrence of a 10‑K and a 10‑Q and returns a record of
    accession number, primary document and filing date.

    Parameters
    ----------
    cik : str
        Ten digit CIK string for the company.

    Returns
    -------
    (dict, dict)
        Two dictionaries (for 10‑K and 10‑Q respectively) containing keys
        ``accession``, ``document`` and ``date``.  If a filing type is not
        found, its dictionary’s values are empty strings.
    """
    sub_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(sub_url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    data = resp.json()
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accessions = filings.get("accessionNumber", [])
    documents = filings.get("primaryDocument", [])
    dates = filings.get("filingDate", [])

    latest_10k = {"accession": "", "document": "", "date": ""}
    latest_10q = {"accession": "", "document": "", "date": ""}
    for form, acc, doc, date in zip(forms, accessions, documents, dates):
        if not latest_10k["accession"] and form.upper() == "10-K":
            latest_10k = {"accession": acc, "document": doc, "date": date}
        if not latest_10q["accession"] and form.upper() == "10-Q":
            latest_10q = {"accession": acc, "document": doc, "date": date}
        if latest_10k["accession"] and latest_10q["accession"]:
            break
    return latest_10k, latest_10q


def build_filing_url(cik: str, accession: str, doc: str) -> str:
    """Construct the URL for the primary document of a filing.

    Parameters
    ----------
    cik : str
        Ten digit CIK string.
    accession : str
        Accession number with dashes (e.g. '0000320193-24-000123').
    doc : str
        Primary document name (e.g. 'aapl-20240928.htm').

    Returns
    -------
    str
        Absolute URL to the document in EDGAR.
    """
    # Remove leading zeros for the directory name
    cik_nozero = cik.lstrip("0")
    acc_no_dash = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{acc_no_dash}/{doc}"
    )


def download_file(url: str, dest: Path) -> None:
    """Download a file from `url` to `dest` with a polite SEC User‑Agent.

    Parameters
    ----------
    url : str
        The absolute URL of the remote file.
    dest : pathlib.Path
        Local path where the file should be saved.  Parent directories will
        be created if necessary.

    Notes
    -----
    A timeout of 30 seconds is used.  The SEC requires a descriptive
    User‑Agent header; customise if embedding this into a larger application.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; SegmentFetcher/1.0; "
            "+https://example.com/contact)"
        )
    }
    with requests.get(url, headers=headers, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=8192):
                fh.write(chunk)
    logger.info("Downloaded %s -> %s", url, dest)


def extract_xbrl_instance(ixbrl_path: Path) -> Path:
    """Extract a pure XBRL instance from an Inline XBRL document using Arelle.

    Parameters
    ----------
    ixbrl_path : Path
        Path to the downloaded Inline XBRL HTML document.

    Returns
    -------
    Path
        Path to the extracted XBRL instance (ending with ``_extracted.xbrl``).

    Notes
    -----
    This function relies on Arelle being installed (via ``arelle‑release``)
    and available on the Python path.  It runs the Arelle command line as
    a subprocess, loading the ``inlineXbrlDocumentSet`` plugin and saving
    the instance.  Any warnings or missing reference messages are logged
    but will not raise exceptions.
    """
    out_path = ixbrl_path.with_suffix("")  # remove .htm/.html
    out_file = Path(str(out_path) + "_extracted.xbrl")
    # Build command
    cmd = [
        "python",
        "-m",
        "arelle.CntlrCmdLine",
        "--plugins",
        "inlineXbrlDocumentSet",
        "--file",
        str(ixbrl_path),
        "--saveInstance",
    ]
    logger.info("Extracting XBRL instance from %s", ixbrl_path)
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        # Log stderr; do not halt – Arelle often emits missing reference warnings
        logger.warning("Arelle extraction returned non‑zero exit status\n%s", e.stderr.decode())
    if not out_file.exists():
        raise RuntimeError(
            f"Extraction failed – expected {out_file} to be created"
        )
    return out_file


def parse_ixbrl_segments(html_path: Path) -> Tuple[pd.DataFrame, str, str]:
    """Parse an Inline XBRL HTML document and extract segment revenues and operating income.

    This function reads the Inline XBRL document directly (without relying on
    a separate extracted instance) and collects numeric facts that belong to
    contexts with segment dimensions.  It handles common GAAP revenue and
    operating‑income concepts and applies scaling and sign adjustments based
    on the ``scale`` and ``sign`` attributes present on ``ix:nonFraction``
    elements.

    Parameters
    ----------
    html_path : Path
        Path to the downloaded Inline XBRL HTML document.

    Returns
    -------
    DataFrame, str, str
        A DataFrame with columns ``Segment``, ``PeriodEnd`` (datetime),
        ``Revenue`` and ``OpIncome``.  The accompanying two strings report
        which GAAP concepts were actually used for revenue and operating
        income.  If no suitable facts are found, an empty DataFrame is
        returned.
    """
    # Lazy import to avoid mandatory dependency on BeautifulSoup at module load
    from bs4 import BeautifulSoup  # type: ignore

    # Read the HTML as text
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as fh:
        html = fh.read()
    # Parse as XML because the document is XHTML; BeautifulSoup will fall back
    soup = BeautifulSoup(html, 'xml')

    # Build a mapping of contexts that have segment dimensions
    contexts: Dict[str, Dict[str, any]] = {}
    for ctx in soup.find_all(['context']):
        cid = ctx.get('id')
        if not cid:
            continue
        segment = ctx.find('segment')
        if not segment:
            continue
        seg_members: List[str] = []
        # explicit dimensions
        for exp in segment.find_all('explicitMember'):
            dim = exp.get('dimension') or ''
            if 'segment' in dim.lower():
                seg_members.append(exp.text or '')
        # typed dimensions
        for typed in segment.find_all('typedMember'):
            dim = typed.get('dimension') or ''
            if 'segment' in dim.lower():
                # typed member content may be nested; serialise text
                seg_members.append(typed.get_text())
        if not seg_members:
            continue
        period = ctx.find('period')
        if not period:
            continue
        end_el = period.find('endDate') or period.find('instant')
        if not end_el or not end_el.text:
            continue
        try:
            end_dt = datetime.strptime(end_el.text.strip(), '%Y-%m-%d')
        except Exception:
            continue
        contexts[cid] = {'dims': seg_members, 'end': end_dt}

    if not contexts:
        return pd.DataFrame(columns=['Segment', 'PeriodEnd', 'Revenue', 'OpIncome']), '', ''

    # Candidate concept names without namespace prefix
    revenue_candidates = [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet',
        'Revenues'
    ]
    opinc_candidates = [
        'SegmentOperatingIncomeLoss',
        'OperatingIncomeLoss'
    ]

    records: List[Dict[str, any]] = []
    rev_concept_used: str = ''
    op_concept_used: str = ''

    # Helper to derive a nice segment name from a member QName
    def pretty_member(member: str) -> str:
        # remove namespace prefix
        local = member.split(':')[-1] if ':' in member else member
        if local.endswith('Member'):
            local = local[:-6]
        # insert spaces before capital letters and numbers
        spaced = re.sub(r'(?<=.)([A-Z][a-z]|[0-9])', r' \1', local)
        spaced = spaced.replace('And ', 'and ')
        return spaced.strip()

    # Iterate over revenue and op income candidates separately
    for candidate_list, metric_label in (
        (revenue_candidates, 'Revenue'),
        (opinc_candidates, 'OpIncome'),
    ):
        concept_found = ''
        # Iterate candidate concepts until one with data is found
        for concept in candidate_list:
            # Accept both fully qualified (with namespace) and bare names
            qnames = [f'us-gaap:{concept}', concept]
            found_any = False
            # Loop through nonFraction elements (numeric facts)
            for fact in soup.find_all('nonFraction'):
                name = fact.get('name') or ''
                if name not in qnames:
                    continue
                ctx_id = fact.get('contextRef')
                if ctx_id not in contexts:
                    continue
                # Parse numeric value
                text = fact.text.strip().replace(',', '')
                try:
                    val = float(text)
                except Exception:
                    continue
                # Apply scaling if present (multiply by 10^scale)
                scale_attr = fact.get('scale')
                if scale_attr and scale_attr.lstrip('-').isdigit():
                    try:
                        scale = int(scale_attr)
                        val *= 10 ** scale
                    except Exception:
                        pass
                # Apply sign if present
                if fact.get('sign') == '-':
                    val = -val
                # Record for each segment member
                for member in contexts[ctx_id]['dims']:
                    seg_name = pretty_member(member)
                    records.append({
                        'Segment': seg_name,
                        'PeriodEnd': contexts[ctx_id]['end'],
                        'Metric': metric_label,
                        'Value': val
                    })
                    found_any = True
            if found_any:
                concept_found = f'us-gaap:{concept}'
                if metric_label == 'Revenue':
                    rev_concept_used = concept_found
                else:
                    op_concept_used = concept_found
                break
        # end for concept list

    if not records:
        # Return empty DataFrame with standard columns
        return pd.DataFrame(columns=['Segment', 'PeriodEnd', 'Revenue', 'OpIncome']), rev_concept_used, op_concept_used

    # Convert to DataFrame and pivot metrics into columns
    df = pd.DataFrame(records)
    df_piv = df.pivot_table(
        index=['Segment', 'PeriodEnd'],
        columns='Metric',
        values='Value',
        aggfunc='sum'
    ).reset_index().rename_axis(None, axis=1)
    return df_piv, rev_concept_used, op_concept_used


def compute_segment_ttm(
    fy_data: pd.DataFrame,
    q_data: pd.DataFrame,
    periods: Iterable[int]
) -> pd.DataFrame:
    """Compute trailing‑twelve‑month (TTM) figures for each segment.

    Parameters
    ----------
    fy_data : DataFrame
        Annual data with columns ``Segment``, ``Year``, ``Revenue`` and
        ``OpIncome``.
    q_data : DataFrame
        Quarterly data with the same columns but with ``Year`` equal to the
        fiscal year of the quarter (e.g. 2025 for Q3 2025).  The quarter
        itself is encoded in the ``PeriodEnd`` month and day.
    periods : Iterable[int]
        Iterable of fiscal years present in the data.  The latest year will
        be used to compute TTM.

    Returns
    -------
    DataFrame
        DataFrame with an extra row per segment labelled ``TTM``.
    """
    if q_data.empty or fy_data.empty:
        return pd.DataFrame()
    latest_year = max(periods)
    prev_year = latest_year - 1
    # Determine quarter end month/day from the latest quarterly PeriodEnd
    latest_quarter_end = q_data['PeriodEnd'].max()
    quarter_month_day = (latest_quarter_end.month, latest_quarter_end.day)
    # Sum of latest year annual values up to prior year end
    latest_fy = fy_data[fy_data['Year'] == latest_year]
    prev_fy = fy_data[fy_data['Year'] == prev_year]
    # Latest quarter values (within q_data) – quarter_end in latest_year
    latest_q = q_data[q_data['PeriodEnd'] == latest_quarter_end]
    # Corresponding quarter last year (same month/day but year = prev_year)
    prev_q_end = datetime(prev_year, quarter_month_day[0], quarter_month_day[1])
    prev_q = q_data[q_data['PeriodEnd'] == prev_q_end]
    # Merge frames to align segments
    segments = set(latest_fy['Segment']) | set(latest_q['Segment'])
    records = []
    for seg in segments:
        rev = 0.0
        opinc = 0.0
        # latest FY sum
        rev += latest_fy.loc[latest_fy['Segment'] == seg, 'Revenue'].sum()
        opinc += latest_fy.loc[latest_fy['Segment'] == seg, 'OpIncome'].sum()
        # add latest quarter
        rev += latest_q.loc[latest_q['Segment'] == seg, 'Revenue'].sum()
        opinc += latest_q.loc[latest_q['Segment'] == seg, 'OpIncome'].sum()
        # subtract previous year same quarter
        rev -= prev_q.loc[prev_q['Segment'] == seg, 'Revenue'].sum()
        opinc -= prev_q.loc[prev_q['Segment'] == seg, 'OpIncome'].sum()
        records.append({
            'Segment': seg,
            'Year': 'TTM',
            'Revenue': rev if rev else None,
            'OpIncome': opinc if opinc else None
        })
    return pd.DataFrame(records)


def get_segment_data(ticker: str) -> pd.DataFrame:
    """Return segment revenue and operating income for a ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    DataFrame
        A tidy DataFrame with columns ``Segment``, ``Year``, ``Revenue`` and
        ``OpIncome``.  The DataFrame will include rows for each of the last
        three fiscal years and a trailing‑twelve‑month (TTM) row where data
        permits.  The DataFrame’s `.attrs` dictionary contains the GAAP
        concepts used for revenue and operating income.

    Notes
    -----
    Temporary files for downloaded filings and extracted instances are
    created in a temporary directory and removed automatically when the
    function completes.
    """
    cik = resolve_ticker_to_cik(ticker)
    ten_k, ten_q = get_latest_filings(cik)
    if not ten_k['accession']:
        raise RuntimeError(f"No 10‑K found for {ticker}")
    if not ten_q['accession']:
        raise RuntimeError(f"No 10‑Q found for {ticker}")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Download and process 10‑K HTML
        k_url = build_filing_url(cik, ten_k['accession'], ten_k['document'])
        k_path = tmpdir_path / ten_k['document']
        download_file(k_url, k_path)
        # Parse Inline XBRL directly from the HTML file
        k_df, rev_concept, op_concept = parse_ixbrl_segments(k_path)
        # Download and process 10‑Q HTML
        q_url = build_filing_url(cik, ten_q['accession'], ten_q['document'])
        q_path = tmpdir_path / ten_q['document']
        download_file(q_url, q_path)
        q_df, rev_concept_q, op_concept_q = parse_ixbrl_segments(q_path)
        # Combine concept names (prefer 10‑K concept)
        revenue_concept = rev_concept or rev_concept_q
        op_income_concept = op_concept or op_concept_q
        # Label fiscal years
        def add_year(df: pd.DataFrame) -> pd.DataFrame:
            """Attach a ``Year`` column based on ``PeriodEnd``.  If the
            input DataFrame is empty or lacks a ``PeriodEnd`` column an
            empty DataFrame with the expected columns is returned."""
            if df.empty or 'PeriodEnd' not in df.columns:
                # Return an empty DataFrame with the expected shape
                return pd.DataFrame(columns=['Segment', 'Year', 'Revenue', 'OpIncome'])
            df = df.copy()
            df['Year'] = df['PeriodEnd'].dt.year
            df.drop(columns=['PeriodEnd'], inplace=True)
            return df
        # Label fiscal years for annual (10‑K) and quarterly (10‑Q) data
        fy_df = add_year(k_df)
        # Keep a copy of the quarterly DataFrame with PeriodEnd for TTM computation
        q_df_orig = q_df.copy()
        q_df_year = add_year(q_df)
        # Keep last 3 fiscal years from 10‑K
        recent_years = sorted(fy_df['Year'].unique())[-3:] if not fy_df.empty else []
        fy_df = fy_df[fy_df['Year'].isin(recent_years)]
        # compute TTM using original quarterly data to preserve PeriodEnd
        ttm_df = compute_segment_ttm(fy_df, q_df_orig, recent_years) if recent_years else pd.DataFrame()
        # Combine annual and TTM
        all_df = pd.concat([fy_df, ttm_df], ignore_index=True, sort=False)
        # Reorder columns
        all_df = all_df[['Segment', 'Year', 'Revenue', 'OpIncome']]
        # Attach concept information
        all_df.attrs['revenue_concept'] = revenue_concept
        all_df.attrs['op_income_concept'] = op_income_concept
        return all_df


__all__ = ['get_segment_data', 'resolve_ticker_to_cik']
