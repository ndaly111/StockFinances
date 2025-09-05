#!/usr/bin/env python3
"""Helper utilities for applying per-ticker segment overrides."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

import pandas as pd

from segment_formatting_helpers import _humanize_segment_name


def canon(s: str) -> str:
    """Canonicalize a string for lookups."""
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def strip_namespace(s: str) -> str:
    """Return text after the first ':' if present."""
    if not isinstance(s, str):
        return s
    return s.split(":", 1)[1] if ":" in s else s


def remove_suffixes(s: str) -> str:
    """Drop common XBRL member suffixes like 'Member' or 'SegmentMember'."""
    if not isinstance(s, str):
        return s
    return re.sub(r"(Segment)?Member$", "", s)


def load_overrides(path: str = "segment_overrides.json") -> Dict:
    """Load overrides JSON, returning {} if missing or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _merge_dict(a: Dict, b: Dict) -> Dict:
    """Deep-merge two dictionaries."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        elif k in out and isinstance(out[k], list) and isinstance(v, list):
            out[k] = out[k] + v
        else:
            out[k] = v
    return out


def _build_exact_map(d: Dict[str, str]) -> Dict[str, str]:
    return {canon(k): v for k, v in d.items()}


def _build_regex_list(lst: List[Dict[str, str]]) -> List:
    out = []
    for item in lst or []:
        pat = item.get("pattern")
        rep = item.get("replace", "")
        if pat is None:
            continue
        out.append((re.compile(pat), rep))
    return out


def apply_segment_overrides(df: pd.DataFrame, ticker: str, overrides: Dict) -> pd.DataFrame:
    """Apply overrides for a ticker; never raises."""
    if df is None or df.empty or not overrides:
        return df
    orig_df = df.copy()

    rules = _merge_dict(overrides.get("GLOBAL", {}), overrides.get(ticker.upper(), {}))
    if not rules:
        return df

    fail_open = rules.get("fail_open", True)

    df = df.copy()
    if "Axis" not in df.columns and "AxisType" in df.columns:
        df = df.rename(columns={"AxisType": "Axis"})

    # 2) Axis search filter
    axis_search = rules.get("axis_search", {})
    include_exact = axis_search.get("include_exact", [])
    include_regex = [re.compile(p) for p in axis_search.get("include_regex", [])]
    exclude_exact = axis_search.get("exclude_exact", [])
    exclude_regex = [re.compile(p) for p in axis_search.get("exclude_regex", [])]

    if include_exact or include_regex:
        mask = pd.Series(False, index=df.index)
        if include_exact:
            mask |= df["Axis"].isin(include_exact)
        if include_regex:
            mask |= df["Axis"].apply(lambda a: any(r.search(a) for r in include_regex))
        df = df[mask]
    if exclude_exact:
        df = df[~df["Axis"].isin(exclude_exact)]
    if exclude_regex:
        df = df[~df["Axis"].apply(lambda a: any(r.search(a) for r in exclude_regex))]
    if df.empty and fail_open:
        return orig_df

    # 3) Axis translate
    axis_translate = rules.get("axis_translate", {})
    if axis_translate:
        df["Axis"] = df["Axis"].replace(axis_translate)
        df = df.groupby(["Axis", "Segment", "Year"], as_index=False)[["Revenue", "OpIncome"]].sum()

    # 4) Axis labels
    axis_labels = rules.get("axis_labels", {})
    df["AxisLabel"] = df["Axis"].map(lambda a: axis_labels.get(a, a.split(":")[-1]))

    # 5) Member translation
    raw_exact = _build_exact_map(rules.get("member_aliases_raw", {}))
    raw_exact_by_axis = {ax: _build_exact_map(m) for ax, m in rules.get("member_aliases_raw_by_axis", {}).items()}
    hum_exact = _build_exact_map(rules.get("member_aliases", {}))
    hum_exact_by_axis = {ax: _build_exact_map(m) for ax, m in rules.get("member_aliases_by_axis", {}).items()}

    raw_regex = _build_regex_list(rules.get("member_regex_raw", []))
    raw_regex_by_axis = {ax: _build_regex_list(lst) for ax, lst in rules.get("member_regex_raw_by_axis", {}).items()}
    hum_regex = _build_regex_list(rules.get("member_regex", []))
    hum_regex_by_axis = {ax: _build_regex_list(lst) for ax, lst in rules.get("member_regex_by_axis", {}).items()}

    def translate_row(row):
        axis = row["Axis"]
        seg = row["Segment"]
        raw_member = strip_namespace(seg)
        raw_base = remove_suffixes(raw_member)
        human = _humanize_segment_name(raw_base)
        canon_raw = canon(raw_base)
        canon_hum = canon(human)

        name = None
        if canon_raw in raw_exact:
            name = raw_exact[canon_raw]
        elif canon_raw in raw_exact_by_axis.get(axis, {}):
            name = raw_exact_by_axis[axis][canon_raw]
        elif canon_hum in hum_exact:
            name = hum_exact[canon_hum]
        elif canon_hum in hum_exact_by_axis.get(axis, {}):
            name = hum_exact_by_axis[axis][canon_hum]
        else:
            for rx, rep in raw_regex:
                new_raw = rx.sub(rep, raw_base)
                if new_raw != raw_base:
                    name = _humanize_segment_name(new_raw)
                    break
            else:
                for rx, rep in raw_regex_by_axis.get(axis, []):
                    new_raw = rx.sub(rep, raw_base)
                    if new_raw != raw_base:
                        name = _humanize_segment_name(new_raw)
                        break
                else:
                    for rx, rep in hum_regex:
                        new_hum = rx.sub(rep, human)
                        if new_hum != human:
                            name = new_hum
                            break
                    else:
                        for rx, rep in hum_regex_by_axis.get(axis, []):
                            new_hum = rx.sub(rep, human)
                            if new_hum != human:
                                name = new_hum
                                break
        if name is None:
            name = human
        return name

    df["Segment"] = df.apply(translate_row, axis=1)
    df = df.groupby(["Axis", "AxisLabel", "Segment", "Year"], as_index=False)[["Revenue", "OpIncome"]].sum()

    # 6) Axis ordering
    prefer_axes = rules.get("prefer_axes", [])
    if prefer_axes:
        order_map = {label: i + 1 for i, label in enumerate(prefer_axes)}
        df["AxisOrder"] = df["AxisLabel"].map(lambda x: order_map.get(x, 999))

    # 7) Fail-open guarantee
    if df.empty and fail_open:
        return orig_df

    # Backward compatibility
    df["AxisType"] = df["Axis"]

    return df

