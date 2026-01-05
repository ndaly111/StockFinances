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
        group_cols = [col for col in ["Axis", "Segment", "Year", "PeriodType"] if col in df.columns]
        df = df.groupby(group_cols, as_index=False)[["Revenue", "OpIncome"]].sum()

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

    axis_series = df["Axis"]
    raw_member = df["Segment"].map(strip_namespace)
    raw_base = raw_member.map(remove_suffixes)
    human = raw_base.map(_humanize_segment_name)

    def _safe_canon_series(series: pd.Series) -> pd.Series:
        return series.map(lambda v: canon("" if pd.isna(v) else v))

    canon_raw = _safe_canon_series(raw_base)
    canon_hum = _safe_canon_series(human)

    raw_base_str = raw_base.fillna("").astype(str)
    human_str = human.fillna("").astype(str)

    translated = pd.Series(pd.NA, index=df.index, dtype="object")

    def assign_from_dict(keys: pd.Series, mapping: Dict[str, str], mask: pd.Series | None = None) -> None:
        if not mapping:
            return
        available = translated.isna()
        if mask is not None:
            available &= mask.fillna(False)
        if not available.any():
            return
        hits = keys[available].map(mapping).dropna()
        if not hits.empty:
            translated.loc[hits.index] = hits

    assign_from_dict(canon_raw, raw_exact)
    for ax, mapping in raw_exact_by_axis.items():
        assign_from_dict(canon_raw, mapping, axis_series == ax)

    assign_from_dict(canon_hum, hum_exact)
    for ax, mapping in hum_exact_by_axis.items():
        assign_from_dict(canon_hum, mapping, axis_series == ax)

    def apply_regex(base: pd.Series, rules: List, humanize: bool, mask: pd.Series | None = None) -> None:
        if not rules:
            return
        mask = mask.fillna(False) if mask is not None else None
        for rx, rep in rules:
            available = translated.isna()
            if mask is not None:
                available &= mask
            if not available.any():
                break
            subset = base[available]
            if subset.empty:
                continue
            replaced = subset.str.replace(rx, rep, regex=True)
            changed = replaced != subset
            if not changed.any():
                continue
            values = replaced[changed]
            if humanize:
                values = values.map(_humanize_segment_name)
            translated.loc[values.index] = values

    apply_regex(raw_base_str, raw_regex, humanize=True)
    for ax, rules in raw_regex_by_axis.items():
        apply_regex(raw_base_str, rules, humanize=True, mask=axis_series == ax)

    apply_regex(human_str, hum_regex, humanize=False)
    for ax, rules in hum_regex_by_axis.items():
        apply_regex(human_str, rules, humanize=False, mask=axis_series == ax)

    df["Segment"] = translated.where(~translated.isna(), human)
    group_cols = [col for col in ["Axis", "AxisLabel", "Segment", "Year", "PeriodType"] if col in df.columns]
    df = df.groupby(group_cols, as_index=False)[["Revenue", "OpIncome"]].sum()

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

