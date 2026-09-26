"""
Reusable dataset profiling.

Processes the raw TSVs in chunks (never loads a full 5M-row file into memory
at once) and produces a COMPACT summary -- counts, rates, sample values,
length statistics -- never a row-level dump. Used to produce
reports/eda_summary.{json,md}.

Usage:
    from src.normalization.profiling import profile_file
    summary = profile_file("dataset/train/train_source2.tsv", chunksize=200_000)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

_NONASCII_RE = re.compile(r"[^\x00-\x7F]")
_DOMAIN_HINT_RE = re.compile(r"\.(?:com|in|net|org)\b", flags=re.IGNORECASE)


@dataclass
class ColumnProfile:
    name: str
    dtype_observed: str = "str"
    n_missing: int = 0          # NaN in the raw file
    n_empty_string: int = 0     # "" after fillna -- distinct from NaN
    n_nonascii: int = 0
    example_values: list = field(default_factory=list)
    length_min: int | None = None
    length_max: int | None = None
    length_sum: int = 0
    length_count: int = 0
    _unique_sample: set = field(default_factory=set, repr=False)

    def length_mean(self):
        return self.length_sum / self.length_count if self.length_count else None


def _update_text_column(prof: ColumnProfile, series: pd.Series, sample_cap: int = 5,
                         unique_cap: int = 200_000) -> None:
    s = series.fillna("")
    is_empty = s.astype(str).str.len() == 0
    prof.n_empty_string += int(is_empty.sum())
    prof.n_missing += int(series.isna().sum())

    nonempty = s[~is_empty].astype(str)
    if len(nonempty):
        lengths = nonempty.str.len()
        prof.length_sum += int(lengths.sum())
        prof.length_count += int(lengths.count())
        prof.length_min = int(lengths.min()) if prof.length_min is None else min(prof.length_min, int(lengths.min()))
        prof.length_max = int(lengths.max()) if prof.length_max is None else max(prof.length_max, int(lengths.max()))
        prof.n_nonascii += int(nonempty.str.contains(_NONASCII_RE).sum())

        if len(prof.example_values) < sample_cap:
            need = sample_cap - len(prof.example_values)
            prof.example_values.extend(nonempty.head(need).tolist())

        if len(prof._unique_sample) < unique_cap:
            prof._unique_sample.update(nonempty.iloc[: unique_cap].tolist())


def profile_file(path: str, chunksize: int = 200_000, nrows: int | None = None) -> dict:
    """
    Stream-profile one TSV file. Returns a JSON-serializable dict:
        {
          "path": ..., "row_count": ..., "columns": {col: {...}, ...},
          "domain_style_name_count": ..., "domain_style_name_rate": ...,
        }
    Cardinality (`approx_unique`) is capped/approximate for very high-
    cardinality columns (e.g. business_address) to bound memory -- see
    unique_cap in _update_text_column. This is a deliberate accuracy/memory
    trade-off, documented here rather than silently exact-but-fragile.
    """
    profiles: dict[str, ColumnProfile] = {}
    row_count = 0
    domain_style_count = 0

    reader = pd.read_csv(
        path, sep="\t", dtype=str, chunksize=chunksize, keep_default_na=False, nrows=nrows
    )
    for chunk in reader:
        row_count += len(chunk)
        for col in chunk.columns:
            if col not in profiles:
                profiles[col] = ColumnProfile(name=col)
            _update_text_column(profiles[col], chunk[col])
        if "business_name" in chunk.columns:
            domain_style_count += int(
                chunk["business_name"].fillna("").astype(str).str.contains(_DOMAIN_HINT_RE).sum()
            )

    result = {
        "path": path,
        "row_count": row_count,
        "columns": {},
    }
    for name, prof in profiles.items():
        d = asdict(prof)
        d.pop("_unique_sample")
        d["approx_unique"] = len(prof._unique_sample)
        d["approx_unique_is_capped"] = len(prof._unique_sample) >= 200_000
        d["length_mean"] = prof.length_mean()
        result["columns"][name] = d

    if "business_name" in profiles:
        result["domain_style_name_count"] = domain_style_count
        result["domain_style_name_rate"] = domain_style_count / row_count if row_count else 0.0

    return result


def profile_ground_truth(path: str, chunksize: int = 200_000) -> dict:
    """
    Profile train_ground_truth.tsv specifically: singleton rate, match-count
    distribution, S2-vs-S3 split. Kept separate from profile_file() because
    its second column has different semantics (a comma-joined ID list, not a
    plain text field).
    """
    n_singleton = 0
    n_nonsingleton = 0
    total_matches = 0
    s2_matches = 0
    s3_matches = 0
    match_counts: list[int] = []

    reader = pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize, keep_default_na=False)
    for chunk in reader:
        ids = chunk["matched_entity_ids"].fillna("")
        empty_mask = ids.str.len() == 0
        n_singleton += int(empty_mask.sum())
        nonempty = ids[~empty_mask]
        n_nonsingleton += len(nonempty)
        for val in nonempty:
            parts = val.split(",")
            match_counts.append(len(parts))
            total_matches += len(parts)
            s2_matches += sum(1 for p in parts if p.startswith("S2-"))
            s3_matches += sum(1 for p in parts if p.startswith("S3-"))

    arr = np.array(match_counts) if match_counts else np.array([0])
    return {
        "path": path,
        "n_singleton": n_singleton,
        "n_nonsingleton": n_nonsingleton,
        "singleton_rate": n_singleton / (n_singleton + n_nonsingleton) if (n_singleton + n_nonsingleton) else 0.0,
        "total_matches": total_matches,
        "avg_matches_per_nonsingleton": float(arr.mean()),
        "max_matches": int(arr.max()),
        "s2_match_count": s2_matches,
        "s3_match_count": s3_matches,
    }


def save_summary(summary: dict, out_path_json: str) -> None:
    """Write a compact JSON summary. Never write row-level data here."""
    with open(out_path_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
