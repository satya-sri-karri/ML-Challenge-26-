from __future__ import annotations

from collections import defaultdict
import os
import re
from time import perf_counter
import tracemalloc
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stopwords and noise filters for blocking keys
# ---------------------------------------------------------------------------

# Common street suffixes, unit types, directionals, and generic address words.
# These appear in thousands of addresses and do NOT distinguish entities when
# used alone; filtering them prevents inverted index block explosion.
ADDRESS_STOPWORDS = {
    "street", "road", "avenue", "drive", "lane", "court", "highway",
    "square", "place", "apartment", "floor", "number", "suite", "unit",
    "block", "building", "north", "south", "east", "west", "dr", "st",
    "rd", "ave", "ln", "ct", "hwy", "sq", "pl", "apt", "fl", "no", "ste",
    "near", "opp", "behind", "bldg", "plot", "sector", "phase", "cross",
    "main", "first", "second", "third", "1st", "2nd", "3rd", "house",
    "po", "box", "at", "by", "in", "of", "on"
}

# Generic corporate suffixes, conjunctions, and high-frequency business tokens.
# Filtered from token-level inverted indexing.
NAME_STOPWORDS = {
    "and", "the", "for", "with", "inc", "ltd", "llc", "corp", "co",
    "company", "enterprises", "services", "solutions", "group", "holdings",
    "international", "associates", "consulting", "industries", "management",
    "private", "limited", "pvt", "corporation", "incorporated"
}


# ---------------------------------------------------------------------------
# Blocking Key Extraction Functions
# ---------------------------------------------------------------------------

def _extract_name_keys(
    row: pd.Series | dict[str, Any],
    name_column: str = "name_core",
) -> set[str]:
    """
    Extract multi-signal name blocking keys from a row:
      1. Exact normalized name: name_core
      2. Domain name key: cleaned domain without TLD (for domain-style names)
      3. Prefix keys: 3-character prefix of name_core (and domain)
      4. Distinctive token keys: tokens >= 3 chars, excluding generic business stopwords
      5. Transliterated ASCII token keys: for cross-script Indian language matches
    """
    keys: set[str] = set()

    # (1) Primary name core
    raw_val = row.get(name_column)
    name_val = "" if pd.isna(raw_val) else str(raw_val).strip()

    if name_val:
        keys.add(f"name:{name_val}")
        if len(name_val) >= 3:
            keys.add(f"name_prefix:{name_val[:3]}")

    # (2) Domain-style name handling (e.g. 'aristosteel.com' -> 'aristosteel')
    is_domain = row.get("name_is_domain_style", False)
    domain_clean = row.get("name_domain_cleaned")
    if is_domain and pd.notna(domain_clean):
        dom_val = str(domain_clean).strip()
        if dom_val and dom_val != "None" and dom_val != "nan":
            keys.add(f"name:{dom_val}")
            if len(dom_val) >= 3:
                keys.add(f"name_prefix:{dom_val[:3]}")
            # Add domain tokens if segmented
            for dom_tok in dom_val.split():
                if len(dom_tok) >= 3 and dom_tok not in NAME_STOPWORDS:
                    keys.add(f"name_token:{dom_tok}")

    # (3) Distinctive name tokens
    tokens = row.get("name_tokens")
    if isinstance(tokens, (list, tuple, np.ndarray)):
        for tok in tokens:
            t = str(tok).lower().strip()
            if len(t) >= 3 and t not in NAME_STOPWORDS:
                keys.add(f"name_token:{t}")

    # (4) Transliterated ASCII fallback tokens (unidecode)
    name_ascii = row.get("name_ascii")
    if pd.notna(name_ascii):
        ascii_str = str(name_ascii).lower().strip()
        if ascii_str and ascii_str != name_val.lower():
            for tok in re.findall(r"[a-z0-9]+", ascii_str):
                if len(tok) >= 3 and tok not in NAME_STOPWORDS:
                    keys.add(f"name_token:{tok}")

    return keys


def _extract_address_keys(row: pd.Series | dict[str, Any]) -> set[str]:
    """
    Extract order-independent, typo-tolerant address blocking keys:
      1. Number + distinctive address word compound keys (e.g. '85_ticonderoga')
      2. Distinctive token pair keys (e.g. 'guthrie_surprise')

    Returns empty set if address is empty (safe for ~3.3% missing addresses).
    """
    keys: set[str] = set()

    # Empty address check
    if row.get("address_is_empty", False):
        return keys

    addr_raw = row.get("business_address")
    if pd.isna(addr_raw) or not str(addr_raw).strip():
        return keys

    # Extract numbers (house/street number, pin code, etc.)
    numbers = row.get("address_numbers")
    if not isinstance(numbers, (list, tuple, np.ndarray)):
        # Fallback if unnormalized
        numbers = re.findall(r"\d+(?:/\d+)?", str(addr_raw))

    # Extract tokens
    addr_tokens = row.get("address_tokens")
    if not isinstance(addr_tokens, (list, tuple, np.ndarray)):
        # Fallback if unnormalized
        addr_tokens = re.findall(r"[^\W\d_]+|\d+(?:/\d+)?", str(addr_raw).lower())

    distinctive_tokens: list[str] = [
        str(t).lower().strip()
        for t in addr_tokens
        if str(t).lower().strip() not in ADDRESS_STOPWORDS
        and not str(t).isdigit()
        and len(str(t).strip()) >= 3
    ]

    # (1) Compound keys: number + distinctive location word
    # Highly selective: '85' alone is huge; '85_ticonderoga' is unique!
    for num in numbers[:3]:
        num_str = str(num).strip()
        if num_str:
            for word in distinctive_tokens[:4]:
                keys.add(f"addr_num_tok:{num_str}_{word}")

    # (2) Order-independent token pair keys (for addresses with or without numbers)
    # Alphabetically sorted pairs: 'AZ, Surprise, Guthrie St' -> 'guthrie_surprise'
    if len(distinctive_tokens) >= 2:
        sorted_tokens = sorted(set(distinctive_tokens))
        limit = min(len(sorted_tokens), 4)
        for i in range(limit):
            for j in range(i + 1, limit):
                keys.add(f"addr_pair:{sorted_tokens[i]}_{sorted_tokens[j]}")

    return keys


def _get_blocking_keys(
    row: pd.Series | dict[str, Any],
    name_column: str = "name_core",
) -> set[str]:
    """
    Extract all blocking keys for a row, unioning name-based and address-based
    signals. If address columns are not present, generates name keys.
    """
    keys = _extract_name_keys(row, name_column=name_column)

    # Address keys are extracted if address information is available
    if "address_tokens" in row or "business_address" in row:
        keys.update(_extract_address_keys(row))

    return keys


def _build_index(df: pd.DataFrame, column: str) -> dict[str, list[int]]:
    """
    Build single-column inverted index (for backward compatibility).
    """
    index: dict[str, list[int]] = defaultdict(list)

    for idx, value in df[column].items():
        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            index[value].append(idx)

    return dict(index)


# ---------------------------------------------------------------------------
# Inverted Index Builder with Frequency Capping
# ---------------------------------------------------------------------------

class InvertedIndex:
    """
    Inverted index mapping blocking keys to target entity row indices.
    Supports per-key-type maximum block size pruning to guarantee sub-quadratic
    runtime and prevent candidate volume explosion.
    """

    def __init__(
        self,
        max_prefix_block_size: int = 1000,
        max_token_block_size: int = 500,
        max_addr_block_size: int = 500,
        max_exact_block_size: int = 2000,
    ) -> None:
        self.max_prefix_block_size = max_prefix_block_size
        self.max_token_block_size = max_token_block_size
        self.max_addr_block_size = max_addr_block_size
        self.max_exact_block_size = max_exact_block_size
        self.index: dict[str, list[int]] = defaultdict(list)
        self.pruned_index: dict[str, list[int]] = {}

    def add(self, key: str, idx: int) -> None:
        self.index[key].append(idx)

    def prune(self) -> None:
        """Prune keys whose block sizes exceed the threshold for their type."""
        pruned: dict[str, list[int]] = {}
        for key, post_list in self.index.items():
            count = len(post_list)
            if key.startswith("name_prefix:"):
                if count <= self.max_prefix_block_size:
                    pruned[key] = post_list
            elif key.startswith("name_token:"):
                if count <= self.max_token_block_size:
                    pruned[key] = post_list
            elif key.startswith("addr_"):
                if count <= self.max_addr_block_size:
                    pruned[key] = post_list
            elif key.startswith("name:"):
                if count <= self.max_exact_block_size:
                    pruned[key] = post_list
            else:
                if count <= self.max_prefix_block_size:
                    pruned[key] = post_list
        self.pruned_index = pruned

    def get_candidates(self, keys: Iterable[str]) -> set[int]:
        """Look up all target indices matching any of the provided keys."""
        candidates: set[int] = set()
        for k in keys:
            hits = self.pruned_index.get(k)
            if hits:
                candidates.update(hits)
        return candidates


# ---------------------------------------------------------------------------
# Candidate Generation (Public Interface)
# ---------------------------------------------------------------------------

def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    *,
    name_column: str = "name_core",
    max_prefix_block_size: int = 1000,
    max_token_block_size: int = 500,
    max_addr_block_size: int = 500,
    max_candidates_per_entity: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate candidate S1-S2 pairs using multi-signal blocking:
      (a) Exact name & prefix blocking on name_column
      (b) Distinctive token inverted index on name_tokens / name_domain_cleaned
      (c) Address compound keys on address_numbers / address_tokens (if present)

    Returns a DataFrame with columns:
      ["source1_index", "source2_index", "source1_entity_id", "source2_entity_id"]
    with duplicate candidate pairs removed.
    """
    required = {"entity_id", name_column}

    for name, df in (("source1", source1), ("source2", source2)):
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    has_address = "address_tokens" in source1.columns or "business_address" in source1.columns

    # Build target inverted index
    target_idx = InvertedIndex(
        max_prefix_block_size=max_prefix_block_size,
        max_token_block_size=max_token_block_size,
        max_addr_block_size=max_addr_block_size,
    )

    s2_ids = source2["entity_id"].values
    s1_ids = source1["entity_id"].values

    # Check if rich signals are available
    has_rich_signals = (
        has_address
        or "name_tokens" in source1.columns
        or "name_is_domain_style" in source1.columns
        or "name_ascii" in source1.columns
    )

    if not has_rich_signals:
        # Fast path for minimal DataFrames (e.g. existing unit tests with only name_core)
        s1 = source1[["entity_id", name_column]].copy()
        s2 = source2[["entity_id", name_column]].copy()

        s1[name_column] = s1[name_column].fillna("").astype(str).str.strip()
        s2[name_column] = s2[name_column].fillna("").astype(str).str.strip()

        s1["source1_index"] = s1.index
        s2["source2_index"] = s2.index

        s1["exact_key"] = s1[name_column].where(s1[name_column] != "", pd.NA)
        s2["exact_key"] = s2[name_column].where(s2[name_column] != "", pd.NA)

        exact = s1.merge(
            s2,
            on="exact_key",
            how="inner",
            suffixes=("_source1", "_source2"),
        )

        s1["prefix_key"] = s1[name_column].where(
            s1[name_column].str.len() >= 3
        ).str[:3]

        s2["prefix_key"] = s2[name_column].where(
            s2[name_column].str.len() >= 3
        ).str[:3]

        prefix_counts = s2["prefix_key"].value_counts()
        allowed_prefixes = prefix_counts[
            prefix_counts <= max_prefix_block_size
        ].index

        s1_prefix = s1[s1["prefix_key"].isin(allowed_prefixes)]
        s2_prefix = s2[s2["prefix_key"].isin(allowed_prefixes)]

        prefix = s1_prefix.merge(
            s2_prefix,
            on="prefix_key",
            how="inner",
            suffixes=("_source1", "_source2"),
        )

        result = pd.concat(
            [
                exact[
                    [
                        "source1_index",
                        "source2_index",
                        "entity_id_source1",
                        "entity_id_source2",
                    ]
                ].rename(
                    columns={
                        "entity_id_source1": "source1_entity_id",
                        "entity_id_source2": "source2_entity_id",
                    }
                ),
                prefix[
                    [
                        "source1_index",
                        "source2_index",
                        "entity_id_source1",
                        "entity_id_source2",
                    ]
                ].rename(
                    columns={
                        "entity_id_source1": "source1_entity_id",
                        "entity_id_source2": "source2_entity_id",
                    }
                ),
            ],
            ignore_index=True,
        )

        return result.drop_duplicates(
            subset=["source1_index", "source2_index"]
        ).reset_index(drop=True)

    # Rich multi-signal blocking path
    # Index source 2
    for s2_i in range(len(source2)):
        row_dict = source2.iloc[s2_i].to_dict()
        keys = _get_blocking_keys(row_dict, name_column=name_column)
        for k in keys:
            target_idx.add(k, s2_i)

    target_idx.prune()

    # Query source 1 against index
    s1_indices: list[int] = []
    s2_indices: list[int] = []
    s1_entity_ids: list[str] = []
    s2_entity_ids: list[str] = []

    for s1_i in range(len(source1)):
        row_dict = source1.iloc[s1_i].to_dict()
        keys = _get_blocking_keys(row_dict, name_column=name_column)
        cands = target_idx.get_candidates(keys)

        if max_candidates_per_entity and len(cands) > max_candidates_per_entity:
            cands = set(list(cands)[:max_candidates_per_entity])

        s1_id = s1_ids[s1_i]
        for c_idx in cands:
            s1_indices.append(s1_i)
            s2_indices.append(c_idx)
            s1_entity_ids.append(s1_id)
            s2_entity_ids.append(s2_ids[c_idx])

    result = pd.DataFrame(
        {
            "source1_index": s1_indices,
            "source2_index": s2_indices,
            "source1_entity_id": s1_entity_ids,
            "source2_entity_id": s2_entity_ids,
        }
    )

    return result.drop_duplicates(
        subset=["source1_index", "source2_index"]
    ).reset_index(drop=True)


def generate_candidates_multi_source(
    source1: pd.DataFrame,
    target_sources: dict[str, pd.DataFrame] | list[pd.DataFrame],
    *,
    name_column: str = "name_core",
    max_prefix_block_size: int = 1000,
    max_token_block_size: int = 500,
    max_addr_block_size: int = 500,
    max_candidates_per_entity: Optional[int] = None,
) -> dict[str, list[str]]:
    """
    Generate candidate entity pairs for Source 1 entities against multiple target
    sources (e.g. Source 2 and Source 3).

    Returns a dictionary mapping:
      source1_entity_id -> list of candidate_entity_ids (deduplicated, order preserved)
    Guarantees every Source 1 entity is present in the dictionary (empty list if singleton).
    """
    if isinstance(target_sources, list):
        target_dict = {f"target_{i}": df for i, df in enumerate(target_sources)}
    else:
        target_dict = target_sources

    # Initialize candidate map for all S1 entities (so singletons map to [])
    candidate_map: dict[str, set[str]] = {
        str(eid): set() for eid in source1["entity_id"]
    }

    for target_name, target_df in target_dict.items():
        if target_df is None or len(target_df) == 0:
            continue

        cands_df = generate_candidates(
            source1,
            target_df,
            name_column=name_column,
            max_prefix_block_size=max_prefix_block_size,
            max_token_block_size=max_token_block_size,
            max_addr_block_size=max_addr_block_size,
            max_candidates_per_entity=max_candidates_per_entity,
        )

        for s1_id, t_id in zip(cands_df["source1_entity_id"], cands_df["source2_entity_id"]):
            candidate_map[s1_id].add(t_id)

    # Convert sets to sorted lists
    return {
        s1_id: sorted(candidates)
        for s1_id, candidates in candidate_map.items()
    }


def write_candidate_pairs_tsv(
    candidate_map: dict[str, list[str]],
    all_source1_ids: list[str],
    output_path: str,
) -> int:
    """
    Write candidate pairs to candidate_pairs.tsv in the exact format required:
      source1_entity_id <TAB> candidate_entity_ids (comma-separated S2-/S3- ids)

    One row per Source 1 entity, empty string when no candidates found, no duplicate IDs.
    Returns total rows written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rows_written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        # Header
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_source1_ids:
            cands = candidate_map.get(s1_id, [])
            cands_str = ",".join(cands)
            f.write(f"{s1_id}\t{cands_str}\n")
            rows_written += 1

    return rows_written


# ---------------------------------------------------------------------------
# Metrics and Performance Measurement
# ---------------------------------------------------------------------------

def measure_candidate_volume(
    candidates: pd.DataFrame,
) -> dict[str, float]:
    """
    Measure the size and distribution of generated candidate pairs.
    """
    required = {"source1_index", "source2_index"}
    missing = required - set(candidates.columns)

    if missing:
        raise ValueError(
            f"candidates is missing required columns: {sorted(missing)}"
        )

    total_pairs = len(candidates)
    unique_source1 = candidates["source1_index"].nunique()
    unique_source2 = candidates["source2_index"].nunique()

    if unique_source1:
        avg_candidates_per_source1 = total_pairs / unique_source1
    else:
        avg_candidates_per_source1 = 0.0

    return {
        "total_candidate_pairs": float(total_pairs),
        "unique_source1_entities": float(unique_source1),
        "unique_source2_entities": float(unique_source2),
        "avg_candidates_per_source1": float(avg_candidates_per_source1),
    }


def measure_blocking_recall(
    candidates: pd.DataFrame | dict[str, list[str]],
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    """
    Measure how many ground-truth matches were recovered by blocking.
    Accepts either candidate pairs DataFrame or candidate dictionary mapping.
    """
    ground_truth_required = {
        "source1_entity_id",
        "matched_entity_ids",
    }
    missing_ground_truth = ground_truth_required - set(ground_truth.columns)
    if missing_ground_truth:
        raise ValueError(
            f"ground_truth is missing required columns: {sorted(missing_ground_truth)}"
        )

    if isinstance(candidates, pd.DataFrame):
        candidate_required = {"source1_entity_id", "source2_entity_id"}
        missing_candidates = candidate_required - set(candidates.columns)
        if missing_candidates:
            raise ValueError(
                f"candidates is missing required columns: {sorted(missing_candidates)}"
            )
        candidate_pairs = set(
            zip(
                candidates["source1_entity_id"],
                candidates["source2_entity_id"],
            )
        )
    else:
        # candidate dictionary: s1_id -> list of target_ids
        candidate_pairs = set()
        for s1_id, t_ids in candidates.items():
            for t_id in t_ids:
                candidate_pairs.add((s1_id, t_id))

    total_true_pairs = 0
    recovered_true_pairs = 0

    for _, row in ground_truth.iterrows():
        source1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            continue

        matched_ids = str(matched_ids).strip()
        if not matched_ids:
            continue

        for target_id in matched_ids.split(","):
            target_id = target_id.strip()
            if not target_id:
                continue

            total_true_pairs += 1
            if (source1_id, target_id) in candidate_pairs:
                recovered_true_pairs += 1

    recall = recovered_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0

    return {
        "total_true_pairs": float(total_true_pairs),
        "recovered_true_pairs": float(recovered_true_pairs),
        "blocking_recall": float(recall),
    }


def evaluate_blocking_metrics(
    candidate_map: dict[str, list[str]] | pd.DataFrame,
    ground_truth: pd.DataFrame,
    total_s1_count: int,
    total_target_count: int,
) -> dict[str, float]:
    """
    Compute comprehensive blocking metrics:
      - Recall ceiling (fraction of true matches present in candidates)
      - Reduction ratio (1 - candidates / all_possible_pairs)
      - Candidate pairs generated
      - True pairs recovered
    """
    recall_stats = measure_blocking_recall(candidate_map, ground_truth)

    if isinstance(candidate_map, pd.DataFrame):
        total_candidates = len(candidate_map)
    else:
        total_candidates = sum(len(c) for c in candidate_map.values())

    all_possible_pairs = float(total_s1_count * total_target_count)
    if all_possible_pairs > 0:
        reduction_ratio = 1.0 - (total_candidates / all_possible_pairs)
    else:
        reduction_ratio = 1.0

    return {
        "recall_ceiling": recall_stats["blocking_recall"],
        "reduction_ratio": float(reduction_ratio),
        "total_candidate_pairs": float(total_candidates),
        "all_possible_pairs": float(all_possible_pairs),
        "recovered_true_pairs": recall_stats["recovered_true_pairs"],
        "total_true_pairs": recall_stats["total_true_pairs"],
    }


def measure_blocking_performance(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    *,
    name_column: str = "name_core",
    max_prefix_block_size: int = 1000,
) -> dict[str, float]:
    """
    Measure blocking runtime and peak Python-tracked memory usage.
    """
    tracemalloc.start()
    start_time = perf_counter()

    candidates = generate_candidates(
        source1,
        source2,
        name_column=name_column,
        max_prefix_block_size=max_prefix_block_size,
    )

    elapsed_seconds = perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "runtime_seconds": float(elapsed_seconds),
        "peak_memory_mb": float(peak_memory / (1024 * 1024)),
        "candidate_pairs": float(len(candidates)),
    }