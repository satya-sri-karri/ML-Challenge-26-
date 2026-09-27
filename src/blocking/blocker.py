from __future__ import annotations

from collections import defaultdict
import gc
import os
import re
from time import perf_counter
import tracemalloc
from typing import Any, Callable, Iterable, Optional, TextIO

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
# Helper: Fast, lightweight row field access for dict, Series, or namedtuple
# ---------------------------------------------------------------------------

def _get_val(row: Any, key: str, default: Any = None) -> Any:
    """Safely get field from dict, Series, or namedtuple row without overhead."""
    if isinstance(row, dict):
        return row.get(key, default)
    if hasattr(row, key):
        val = getattr(row, key)
        return default if val is None else val
    if hasattr(row, "get"):
        return row.get(key, default)
    return default


# ---------------------------------------------------------------------------
# Blocking Key Extraction Functions
# ---------------------------------------------------------------------------

def _extract_name_keys(
    row: pd.Series | dict[str, Any] | Any,
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
    raw_val = _get_val(row, name_column)
    name_val = "" if pd.isna(raw_val) else str(raw_val).strip()

    if name_val:
        keys.add(f"name:{name_val}")
        if len(name_val) >= 3:
            keys.add(f"name_prefix:{name_val[:3]}")

    # (2) Domain-style name handling (e.g. 'aristosteel.com' -> 'aristosteel')
    is_domain = _get_val(row, "name_is_domain_style", False)
    domain_clean = _get_val(row, "name_domain_cleaned")
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
    tokens = _get_val(row, "name_tokens")
    if isinstance(tokens, (list, tuple, np.ndarray)):
        for tok in tokens:
            t = str(tok).lower().strip()
            if len(t) >= 3 and t not in NAME_STOPWORDS:
                keys.add(f"name_token:{t}")

    # (4) Transliterated ASCII fallback tokens (unidecode)
    name_ascii = _get_val(row, "name_ascii")
    if pd.notna(name_ascii):
        ascii_str = str(name_ascii).lower().strip()
        if ascii_str and ascii_str != name_val.lower():
            for tok in re.findall(r"[a-z0-9]+", ascii_str):
                if len(tok) >= 3 and tok not in NAME_STOPWORDS:
                    keys.add(f"name_token:{tok}")

    return keys


def _extract_address_keys(row: pd.Series | dict[str, Any] | Any) -> set[str]:
    """
    Extract order-independent, typo-tolerant address blocking keys:
      1. Number + distinctive address word compound keys (e.g. '85_ticonderoga')
      2. Distinctive token pair keys (e.g. 'guthrie_surprise')

    Returns empty set if address is empty (safe for ~3.3% missing addresses).
    """
    keys: set[str] = set()

    # Empty address check
    if _get_val(row, "address_is_empty", False):
        return keys

    addr_raw = _get_val(row, "business_address")
    if pd.isna(addr_raw) or not str(addr_raw).strip():
        return keys

    # Extract numbers (house/street number, pin code, etc.)
    numbers = _get_val(row, "address_numbers")
    if not isinstance(numbers, (list, tuple, np.ndarray)):
        # Fallback if unnormalized
        numbers = re.findall(r"\d+(?:/\d+)?", str(addr_raw))

    # Extract tokens
    addr_tokens = _get_val(row, "address_tokens")
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
    row: pd.Series | dict[str, Any] | Any,
    name_column: str = "name_core",
) -> set[str]:
    """
    Extract all blocking keys for a row, unioning name-based and address-based
    signals. If address columns are not present, generates name keys.
    """
    keys = _extract_name_keys(row, name_column=name_column)

    # Address keys are extracted if address information is available
    if (
        (isinstance(row, dict) and ("address_tokens" in row or "business_address" in row))
        or hasattr(row, "address_tokens")
        or hasattr(row, "business_address")
    ):
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
        self.index.clear()
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
    for s2_i, row_obj in enumerate(source2.itertuples(index=False)):
        keys = _get_blocking_keys(row_obj, name_column=name_column)
        for k in keys:
            target_idx.add(k, s2_i)

    target_idx.prune()

    # Query source 1 against index
    s1_indices: list[int] = []
    s2_indices: list[int] = []
    s1_entity_ids: list[str] = []
    s2_entity_ids: list[str] = []

    for s1_i, row_obj in enumerate(source1.itertuples(index=False)):
        keys = _get_blocking_keys(row_obj, name_column=name_column)
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


def write_candidate_pairs_streaming(
    chunk_candidate_map: dict[str, list[str]],
    file_obj: TextIO,
    all_chunk_s1_ids: Optional[list[str]] = None,
    write_header: bool = False,
) -> int:
    """
    Write candidate pairs incrementally for a chunk to an open file handle.
    If all_chunk_s1_ids is provided, writes in that exact order (mapping missing to empty string).
    Otherwise, writes the entries in chunk_candidate_map.
    """
    rows_written = 0
    if write_header:
        file_obj.write("source1_entity_id\tcandidate_entity_ids\n")
    order = all_chunk_s1_ids if all_chunk_s1_ids is not None else list(chunk_candidate_map.keys())
    for s1_id in order:
        cands = chunk_candidate_map.get(s1_id, [])
        cands_str = ",".join(cands)
        file_obj.write(f"{s1_id}\t{cands_str}\n")
        rows_written += 1
    return rows_written


def _ensure_normalized(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Ensure DataFrame has required normalized columns, normalizing if needed."""
    if "name_core" in df.columns and "address_tokens" in df.columns:
        return df
    from src.normalization.normalizer import normalize_source
    return normalize_source(df, source_name=source_name)


def _iter_source1_chunks(
    source1_input: str | pd.DataFrame,
    chunk_size: int = 50000,
) -> Iterable[pd.DataFrame]:
    """Iterate over Source 1 entities in manageable chunks."""
    if isinstance(source1_input, pd.DataFrame):
        for start in range(0, len(source1_input), chunk_size):
            yield source1_input.iloc[start : start + chunk_size]
    elif str(source1_input).endswith(".parquet"):
        df = pd.read_parquet(source1_input)
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start : start + chunk_size]
    else:
        for chunk in pd.read_csv(
            str(source1_input),
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
        ):
            yield chunk


def build_target_index(
    target_data: str | pd.DataFrame,
    source_name: str = "target",
    *,
    name_column: str = "name_core",
    chunk_size: int = 100000,
    max_prefix_block_size: int = 1000,
    max_token_block_size: int = 500,
    max_addr_block_size: int = 500,
    max_exact_block_size: int = 2000,
) -> tuple[InvertedIndex, list[str]]:
    """
    Build a memory-efficient inverted index from target data (DataFrame or file path).
    When a file path is provided, reads in chunks to keep memory footprint minimal.
    Only retains necessary blocking columns during chunk processing.
    """
    target_idx = InvertedIndex(
        max_prefix_block_size=max_prefix_block_size,
        max_token_block_size=max_token_block_size,
        max_addr_block_size=max_addr_block_size,
        max_exact_block_size=max_exact_block_size,
    )
    target_ids: list[str] = []

    if isinstance(target_data, pd.DataFrame):
        df = _ensure_normalized(target_data, source_name=source_name)
        target_ids = df["entity_id"].astype(str).tolist()
        for idx, row in enumerate(df.itertuples(index=False)):
            keys = _get_blocking_keys(row, name_column=name_column)
            for k in keys:
                target_idx.add(k, idx)
        target_idx.prune()
        return target_idx, target_ids

    # Path to file (TSV or Parquet)
    path = str(target_data)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        return build_target_index(
            df,
            source_name=source_name,
            name_column=name_column,
            max_prefix_block_size=max_prefix_block_size,
            max_token_block_size=max_token_block_size,
            max_addr_block_size=max_addr_block_size,
            max_exact_block_size=max_exact_block_size,
        )

    # TSV chunked reader
    offset = 0
    for chunk_df in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    ):
        chunk_df = _ensure_normalized(chunk_df, source_name=source_name)
        chunk_ids = chunk_df["entity_id"].astype(str).tolist()
        target_ids.extend(chunk_ids)

        for i, row in enumerate(chunk_df.itertuples(index=False)):
            keys = _get_blocking_keys(row, name_column=name_column)
            for k in keys:
                target_idx.add(k, offset + i)

        offset += len(chunk_df)
        del chunk_df
        gc.collect()

    target_idx.prune()
    gc.collect()
    return target_idx, target_ids


def query_source1_chunk(
    s1_chunk_df: pd.DataFrame,
    target_idx: InvertedIndex,
    target_ids: list[str],
    *,
    name_column: str = "name_core",
    max_candidates_per_entity: Optional[int] = None,
) -> dict[str, list[str]]:
    """
    Query a chunk of Source 1 entities against a target InvertedIndex.
    Returns mapping: s1_entity_id -> sorted unique list of candidate target entity IDs.
    """
    candidate_map: dict[str, list[str]] = {}
    s1_ids = s1_chunk_df["entity_id"].astype(str).tolist()

    for idx, row in enumerate(s1_chunk_df.itertuples(index=False)):
        s1_id = s1_ids[idx]
        keys = _get_blocking_keys(row, name_column=name_column)
        hits = target_idx.get_candidates(keys)

        if max_candidates_per_entity and len(hits) > max_candidates_per_entity:
            hits = set(list(hits)[:max_candidates_per_entity])

        if hits:
            cand_ids = sorted({target_ids[c] for c in hits if c < len(target_ids)})
            candidate_map[s1_id] = cand_ids
        else:
            candidate_map[s1_id] = []

    return candidate_map


def _iter_target_partitions(
    target_data: str | pd.DataFrame,
    source_name: str,
    partition_size: Optional[int] = 250000,
) -> Iterable[tuple[str, pd.DataFrame]]:
    """
    Yield target data in bounded partitions (DataFrames) so peak indexing memory
    is strictly bounded below system limits.
    Each item yielded is (partition_label, partition_df).
    """
    if isinstance(target_data, pd.DataFrame):
        df = target_data
        if partition_size is None or len(df) <= partition_size:
            yield (source_name, df)
        else:
            part_idx = 0
            for start in range(0, len(df), partition_size):
                yield (f"{source_name}_p{part_idx}", df.iloc[start : start + partition_size])
                part_idx += 1
    elif str(target_data).endswith(".parquet"):
        df = pd.read_parquet(target_data)
        if partition_size is None or len(df) <= partition_size:
            yield (source_name, df)
        else:
            part_idx = 0
            for start in range(0, len(df), partition_size):
                yield (f"{source_name}_p{part_idx}", df.iloc[start : start + partition_size])
                part_idx += 1
    else:
        # TSV file path
        if partition_size is None:
            df = pd.read_csv(str(target_data), sep="\t", dtype=str, keep_default_na=False)
            yield (source_name, df)
        else:
            part_idx = 0
            for chunk_df in pd.read_csv(
                str(target_data),
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=partition_size,
            ):
                yield (f"{source_name}_p{part_idx}", chunk_df)
                part_idx += 1


def stream_candidate_pairs(
    source1_input: str | pd.DataFrame,
    target_inputs: list[tuple[str, str | pd.DataFrame]] | dict[str, str | pd.DataFrame],
    output_path: str,
    *,
    chunk_size: int = 50000,
    target_chunk_size: int = 100000,
    target_partition_size: Optional[int] = 250000,
    name_column: str = "name_core",
    max_prefix_block_size: int = 1000,
    max_token_block_size: int = 500,
    max_addr_block_size: int = 500,
    max_exact_block_size: int = 2000,
    max_candidates_per_entity: Optional[int] = None,
) -> dict[str, Any]:
    """
    Stream Source 1 in chunks against target sources sequentially in bounded partitions.
    Guarantees:
      - Memory safe: Only one target partition index and one S1 chunk in RAM at a time.
      - Exactly ONE row per Source 1 entity in output.
      - candidate_entity_ids is the unique deduplicated union of all target candidates.
      - Intermediate disk-backed temporary files are cleaned up automatically.
    """
    t_start = perf_counter()
    tracemalloc.start()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Normalize target_inputs to list of (source_name, target_obj)
    if isinstance(target_inputs, dict):
        targets = [(k, v) for k, v in target_inputs.items() if v is not None]
    else:
        targets = [(k, v) for k, v in target_inputs if v is not None]

    if not targets:
        # Edge case: No target sources provided. Write empty rows for all S1.
        total_s1 = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write("source1_entity_id\tcandidate_entity_ids\n")
            for chunk_df in _iter_source1_chunks(source1_input, chunk_size):
                for s1_id in chunk_df["entity_id"].astype(str):
                    out_f.write(f"{s1_id}\t\n")
                    total_s1 += 1
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "total_source1_entities": total_s1,
            "total_candidate_pairs": 0,
            "entities_with_candidates": 0,
            "singleton_count": total_s1,
            "singleton_rate": 1.0,
            "avg_candidates_per_entity": 0.0,
            "runtime_seconds": perf_counter() - t_start,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    temp_files: list[str] = []
    temp_prev: Optional[str] = None
    pass_num = 0

    try:
        for src_name, target_obj in targets:
            for part_label, part_df in _iter_target_partitions(target_obj, src_name, target_partition_size):
                curr_temp = output_path + f".tmp_pass{pass_num}"
                temp_files.append(curr_temp)

                # Build index for this partition only
                target_idx, target_ids = build_target_index(
                    part_df,
                    source_name=src_name,
                    name_column=name_column,
                    chunk_size=target_chunk_size,
                    max_prefix_block_size=max_prefix_block_size,
                    max_token_block_size=max_token_block_size,
                    max_addr_block_size=max_addr_block_size,
                    max_exact_block_size=max_exact_block_size,
                )
                del part_df
                gc.collect()

                if pass_num == 0:
                    with open(curr_temp, "w", encoding="utf-8") as out_f:
                        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
                        for s1_chunk in _iter_source1_chunks(source1_input, chunk_size):
                            s1_norm = _ensure_normalized(s1_chunk, "source1")
                            cand_map = query_source1_chunk(
                                s1_norm,
                                target_idx,
                                target_ids,
                                name_column=name_column,
                                max_candidates_per_entity=max_candidates_per_entity,
                            )
                            s1_ids = s1_norm["entity_id"].astype(str).tolist()
                            for s1_id in s1_ids:
                                cands = cand_map.get(s1_id, [])
                                out_f.write(f"{s1_id}\t{','.join(cands)}\n")
                            del s1_chunk, s1_norm, cand_map
                            gc.collect()
                else:
                    with open(temp_prev, "r", encoding="utf-8") as in_prev, open(curr_temp, "w", encoding="utf-8") as out_curr:
                        in_prev.readline()  # Skip header
                        out_curr.write("source1_entity_id\tcandidate_entity_ids\n")
                        for s1_chunk in _iter_source1_chunks(source1_input, chunk_size):
                            s1_norm = _ensure_normalized(s1_chunk, "source1")
                            cand_map = query_source1_chunk(
                                s1_norm,
                                target_idx,
                                target_ids,
                                name_column=name_column,
                                max_candidates_per_entity=max_candidates_per_entity,
                            )
                            s1_ids = s1_norm["entity_id"].astype(str).tolist()

                            for s1_id in s1_ids:
                                prev_line = in_prev.readline().rstrip("\r\n")
                                if not prev_line:
                                    prev_cands: list[str] = []
                                else:
                                    parts = prev_line.split("\t", 1)
                                    prev_s1_id = parts[0]
                                    if prev_s1_id != s1_id:
                                        raise ValueError(
                                            f"Streaming alignment mismatch: expected {s1_id}, found {prev_s1_id}"
                                        )
                                    prev_cands = parts[1].split(",") if len(parts) > 1 and parts[1] else []

                                new_cands = cand_map.get(s1_id, [])
                                if prev_cands and new_cands:
                                    merged = sorted(set(prev_cands).union(new_cands))
                                elif prev_cands:
                                    merged = prev_cands
                                else:
                                    merged = new_cands

                                out_curr.write(f"{s1_id}\t{','.join(merged)}\n")

                            del s1_chunk, s1_norm, cand_map
                            gc.collect()

                    # Clean up the previous temp file
                    if os.path.exists(temp_prev):
                        os.remove(temp_prev)
                        if temp_prev in temp_files:
                            temp_files.remove(temp_prev)

                del target_idx, target_ids
                gc.collect()

                temp_prev = curr_temp
                pass_num += 1

        total_s1 = 0
        total_pairs = 0
        entities_with_cands = 0

        if temp_prev is not None and os.path.exists(temp_prev):
            with open(temp_prev, "r", encoding="utf-8") as f:
                f.readline()  # header
                for line in f:
                    line_str = line.rstrip("\r\n")
                    if not line_str:
                        continue
                    parts = line_str.split("\t", 1)
                    cand_str = parts[1] if len(parts) > 1 else ""
                    cands = cand_str.split(",") if cand_str else []
                    total_s1 += 1
                    total_pairs += len(cands)
                    if cands:
                        entities_with_cands += 1

            if os.path.exists(output_path):
                os.remove(output_path)
            os.replace(temp_prev, output_path)
            if temp_prev in temp_files:
                temp_files.remove(temp_prev)

    finally:
        # Guarantee all temporary files are cleaned up even if interrupted
        for tf in temp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except OSError:
                    pass

    t_runtime = perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    singletons = total_s1 - entities_with_cands
    singleton_rate = (singletons / max(1, total_s1)) if total_s1 > 0 else 0.0
    avg_cands = (total_pairs / max(1, total_s1)) if total_s1 > 0 else 0.0

    return {
        "total_source1_entities": total_s1,
        "total_candidate_pairs": total_pairs,
        "entities_with_candidates": entities_with_cands,
        "singleton_count": singletons,
        "singleton_rate": singleton_rate,
        "avg_candidates_per_entity": avg_cands,
        "runtime_seconds": t_runtime,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }


def _iter_target_chunks(
    target_data: str | pd.DataFrame,
    chunk_size: int = 100000,
) -> Iterable[pd.DataFrame]:
    """Iterate over target data in manageable chunks."""
    if isinstance(target_data, pd.DataFrame):
        for start in range(0, len(target_data), chunk_size):
            yield target_data.iloc[start : start + chunk_size]
    elif str(target_data).endswith(".parquet"):
        df = pd.read_parquet(target_data)
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start : start + chunk_size]
    else:
        for chunk in pd.read_csv(
            str(target_data),
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
        ):
            yield chunk


def find_target_pruned_keys(
    target_data: str | pd.DataFrame,
    source_name: str = "target",
    *,
    name_column: str = "name_core",
    chunk_size: int = 100000,
    max_prefix_block_size: int = 1000,
    max_token_block_size: int = 500,
    max_addr_block_size: int = 500,
    max_exact_block_size: int = 2000,
) -> set[str]:
    """
    Find keys in target data that exceed maximum block size thresholds globally.
    """
    key_counts: dict[str, int] = defaultdict(int)

    for chunk_df in _iter_target_chunks(target_data, chunk_size=chunk_size):
        chunk_norm = _ensure_normalized(chunk_df, source_name=source_name)
        for row in chunk_norm.itertuples(index=False):
            keys = _get_blocking_keys(row, name_column=name_column)
            for k in keys:
                key_counts[k] += 1
        del chunk_df, chunk_norm

    pruned: set[str] = set()
    for key, count in key_counts.items():
        if key.startswith("name_prefix:"):
            if count > max_prefix_block_size:
                pruned.add(key)
        elif key.startswith("name_token:"):
            if count > max_token_block_size:
                pruned.add(key)
        elif key.startswith("addr_"):
            if count > max_addr_block_size:
                pruned.add(key)
        elif key.startswith("name:"):
            if count > max_exact_block_size:
                pruned.add(key)
        else:
            if count > max_prefix_block_size:
                pruned.add(key)

    return pruned


def stream_candidates_s1_chunked(
    source1_input: str | pd.DataFrame,
    target_inputs: list[tuple[str, str | pd.DataFrame]] | dict[str, str | pd.DataFrame],
    output_path: str,
    *,
    s1_chunk_size: int = 50000,
    target_chunk_size: int = 100000,
    name_column: str = "name_core",
    max_prefix_block_size: int = 1000,
    max_token_block_size: int = 500,
    max_addr_block_size: int = 500,
    max_exact_block_size: int = 2000,
    max_candidates_per_entity: Optional[int] = None,
    target_pruned_keys: Optional[dict[str, set[str]]] = None,
    precompute_pruned_keys: bool = True,
) -> dict[str, Any]:
    """
    Stream Source 1 in chunks, building an in-memory index for each S1 chunk and
    streaming target datasets (S2, S3) sequentially against it.

    Guarantees:
      - Memory safe: Only one S1 chunk index + candidate sets in RAM.
      - Exactly ONE row per Source 1 entity in output.
      - candidate_entity_ids is the unique deduplicated union of all target candidates.
      - Bitwise exact equivalence with batch blocking when target_pruned_keys is used.
      - Zero intermediate disk file rewriting.
    """
    t_start = perf_counter()
    tracemalloc.start()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if isinstance(target_inputs, dict):
        targets = [(k, v) for k, v in target_inputs.items() if v is not None]
    else:
        targets = [(k, v) for k, v in target_inputs if v is not None]

    if not targets:
        total_s1 = 0
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write("source1_entity_id\tcandidate_entity_ids\n")
            for chunk_df in _iter_source1_chunks(source1_input, s1_chunk_size):
                for s1_id in chunk_df["entity_id"].astype(str):
                    out_f.write(f"{s1_id}\t\n")
                    total_s1 += 1
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "total_source1_entities": total_s1,
            "total_candidate_pairs": 0,
            "entities_with_candidates": 0,
            "singleton_count": total_s1,
            "singleton_rate": 1.0,
            "avg_candidates_per_entity": 0.0,
            "runtime_seconds": perf_counter() - t_start,
            "peak_memory_mb": peak_mem / (1024 * 1024),
        }

    # Precompute pruned keys per target if requested and not provided
    pruned_by_src: dict[str, set[str]] = {}
    if target_pruned_keys is not None:
        pruned_by_src = target_pruned_keys
    elif precompute_pruned_keys:
        for src_name, target_obj in targets:
            pruned_by_src[src_name] = find_target_pruned_keys(
                target_obj,
                source_name=src_name,
                name_column=name_column,
                chunk_size=target_chunk_size,
                max_prefix_block_size=max_prefix_block_size,
                max_token_block_size=max_token_block_size,
                max_addr_block_size=max_addr_block_size,
                max_exact_block_size=max_exact_block_size,
            )

    temp_out = output_path + ".tmp_s1chunk"
    total_s1 = 0
    total_pairs = 0
    entities_with_cands = 0

    try:
        with open(temp_out, "w", encoding="utf-8") as out_f:
            out_f.write("source1_entity_id\tcandidate_entity_ids\n")

            for s1_chunk in _iter_source1_chunks(source1_input, s1_chunk_size):
                s1_norm = _ensure_normalized(s1_chunk, "source1")
                s1_ids = s1_norm["entity_id"].astype(str).tolist()

                # Build inverted index for this S1 chunk
                s1_idx: dict[str, list[int]] = defaultdict(list)
                for i, row in enumerate(s1_norm.itertuples(index=False)):
                    keys = _get_blocking_keys(row, name_column=name_column)
                    for k in keys:
                        s1_idx[k].append(i)

                # Initialize candidate sets for every S1 entity in this chunk
                cand_map: dict[str, set[str]] = {sid: set() for sid in s1_ids}

                # Stream each target source against this S1 chunk index
                for src_name, target_obj in targets:
                    src_pruned = pruned_by_src.get(src_name, set())

                    for target_chunk in _iter_target_chunks(target_obj, target_chunk_size):
                        target_norm = _ensure_normalized(target_chunk, source_name=src_name)
                        for target_row in target_norm.itertuples(index=False):
                            t_id = str(target_row.entity_id)
                            t_keys = _get_blocking_keys(target_row, name_column=name_column)

                            # Only match on keys that are not pruned in this target source
                            for k in t_keys:
                                if k not in src_pruned:
                                    hits = s1_idx.get(k)
                                    if hits:
                                        for h_idx in hits:
                                            cand_map[s1_ids[h_idx]].add(t_id)

                        del target_chunk, target_norm

                # Write this S1 chunk directly to temp_out
                for sid in s1_ids:
                    cands = cand_map[sid]
                    if max_candidates_per_entity and len(cands) > max_candidates_per_entity:
                        cand_list = sorted(cands)[:max_candidates_per_entity]
                    else:
                        cand_list = sorted(cands)

                    cands_count = len(cand_list)
                    total_s1 += 1
                    total_pairs += cands_count
                    if cands_count > 0:
                        entities_with_cands += 1

                    out_f.write(f"{sid}\t{','.join(cand_list)}\n")

                del s1_idx, cand_map, s1_chunk, s1_norm, s1_ids
                gc.collect()

        if os.path.exists(output_path):
            os.remove(output_path)
        os.replace(temp_out, output_path)

    finally:
        if os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except OSError:
                pass

    t_runtime = perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    singletons = total_s1 - entities_with_cands
    singleton_rate = (singletons / max(1, total_s1)) if total_s1 > 0 else 0.0
    avg_cands = (total_pairs / max(1, total_s1)) if total_s1 > 0 else 0.0

    return {
        "total_source1_entities": total_s1,
        "total_candidate_pairs": total_pairs,
        "entities_with_candidates": entities_with_cands,
        "singleton_count": singletons,
        "singleton_rate": singleton_rate,
        "avg_candidates_per_entity": avg_cands,
        "runtime_seconds": t_runtime,
        "peak_memory_mb": peak_mem / (1024 * 1024),
    }


def evaluate_candidate_file_recall(
    candidate_file_path: str,
    ground_truth: str | pd.DataFrame,
    total_target_count: int,
) -> dict[str, float]:
    """
    Stream a candidate_pairs.tsv file and evaluate recall against ground truth
    without loading all candidate pairs into memory.
    """
    if isinstance(ground_truth, str):
        gt_df = pd.read_csv(ground_truth, sep="\t", dtype=str, keep_default_na=False)
    else:
        gt_df = ground_truth

    gt_map: dict[str, set[str]] = {}
    total_true_pairs = 0
    for row in gt_df.itertuples(index=False):
        s1_id = str(row.source1_entity_id).strip()
        matched = str(getattr(row, "matched_entity_ids", "")).strip()
        if matched and matched != "nan" and matched != "None":
            m_set = {m.strip() for m in matched.split(",") if m.strip()}
            gt_map[s1_id] = m_set
            total_true_pairs += len(m_set)

    total_s1 = 0
    total_cands = 0
    recovered_true_pairs = 0

    with open(candidate_file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str:
                continue
            parts = line_str.split("\t", 1)
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            total_s1 += 1

            if cand_str:
                cands = set(cand_str.split(","))
                total_cands += len(cands)
            else:
                cands = set()

            true_matches = gt_map.get(s1_id)
            if true_matches:
                recovered_true_pairs += len(true_matches.intersection(cands))

    recall_ceiling = (
        recovered_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    )
    all_possible = float(total_s1 * total_target_count) if total_target_count > 0 else 1.0
    reduction_ratio = 1.0 - (total_cands / all_possible) if all_possible > 0 else 0.0

    return {
        "recall_ceiling": float(recall_ceiling),
        "reduction_ratio": float(reduction_ratio),
        "total_candidate_pairs": float(total_cands),
        "all_possible_pairs": float(all_possible),
        "total_true_pairs": float(total_true_pairs),
        "recovered_true_pairs": float(recovered_true_pairs),
        "total_source1_entities": float(total_s1),
    }


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