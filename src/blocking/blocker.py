from __future__ import annotations

from collections import defaultdict
from time import perf_counter
import tracemalloc

import pandas as pd


def _build_index(df: pd.DataFrame, column: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)

    for idx, value in df[column].items():
        if pd.isna(value):
            continue

        value = str(value).strip()

        if value:
            index[value].append(idx)

    return dict(index)


def _get_blocking_keys(
    row: pd.Series,
    name_column: str = "name_core",
) -> set[str]:
    value = row.get(name_column)

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    keys = {f"name:{value}"}

    if len(value) >= 3:
        keys.add(f"name_prefix:{value[:3]}")

    return keys


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    *,
    name_column: str = "name_core",
    max_prefix_block_size: int = 1000,
) -> pd.DataFrame:
    """
    Generate candidate S1-S2 pairs using exact-name and prefix blocking.
    """

    required = {"entity_id", name_column}

    for name, df in (("source1", source1), ("source2", source2)):
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} is missing required columns: {sorted(missing)}"
            )

    s1 = source1[["entity_id", name_column]].copy()
    s2 = source2[["entity_id", name_column]].copy()

    s1[name_column] = s1[name_column].fillna("").astype(str).str.strip()
    s2[name_column] = s2[name_column].fillna("").astype(str).str.strip()

    s1["source1_index"] = s1.index
    s2["source2_index"] = s2.index

    s1["exact_key"] = s1[name_column].where(
        s1[name_column] != "", pd.NA
    )
    s2["exact_key"] = s2[name_column].where(
        s2[name_column] != "", pd.NA
    )

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
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> dict[str, float]:
    """
    Measure how many ground-truth matches were recovered by blocking.
    """
    candidate_required = {
        "source1_entity_id",
        "source2_entity_id",
    }

    ground_truth_required = {
        "source1_entity_id",
        "matched_entity_ids",
    }

    missing_candidates = candidate_required - set(candidates.columns)

    if missing_candidates:
        raise ValueError(
            f"candidates is missing required columns: "
            f"{sorted(missing_candidates)}"
        )

    missing_ground_truth = ground_truth_required - set(ground_truth.columns)

    if missing_ground_truth:
        raise ValueError(
            f"ground_truth is missing required columns: "
            f"{sorted(missing_ground_truth)}"
        )

    candidate_pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["source2_entity_id"],
        )
    )

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

        for source2_id in matched_ids.split(","):
            source2_id = source2_id.strip()

            if not source2_id:
                continue

            total_true_pairs += 1

            if (source1_id, source2_id) in candidate_pairs:
                recovered_true_pairs += 1

    if total_true_pairs:
        recall = recovered_true_pairs / total_true_pairs
    else:
        recall = 0.0

    return {
        "total_true_pairs": float(total_true_pairs),
        "recovered_true_pairs": float(recovered_true_pairs),
        "blocking_recall": float(recall),
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