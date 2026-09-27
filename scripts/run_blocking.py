#!/usr/bin/env python3
"""
Run multi-signal blocking / candidate generation and write candidate_pairs.tsv.

Usage:
  python scripts/run_blocking.py --dataset-dir /path/to/dataset --split train
  python scripts/run_blocking.py --source1 path/to/s1.tsv --source2 path/to/s2.tsv --source3 path/to/s3.tsv --output output/candidate_pairs.tsv
"""
from __future__ import annotations

import argparse
import os
import sys
from time import perf_counter
import tracemalloc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.blocking.blocker import (
    evaluate_blocking_metrics,
    generate_candidates_multi_source,
    measure_candidate_volume,
    write_candidate_pairs_tsv,
)
from src.normalization.normalizer import normalize_source


def load_or_normalize(path: str, source_name: str) -> pd.DataFrame:
    """
    Load data from parquet (already normalized) or TSV (applies normalize_source).
    """
    print(f"Loading {source_name} from {path} ...", file=sys.stderr)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)

    # TSV format
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    # Check if already normalized
    if "name_core" in df.columns and "address_tokens" in df.columns:
        return df
    print(f"Normalizing {source_name} ({len(df):,} rows) ...", file=sys.stderr)
    return normalize_source(df, source_name=source_name)


def main():
    parser = argparse.ArgumentParser(description="Multi-signal blocking and candidate pair generation.")
    parser.add_argument("--dataset-dir", default=None, help="Root path to local dataset/ directory")
    parser.add_argument("--split", choices=["train", "test"], default="train", help="Dataset split")
    parser.add_argument("--source1", default=None, help="Explicit path to Source 1 TSV/parquet")
    parser.add_argument("--source2", default=None, help="Explicit path to Source 2 TSV/parquet")
    parser.add_argument("--source3", default=None, help="Explicit path to Source 3 TSV/parquet")
    parser.add_argument("--ground-truth", default=None, help="Explicit path to train_ground_truth.tsv")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "output", "candidate_pairs.tsv"))
    parser.add_argument("--report", default=os.path.join(os.path.dirname(__file__), "..", "reports", "blocking_recall_report.md"))
    parser.add_argument("--val-fraction", type=float, default=0.20, help="Fraction of S1 entities held out for validation")
    parser.add_argument("--val-seed", type=int, default=42, help="Random seed for validation split")
    parser.add_argument("--max-prefix-block-size", type=int, default=1000)
    parser.add_argument("--max-token-block-size", type=int, default=500)
    parser.add_argument("--max-addr-block-size", type=int, default=500)
    parser.add_argument("--max-candidates-per-entity", type=int, default=None)

    args = parser.parse_args()

    # Determine input paths
    s1_path = args.source1
    s2_path = args.source2
    s3_path = args.source3
    gt_path = args.ground_truth

    if args.dataset_dir:
        split_dir = os.path.join(args.dataset_dir, args.split)
        if not s1_path:
            # Check normalized parquet first, then TSV
            norm_p = os.path.join(args.dataset_dir, "normalized", f"{args.split}_source1.parquet")
            tsv_p = os.path.join(split_dir, f"{args.split}_source1.tsv")
            s1_path = norm_p if os.path.exists(norm_p) else tsv_p

        if not s2_path:
            norm_p = os.path.join(args.dataset_dir, "normalized", f"{args.split}_source2.parquet")
            tsv_p = os.path.join(split_dir, f"{args.split}_source2.tsv")
            s2_path = norm_p if os.path.exists(norm_p) else tsv_p

        if not s3_path:
            norm_p = os.path.join(args.dataset_dir, "normalized", f"{args.split}_source3.parquet")
            tsv_p = os.path.join(split_dir, f"{args.split}_source3.tsv")
            s3_path = norm_p if os.path.exists(norm_p) else tsv_p

        if not gt_path and args.split == "train":
            candidate_gt = os.path.join(split_dir, "train_ground_truth.tsv")
            if os.path.exists(candidate_gt):
                gt_path = candidate_gt

    if not s1_path or not os.path.exists(s1_path):
        print(f"Error: Source 1 file not found at {s1_path}", file=sys.stderr)
        sys.exit(1)

    tracemalloc.start()
    t_start = perf_counter()

    # Load sources
    s1_df = load_or_normalize(s1_path, "source1")
    s2_df = load_or_normalize(s2_path, "source2") if s2_path and os.path.exists(s2_path) else None
    s3_df = load_or_normalize(s3_path, "source3") if s3_path and os.path.exists(s3_path) else None

    target_sources = {}
    if s2_df is not None:
        target_sources["source2"] = s2_df
    if s3_df is not None:
        target_sources["source3"] = s3_df

    total_target_rows = sum(len(df) for df in target_sources.values())
    print(f"Blocking {len(s1_df):,} Source 1 entities against {total_target_rows:,} target entities ...", file=sys.stderr)

    candidate_map = generate_candidates_multi_source(
        s1_df,
        target_sources,
        max_prefix_block_size=args.max_prefix_block_size,
        max_token_block_size=args.max_token_block_size,
        max_addr_block_size=args.max_addr_block_size,
        max_candidates_per_entity=args.max_candidates_per_entity,
    )

    t_blocking = perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Write output candidate_pairs.tsv
    all_s1_ids = s1_df["entity_id"].astype(str).tolist()
    rows_written = write_candidate_pairs_tsv(candidate_map, all_s1_ids, args.output)

    total_candidates = sum(len(c) for c in candidate_map.values())
    entities_with_cands = sum(1 for c in candidate_map.values() if len(c) > 0)
    avg_cands = total_candidates / len(s1_df) if len(s1_df) > 0 else 0.0

    print(f"\n--- Blocking Summary ---")
    print(f"Output written to: {args.output}")
    print(f"Total Source 1 entities: {rows_written:,}")
    print(f"Total candidate pairs generated: {total_candidates:,}")
    print(f"Entities with >= 1 candidate: {entities_with_cands:,} ({entities_with_cands / max(1, len(s1_df))*100:.2f}%)")
    print(f"Average candidates per Source 1 entity: {avg_cands:.2f}")
    print(f"Blocking runtime: {t_blocking:.2f} seconds")
    print(f"Peak memory: {peak_mem / (1024 * 1024):.2f} MB")

    # Evaluate recall if ground truth is available
    if gt_path and os.path.exists(gt_path):
        print(f"\nEvaluating recall against ground truth: {gt_path} ...", file=sys.stderr)
        gt_df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

        # 1. Full training recall
        full_metrics = evaluate_blocking_metrics(
            candidate_map,
            gt_df,
            total_s1_count=len(s1_df),
            total_target_count=total_target_rows,
        )

        # 2. Held-out validation split
        rng = np.random.default_rng(args.val_seed)
        val_mask = rng.random(len(s1_df)) < args.val_fraction
        val_s1_ids = set(s1_df["entity_id"].iloc[val_mask])
        train_s1_ids = set(s1_df["entity_id"].iloc[~val_mask])

        val_candidate_map = {sid: candidate_map.get(sid, []) for sid in val_s1_ids}
        val_gt = gt_df[gt_df["source1_entity_id"].isin(val_s1_ids)]

        val_metrics = evaluate_blocking_metrics(
            val_candidate_map,
            val_gt,
            total_s1_count=len(val_s1_ids),
            total_target_count=total_target_rows,
        )

        print(f"\n--- Ground Truth Evaluation ---")
        print(f"Full Train Recall Ceiling: {full_metrics['recall_ceiling']*100:.2f}% ({int(full_metrics['recovered_true_pairs']):,} / {int(full_metrics['total_true_pairs']):,})")
        print(f"Full Train Reduction Ratio: {full_metrics['reduction_ratio']*100:.4f}%")
        print(f"Held-out Validation Recall Ceiling: {val_metrics['recall_ceiling']*100:.2f}% ({int(val_metrics['recovered_true_pairs']):,} / {int(val_metrics['total_true_pairs']):,})")
        print(f"Held-out Validation Reduction Ratio: {val_metrics['reduction_ratio']*100:.4f}%")

        # Write markdown report
        if args.report:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
            with open(args.report, "w", encoding="utf-8") as f:
                f.write(f"# Blocking Recall & Candidate Generation Report\n\n")
                f.write(f"Generated by `scripts/run_blocking.py` on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}.\n\n")
                f.write(f"## Evaluation Metrics\n\n")
                f.write(f"| Split | S1 Entities | Total True Pairs | Recovered Pairs | Recall Ceiling | Candidates Generated | Reduction Ratio |\n")
                f.write(f"|---|---|---|---|---|---|---|\n")
                f.write(f"| **Full Set** | {len(s1_df):,} | {int(full_metrics['total_true_pairs']):,} | {int(full_metrics['recovered_true_pairs']):,} | **{full_metrics['recall_ceiling']*100:.2f}%** | {int(full_metrics['total_candidate_pairs']):,} | **{full_metrics['reduction_ratio']*100:.4f}%** |\n")
                f.write(f"| **Held-out Val ({args.val_fraction*100:.0f}%)** | {len(val_s1_ids):,} | {int(val_metrics['total_true_pairs']):,} | {int(val_metrics['recovered_true_pairs']):,} | **{val_metrics['recall_ceiling']*100:.2f}%** | {int(val_metrics['total_candidate_pairs']):,} | **{val_metrics['reduction_ratio']*100:.4f}%** |\n\n")
                f.write(f"## Runtime & Memory\n\n")
                f.write(f"- **Runtime**: {t_blocking:.2f} seconds\n")
                f.write(f"- **Peak Memory (Python-tracked)**: {peak_mem / (1024 * 1024):.2f} MB\n")
                f.write(f"- **Average Candidates / S1 Entity**: {avg_cands:.2f}\n")
                f.write(f"- **Singleton Rate**: {(len(s1_df) - entities_with_cands) / max(1, len(s1_df))*100:.2f}%\n\n")
            print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
