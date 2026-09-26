#!/usr/bin/env python3
"""
Run dataset profiling and write a compact summary report.

No hardcoded paths -- pass your local dataset location explicitly:

    python scripts/run_profiling.py --dataset-dir /path/to/dataset --split train

Writes reports/eda_summary.json (machine-readable) and prints a short
human-readable summary to stdout. Does NOT write any row-level output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.normalization.profiling import profile_file, profile_ground_truth, save_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Path to the local dataset/ root")
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "eda_summary.json"))
    args = ap.parse_args()

    split_dir = os.path.join(args.dataset_dir, args.split)
    summary = {"split": args.split, "sources": {}}

    for source_name in ["source1", "source2", "source3"]:
        path = os.path.join(split_dir, f"{args.split}_{source_name}.tsv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found", file=sys.stderr)
            continue
        print(f"Profiling {path} ...", file=sys.stderr)
        summary["sources"][source_name] = profile_file(path, chunksize=args.chunksize)

    if args.split == "train":
        gt_path = os.path.join(split_dir, "train_ground_truth.tsv")
        if os.path.exists(gt_path):
            print(f"Profiling {gt_path} ...", file=sys.stderr)
            summary["ground_truth"] = profile_ground_truth(gt_path, chunksize=args.chunksize)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_summary(summary, args.out)
    print(f"Wrote {args.out}")

    for name, s in summary["sources"].items():
        print(f"\n{name}: {s['row_count']:,} rows")
        if "domain_style_name_rate" in s:
            print(f"  domain-style names: {s['domain_style_name_rate']*100:.2f}%")
    if "ground_truth" in summary:
        gt = summary["ground_truth"]
        print(f"\nground truth: singleton_rate={gt['singleton_rate']*100:.2f}%, "
              f"avg_matches_per_nonsingleton={gt['avg_matches_per_nonsingleton']:.2f}")


if __name__ == "__main__":
    main()
