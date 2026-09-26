#!/usr/bin/env python3
"""
Run chunked normalization over one source file and write normalized parquet.

No hardcoded paths:

    python scripts/run_normalization.py \\
        --input /path/to/dataset/train/train_source1.tsv \\
        --output /path/to/dataset/normalized/train_source1.parquet \\
        --source-name source1

Output path is caller-supplied. Recommended convention: write under
dataset/normalized/ (already covered by the existing "dataset/" gitignore
rule) so normalized output is never accidentally committed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.normalization.normalizer import normalize_file_chunked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--source-name", required=True, help='e.g. "source1", "test_source2"')
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t0 = time.time()
    n = normalize_file_chunked(
        args.input, args.output, source_name=args.source_name, chunksize=args.chunksize
    )
    elapsed = time.time() - t0
    print(f"Normalized {n:,} rows from {args.input} -> {args.output} in {elapsed:.1f}s "
          f"({n/elapsed:,.0f} rows/sec)")


if __name__ == "__main__":
    main()
