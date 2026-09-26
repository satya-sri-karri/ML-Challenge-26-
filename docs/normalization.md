# Normalization module — how to use it

Owned by Person A. Branch: `feature/normalization`.

## What it does

`src/normalization/` turns raw `entity_id, business_name, business_address,
country` rows into the same rows plus a fixed set of additional normalized
columns — see `src/normalization/schema.py` for the full column-by-column
contract (types, derivation, rationale). Raw columns are never modified or
removed.

## Quick start (in-memory, one DataFrame)

```python
import pandas as pd
from src.normalization.normalizer import normalize_source

df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False)
normalized = normalize_source(df, source_name="source1")
```

`source_name` is just a label (used only as a fallback if `entity_id`
doesn't carry a recognizable `S1-`/`S2-`/`S3-` prefix) — it does not change
normalization behavior. **Call this identically for train and test data.**

## Large files (chunked, writes parquet)

```bash
python scripts/run_normalization.py \
    --input  /path/to/dataset/train/train_source2.tsv \
    --output /path/to/dataset/normalized/train_source2.parquet \
    --source-name source2
```

Throughput observed on this container (1 CPU): **~25,000 rows/sec** —
roughly 3.5 minutes for the full ~5M-row source2/source3 files. Output is
parquet (not TSV) because several columns (`name_tokens`, `address_tokens`,
`address_numbers`) are list-typed, which parquet handles natively.

Recommended convention: write normalized output under `dataset/normalized/`
— this is already covered by the repo's `dataset/` gitignore rule, so it
won't be accidentally committed.

## Columns downstream code should actually use

For **blocking** (Person B), the most useful columns are almost certainly:
- `name_core`, `name_ascii`, `name_tokens` — for token/n-gram/embedding blocking keys
- `address_tokens`, `address_numbers` — for token-set and numeric blocking keys
- `name_is_empty` / `address_is_empty` — to route empty-field records differently
  rather than treating a missing field as "no similarity"

For **features** (Person C), all normalized columns are fair game as
similarity-feature inputs; `name_suffix_norm` and `name_is_domain_style`
are also useful as categorical features in their own right.

**Do not re-derive these in blocking/features code** — import from
`src.normalization`, so a future change to normalization logic (e.g. an
improved domain-name cleaner) automatically propagates everywhere instead
of silently diverging between components.

## Full column reference, EDA findings, and known limitations

See `src/normalization/schema.py` (docstring + `NORMALIZED_COLUMNS` table)
and `reports/eda_summary.md`.
