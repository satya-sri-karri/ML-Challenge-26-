"""
Shared normalization interface.

This is the ONLY function downstream components (Person B's blocking, Person
C's features) should call to get normalized fields -- so normalization logic
lives in exactly one place and train/test are guaranteed to go through the
identical transformation.

    from src.normalization.normalizer import normalize_source
    normalized_df = normalize_source(df, source_name="source1")

Guarantee: same input schema -> same normalization logic -> same output
schema, regardless of whether df came from train_source*.tsv or
test_source*.tsv. This module does not look at country values or entity_id
ranges to change its behavior, specifically so it generalizes to unseen
countries (e.g. France, present only in the test set).
"""

from __future__ import annotations

import pandas as pd

from . import address as addr
from . import text as txt
from .schema import RAW_COLUMNS


def normalize_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """
    Add normalized columns (see schema.NORMALIZED_COLUMNS) to df IN A COPY.
    Raw columns are preserved unchanged. df must contain at least
    schema.RAW_COLUMNS; extra columns are passed through untouched.

    source_name is a free-form label (e.g. "source1", "test_source2") used
    only to populate the entity_source column as a fallback when an
    entity_id does not carry a recognizable "S1-"/"S2"-/"S3-" prefix -- it
    does not otherwise change any normalization behavior.
    """
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"normalize_source: input is missing required columns {missing}; "
            f"got {list(df.columns)}. This function only handles the schema "
            f"documented in src/normalization/schema.py -- if the real file "
            f"has a different schema, update schema.py first, do not patch "
            f"around it here."
        )

    out = df.copy()

    # -- entity_source: parse "S1"/"S2"/"S3" prefix from entity_id --
    out["entity_source"] = (
        out["entity_id"].astype(str).str.extract(r"^(S\d+)-", expand=False)
    )
    out["entity_source"] = out["entity_source"].fillna(source_name)

    # -- name fields --
    raw_name = out["business_name"].fillna("").astype(str)
    out["name_is_empty"] = raw_name.str.len() == 0
    out["name_clean"] = raw_name.map(txt.clean_name)
    out["name_lower"] = out["name_clean"].str.lower()
    out["name_ascii"] = out["name_clean"].map(txt.to_ascii)
    out["name_is_domain_style"] = raw_name.map(txt.is_domain_style)
    out["name_domain_cleaned"] = [
        txt.clean_domain_name(n) if is_dom else None
        for n, is_dom in zip(raw_name, out["name_is_domain_style"])
    ]
    _core_suffix = out["name_lower"].map(txt.strip_legal_suffix)
    out["name_core"] = _core_suffix.map(lambda t: t[0])
    out["name_suffix_norm"] = _core_suffix.map(lambda t: t[1])
    out["name_tokens"] = out["name_core"].map(txt.tokenize_name)

    # -- address fields --
    raw_address = out["business_address"].fillna("").astype(str)
    out["address_is_empty"] = raw_address.str.len() == 0
    out["address_clean"] = raw_address.map(addr.clean_address)
    out["address_ascii"] = out["address_clean"].map(addr.to_ascii)
    out["address_tokens"] = out["address_clean"].map(addr.tokenize_address)
    out["address_numbers"] = raw_address.map(addr.extract_numbers)

    # -- country: normalized copy only, raw preserved, no filtering --
    out["country_norm"] = (
        out["country"].fillna("").astype(str).str.strip().str.title()
    )

    return out


def normalize_file_chunked(
    input_path: str,
    output_path: str,
    source_name: str,
    chunksize: int = 200_000,
    output_format: str = "parquet",
) -> int:
    """
    Stream-normalize a large TSV without loading it fully into memory.
    Writes normalized output to output_path (parquet by default -- cheaper
    to store list-typed columns like name_tokens than CSV/TSV, and this
    dataset is too large to comfortably re-materialize as TSV repeatedly).

    Returns the total number of rows processed.

    output_path is caller-supplied and NOT hardcoded -- keep generated
    normalized files under a gitignored path (e.g. dataset/normalized/,
    which matches the existing "dataset/" gitignore rule) rather than
    committing them.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    total_rows = 0
    writer = None
    try:
        reader = pd.read_csv(
            input_path, sep="\t", dtype=str, chunksize=chunksize, keep_default_na=False
        )
        for chunk in reader:
            normalized = normalize_source(chunk, source_name=source_name)
            # Parquet can't store Python lists directly via fastparquet in all
            # setups -- pyarrow handles list[str] columns natively, hence the
            # explicit pyarrow table conversion instead of df.to_parquet(...).
            table = pa.Table.from_pandas(normalized, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            total_rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()

    return total_rows
