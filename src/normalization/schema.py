"""
Schema contract for the entity-resolution pipeline.

This module is the single source of truth for column names. It was written
after inspecting the ACTUAL train_source1/2/3.tsv and train_ground_truth.tsv
files (see reports/eda_summary.md for the full profiling output). Do not add
columns here that were not observed in the real data.

Raw schema (identical across source1.tsv, source2.tsv, source3.tsv,
and test_source1/2/3.tsv):

    entity_id        str   e.g. "S1-925783039" / "S2-166376419" / "S3-202863386"
    business_name     str   free text, may be empty, may be non-ASCII / a bare domain
    business_address  str   free text, MAY BE EMPTY (~3.3% empty in source2/3, 0% in source1)
    country           str   open string label — do NOT assume a fixed set.
                            Training only contains {"US", "India"}; the test set
                            adds at least one unseen value ("France"). Never
                            filter, one-hot, or hard-code against a closed set.

train_ground_truth.tsv:

    source1_entity_id   str  an entity_id from source1
    matched_entity_ids  str  comma-separated entity_ids from source2/source3,
                             EMPTY STRING when the source1 entity is a singleton
                             (no true match). ~5.6% of source1 entities are
                             singletons in the training set.

This module does not read files -- see profiling.py / normalizer.py for that.
"""

from __future__ import annotations

# ---- Raw columns, exactly as they appear in the provided TSVs ----
RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

# ---- Normalized columns this module adds. Every one of these is ADDITIVE --
# ---- the raw columns above are always preserved unchanged alongside them. --
#
# Column                  | Type          | Derived from       | Notes
# ------------------------|---------------|--------------------|----------------------------------
# entity_source            | str            | entity_id          | "S1" / "S2" / "S3", parsed from the
#                          |                |                    | entity_id prefix. Purely a convenience
#                          |                |                    | column -- no new information.
# name_clean               | str            | business_name      | Unicode NFKC, whitespace-collapsed,
#                          |                |                    | trimmed. Case and script preserved.
# name_lower               | str            | name_clean         | lower-cased copy of name_clean.
# name_ascii               | str            | name_clean         | Transliterated to ASCII (unidecode).
#                          |                |                    | Fallback for cross-script comparison
#                          |                |                    | (~11-15% of source2/3 names are
#                          |                |                    | non-ASCII, mostly Indian-language
#                          |                |                    | scripts -- see reports/eda_summary.md).
# name_is_domain_style     | bool           | business_name      | True if the name looks like a bare
#                          |                |                    | domain (e.g. "aristosteel.com").
#                          |                |                    | ~4% of source2 names match this.
# name_domain_cleaned      | str or None    | business_name      | TLD stripped, separators normalized,
#                          |                |                    | lower-cased. None when
#                          |                |                    | name_is_domain_style is False.
# name_core                | str            | name_lower          | Legal-suffix removed (see
#                          |                |                    | LEGAL_SUFFIXES below).
# name_suffix_norm         | str or None    | name_lower          | Canonical form of the detected legal
#                          |                |                    | suffix (e.g. "pvt" / "private" ->
#                          |                |                    | "private_limited"), or None.
# name_tokens              | list[str]      | name_core           | Whitespace/punctuation-split tokens,
#                          |                |                    | used for token-set (bag-of-words)
#                          |                |                    | comparison -- NOT positional.
# name_is_empty            | bool           | business_name       | True when business_name == "".
# address_clean            | str            | business_address    | Unicode NFKC, punctuation normalized
#                          |                |                    | to spaces/commas, whitespace collapsed.
# address_ascii            | str            | address_clean       | Transliterated to ASCII.
# address_tokens           | list[str]      | address_clean       | Bag-of-tokens with common abbreviations
#                          |                |                    | expanded (St->street, Rd->road, ...).
#                          |                |                    | Order-INDEPENDENT by design: real
#                          |                |                    | matched addresses in the training data
#                          |                |                    | are NOT reliably in a fixed field
#                          |                |                    | order (see reports/eda_summary.md).
# address_numbers          | list[str]      | business_address    | Numeric substrings extracted separately
#                          |                |                    | (house/street numbers, PIN/ZIP codes),
#                          |                |                    | for small-delta-tolerant comparison
#                          |                |                    | downstream -- exact numeric match is
#                          |                |                    | NOT reliable even for true matches.
# address_is_empty         | bool           | business_address    | True when business_address == ""
#                          |                |                    | (~3.3% of source2/3 rows).
# country_norm             | str            | country             | Trimmed + title-cased. The RAW value
#                          |                |                    | is also kept; country_norm is never
#                          |                |                    | used to filter or gate records.
#
NORMALIZED_COLUMNS = [
    "entity_source",
    "name_clean", "name_lower", "name_ascii",
    "name_is_domain_style", "name_domain_cleaned",
    "name_core", "name_suffix_norm", "name_tokens", "name_is_empty",
    "address_clean", "address_ascii", "address_tokens", "address_numbers",
    "address_is_empty",
    "country_norm",
]

# Canonical legal-suffix normalization table. Keys are lower-cased, punctuation
# stripped tokens observed (or reasonably expected as abbreviation variants) in
# business_name; values are the canonical bucket. This is intentionally small
# and explicit rather than a giant hardcoded gazetteer -- extend only on
# evidence from real data (see reports/eda_summary.md for what was observed).
LEGAL_SUFFIXES = {
    # India-style
    "pvt": "private_limited", "private": "private_limited",
    "ltd": "limited", "limited": "limited",
    "pvt ltd": "private_limited", "pvt. ltd.": "private_limited",
    # US-style
    "inc": "incorporated", "incorporated": "incorporated",
    "corp": "corporation", "corporation": "corporation",
    "llc": "llc", "l.l.c": "llc",
    "co": "company", "company": "company",
}
