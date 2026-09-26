"""
Unit tests for src/normalization/.

Test cases are drawn from patterns actually observed in the training data
(see reports/eda_summary.md) -- not invented edge cases with no basis in
the real dataset.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.normalization import text as txt
from src.normalization import address as addr
from src.normalization.normalizer import normalize_source


# ---------------------------------------------------------------- text.py --

def test_clean_name_collapses_whitespace_and_trims():
    assert txt.clean_name("  Atlantic   Pinnacle  Deli   Inc.  ") == "Atlantic Pinnacle Deli Inc."


def test_clean_name_handles_none_and_empty():
    assert txt.clean_name(None) == ""
    assert txt.clean_name("") == ""


def test_clean_name_preserves_casing_and_script():
    # Real observed value -- must not be lower-cased or transliterated by clean_name.
    raw = "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert txt.clean_name(raw) == raw


def test_to_ascii_transliterates_nonascii():
    out = txt.to_ascii("Stepha N. Nygrén, DO")
    assert out.isascii()
    assert "Nygr" in out  # core content preserved even if accents are dropped


def test_is_domain_style_true_cases():
    for name in ["aristosteel.com", "wilfordhancock.com", "www.example.in", "SHOP.NET"]:
        assert txt.is_domain_style(name), name


def test_is_domain_style_false_for_ordinary_names():
    for name in ["Aristo Steel Private Limited", "Maure Williams Colombier Inc", "A & B Traders"]:
        assert not txt.is_domain_style(name), name


def test_clean_domain_name_strips_tld():
    assert txt.clean_domain_name("aristosteel.com") == "aristosteel"
    assert txt.clean_domain_name("www.example.in") == "example"


def test_strip_legal_suffix_various_forms():
    core, suffix = txt.strip_legal_suffix("aristo steel private limited")
    assert core == "aristo steel private"
    assert suffix == "limited"

    core, suffix = txt.strip_legal_suffix("summit inc")
    assert core == "summit"
    assert suffix == "incorporated"

    core, suffix = txt.strip_legal_suffix("chavira platinum chimera llc")
    assert core == "chavira platinum chimera"
    assert suffix == "llc"


def test_strip_legal_suffix_no_suffix_present():
    core, suffix = txt.strip_legal_suffix("orelee's barbershop")
    assert core == "orelee's barbershop"
    assert suffix is None


def test_tokenize_name_splits_on_punctuation():
    assert txt.tokenize_name("atlantic pinnacle deli") == ["atlantic", "pinnacle", "deli"]


# ------------------------------------------------------------- address.py --

def test_clean_address_collapses_whitespace():
    assert addr.clean_address("  105  ELM ST,  MORGANTON,  NC ") == "105 ELM ST, MORGANTON, NC"


def test_clean_address_handles_missing():
    assert addr.clean_address(None) == ""
    assert addr.clean_address("") == ""


def test_extract_numbers_handles_fractions():
    # "14TH" also contributes a numeric token ("14") from the ordinal --
    # that's correct: it's still a number worth comparing downstream.
    assert addr.extract_numbers("3928 1/2 14TH AVE") == ["3928", "1/2", "14"]


def test_extract_numbers_no_numbers():
    assert addr.extract_numbers("Near SBI ATM, Main Road") == []


def test_tokenize_address_expands_abbreviations():
    tokens = addr.tokenize_address("105 Elm St, Morganton, NC")
    assert "street" in tokens
    assert "st" not in tokens


def test_tokenize_address_is_order_independent_by_use():
    # Real observed true-match case: field order scrambled (state before
    # street number). We only assert both produce the SAME token set,
    # not that either is "correctly" ordered -- token-set comparison is
    # the whole point.
    a = addr.tokenize_address("17437 Guthrie Street, Surprise, AZ")
    b = addr.tokenize_address("AZ, Surprise, 17437 Guthrie Street")
    assert set(a) == set(b)


# ---------------------------------------------------------- normalizer.py --

def _sample_df():
    return pd.DataFrame(
        [
            {
                "entity_id": "S1-925783039",
                "business_name": "Orelee's Barbershop",
                "business_address": "1795 Westchester Drive, High Point, NC",
                "country": "US",
            },
            {
                "entity_id": "S2-166376419",
                "business_name": "राम मार्केटिंग प्राइवेट लिमिटेड",
                "business_address": "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
                "country": "India",
            },
            {
                "entity_id": "S3-202863386",
                "business_name": "wilfordhancock.com",
                "business_address": "",
                "country": "US",
            },
            {
                "entity_id": "S1-000000001",
                "business_name": "",
                "business_address": "1 Main St, Nowhere, NY",
                "country": "France",  # unseen-in-training country must not break anything
            },
        ]
    )


def test_normalize_source_adds_expected_columns():
    out = normalize_source(_sample_df(), source_name="mixed")
    from src.normalization.schema import NORMALIZED_COLUMNS

    for col in NORMALIZED_COLUMNS:
        assert col in out.columns, col


def test_normalize_source_preserves_raw_columns_unchanged():
    df = _sample_df()
    out = normalize_source(df, source_name="mixed")
    for col in ["entity_id", "business_name", "business_address", "country"]:
        assert (out[col] == df[col]).all()


def test_normalize_source_handles_empty_address_and_name():
    out = normalize_source(_sample_df(), source_name="mixed")
    row = out[out["entity_id"] == "S3-202863386"].iloc[0]
    assert row["address_is_empty"]
    assert row["name_is_domain_style"]
    assert row["name_domain_cleaned"] == "wilfordhancock"

    row2 = out[out["entity_id"] == "S1-000000001"].iloc[0]
    assert row2["name_is_empty"]


def test_normalize_source_does_not_gate_on_country():
    # France is unseen in training -- must be processed identically to any
    # other country string, not filtered/dropped/special-cased.
    out = normalize_source(_sample_df(), source_name="mixed")
    france_row = out[out["country"] == "France"]
    assert len(france_row) == 1
    assert france_row.iloc[0]["country_norm"] == "France"


def test_normalize_source_entity_source_prefix():
    out = normalize_source(_sample_df(), source_name="mixed")
    assert list(out["entity_source"]) == ["S1", "S2", "S3", "S1"]


def test_normalize_source_raises_on_missing_columns():
    import pytest

    bad_df = pd.DataFrame([{"entity_id": "S1-1", "business_name": "x"}])
    try:
        normalize_source(bad_df, source_name="mixed")
        assert False, "expected ValueError"
    except ValueError:
        pass
