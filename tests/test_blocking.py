import pandas as pd

from src.blocking.blocker import (
    generate_candidates,
    generate_candidates_multi_source,
    measure_blocking_performance,
    measure_candidate_volume,
    measure_blocking_recall,
    evaluate_blocking_metrics,
    write_candidate_pairs_tsv,
    _extract_name_keys,
    _extract_address_keys,
)
from src.normalization.normalizer import normalize_source


def test_generate_candidates():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1_1", "S1_2"],
            "name_core": ["microsoft", "google"],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": ["S2_1", "S2_2", "S2_3"],
            "name_core": ["microsoft", "apple", "google"],
        }
    )

    result = generate_candidates(source1, source2)

    assert len(result) == 2
    assert set(
        zip(result["source1_entity_id"], result["source2_entity_id"])
    ) == {
        ("S1_1", "S2_1"),
        ("S1_2", "S2_3"),
    }


def test_prefix_blocking():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1_1"],
            "name_core": ["microsoft"],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": ["S2_1", "S2_2"],
            "name_core": ["micromax", "apple"],
        }
    )

    result = generate_candidates(source1, source2)

    assert set(
        zip(result["source1_entity_id"], result["source2_entity_id"])
    ) == {
        ("S1_1", "S2_1"),
    }


def test_measure_candidate_volume():
    candidates = pd.DataFrame(
        {
            "source1_index": [0, 0, 1],
            "source2_index": [0, 1, 2],
        }
    )

    result = measure_candidate_volume(candidates)

    assert result["total_candidate_pairs"] == 3.0
    assert result["unique_source1_entities"] == 2.0
    assert result["unique_source2_entities"] == 3.0
    assert result["avg_candidates_per_source1"] == 1.5


def test_blocking_performance():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1", "S2"],
            "name_core": ["microsoft", "apple"],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": ["T1", "T2", "T3"],
            "name_core": ["microsoft", "micromax", "apple"],
        }
    )

    result = measure_blocking_performance(source1, source2)

    assert "runtime_seconds" in result
    assert "peak_memory_mb" in result
    assert "candidate_pairs" in result

    assert result["runtime_seconds"] >= 0
    assert result["peak_memory_mb"] >= 0
    assert result["candidate_pairs"] >= 0

def test_measure_blocking_recall():
    candidates = pd.DataFrame(
        {
            "source1_entity_id": ["S1_1", "S1_2"],
            "source2_entity_id": ["S2_1", "S2_3"],
        }
    )

    ground_truth = pd.DataFrame(
        {
            "source1_entity_id": ["S1_1", "S1_2"],
            "matched_entity_ids": ["S2_1", "S2_3"],
        }
    )

    result = measure_blocking_recall(
        candidates,
        ground_truth,
    )

    assert result["total_true_pairs"] == 2.0
    assert result["recovered_true_pairs"] == 2.0
    assert result["blocking_recall"] == 1.0  
    
def test_generate_candidates_raises_on_missing_name_column():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1_1"],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": ["S2_1"],
            "name_core": ["microsoft"],
        }
    )

    try:
        generate_candidates(source1, source2)
        assert False
    except ValueError as exc:
        assert "name_core" in str(exc)


def test_generate_candidates_handles_empty_names():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1_1"],
            "name_core": ["microsoft"],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": ["S2_1", "S2_2"],
            "name_core": ["", None],
        }
    )

    result = generate_candidates(source1, source2)

    assert len(result) == 0

def test_large_prefix_block_is_skipped():
    source1 = pd.DataFrame(
        {
            "entity_id": ["S1_1"],
            "name_core": ["company_target"],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": [f"S2_{i}" for i in range(1501)],
            "name_core": [f"company_{i}" for i in range(1501)],
        }
    )

    result = generate_candidates(
        source1,
        source2,
        max_prefix_block_size=1000,
    )

    assert len(result) == 0


def test_name_unrelated_address_matches_blocking():
    """
    Real pattern from reports/eda_summary.md:
    Names differ completely ('Dréxkor' vs true 'Atlantic'), but address
    carries the match (even with typo 'Wanye' vs 'Wayne' and 'Townshiip').
    Blocking MUST recover this via address compound keys.
    """
    s1_raw = pd.DataFrame([
        {
            "entity_id": "S1-101",
            "business_name": "Atlantic",
            "business_address": "85 Wayne Avenue, Ticonderoga, NY",
            "country": "US",
        }
    ])
    s2_raw = pd.DataFrame([
        {
            "entity_id": "S2-201",
            "business_name": "Dréxkor",
            "business_address": "85 Wanye Avenue, Ticonderoga Townshiip, New York",
            "country": "US",
        }
    ])

    s1_norm = normalize_source(s1_raw, "source1")
    s2_norm = normalize_source(s2_raw, "source2")

    result = generate_candidates(s1_norm, s2_norm)
    recovered_pairs = set(zip(result["source1_entity_id"], result["source2_entity_id"]))
    assert ("S1-101", "S2-201") in recovered_pairs


def test_scrambled_address_order_blocking():
    """
    Real pattern from reports/eda_summary.md:
    Address field order is not reliable (state-first, number-last).
    Token-set/bag-of-words blocking keys must match them.
    """
    s1_raw = pd.DataFrame([
        {
            "entity_id": "S1-102",
            "business_name": "Surprise Tools",
            "business_address": "17437 Guthrie Street, 1st Floor, Surprise, AZ",
            "country": "US",
        }
    ])
    s2_raw = pd.DataFrame([
        {
            "entity_id": "S2-202",
            "business_name": "Surprise Tools",
            "business_address": "AZ, Fl 1st Floor, Surprise, 17437 Guthrie Street",
            "country": "US",
        }
    ])

    s1_norm = normalize_source(s1_raw, "source1")
    s2_norm = normalize_source(s2_raw, "source2")

    result = generate_candidates(s1_norm, s2_norm)
    recovered_pairs = set(zip(result["source1_entity_id"], result["source2_entity_id"]))
    assert ("S1-102", "S2-202") in recovered_pairs


def test_empty_business_address_does_not_crash_and_uses_name_path():
    """
    Real pattern from reports/eda_summary.md:
    ~3.3% of Source 2/3 rows have empty business_address.
    Must not crash, and must generate candidates via name tokens.
    """
    s1_raw = pd.DataFrame([
        {
            "entity_id": "S1-103",
            "business_name": "Atlantic",
            "business_address": "85 Wayne Avenue, Ticonderoga, NY",
            "country": "US",
        }
    ])
    s2_raw = pd.DataFrame([
        {
            "entity_id": "S2-203",
            "business_name": "Atlantic Pinnacle Deli Inc.",
            "business_address": "",
            "country": "US",
        }
    ])

    s1_norm = normalize_source(s1_raw, "source1")
    s2_norm = normalize_source(s2_raw, "source2")

    # Address extraction must return empty set without crashing
    addr_keys = _extract_address_keys(s2_norm.iloc[0])
    assert addr_keys == set()

    # Blocking should recover the candidate pair via token 'atlantic'
    result = generate_candidates(s1_norm, s2_norm)
    recovered_pairs = set(zip(result["source1_entity_id"], result["source2_entity_id"]))
    assert ("S1-103", "S2-203") in recovered_pairs


def test_domain_style_name_blocking():
    """
    Real pattern from reports/eda_summary.md:
    ~4% of Source 2/3 rows have domain-style names (e.g. 'aristosteel.com').
    Must match against non-domain reference ('Aristo Steel').
    """
    s1_raw = pd.DataFrame([
        {
            "entity_id": "S1-104",
            "business_name": "Aristo Steel",
            "business_address": "797 Lake Town Block A, Howrah, West Bengal",
            "country": "India",
        }
    ])
    s2_raw = pd.DataFrame([
        {
            "entity_id": "S2-204",
            "business_name": "aristosteel.com",
            "business_address": "Lake Town, Howrah, West Bengal",
            "country": "India",
        }
    ])

    s1_norm = normalize_source(s1_raw, "source1")
    s2_norm = normalize_source(s2_raw, "source2")

    result = generate_candidates(s1_norm, s2_norm)
    recovered_pairs = set(zip(result["source1_entity_id"], result["source2_entity_id"]))
    assert ("S1-104", "S2-204") in recovered_pairs


def test_multi_source_and_candidate_pairs_output(tmp_path):
    """
    Test multi-source candidate generation and write_candidate_pairs_tsv.
    Validates TSV structure: header, comma-separated IDs, singletons with empty string.
    """
    s1_raw = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Alpha Labs", "business_address": "100 Main St, Austin, TX", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Beta Dynamics", "business_address": "200 Oak St, Dallas, TX", "country": "US"},
        {"entity_id": "S1-3", "business_name": "Sole Singleton", "business_address": "300 Pine St, Houston, TX", "country": "US"},
    ])
    s2_raw = pd.DataFrame([
        {"entity_id": "S2-10", "business_name": "Alpha Labs Inc", "business_address": "100 Main St, Austin, TX", "country": "US"},
    ])
    s3_raw = pd.DataFrame([
        {"entity_id": "S3-20", "business_name": "Beta Dynamics LLC", "business_address": "200 Oak St, Dallas, TX", "country": "US"},
    ])

    s1_norm = normalize_source(s1_raw, "source1")
    s2_norm = normalize_source(s2_raw, "source2")
    s3_norm = normalize_source(s3_raw, "source3")

    candidate_map = generate_candidates_multi_source(
        s1_norm,
        {"source2": s2_norm, "source3": s3_norm},
    )

    assert "S1-1" in candidate_map
    assert "S2-10" in candidate_map["S1-1"]
    assert "S1-2" in candidate_map
    assert "S3-20" in candidate_map["S1-2"]
    assert "S1-3" in candidate_map
    assert candidate_map["S1-3"] == []  # True singleton

    out_file = tmp_path / "candidate_pairs.tsv"
    rows_written = write_candidate_pairs_tsv(
        candidate_map,
        s1_norm["entity_id"].tolist(),
        str(out_file),
    )
    assert rows_written == 3

    # Check file contents
    content = [line.rstrip("\r\n") for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert content[0] == "source1_entity_id\tcandidate_entity_ids"
    lines_map = dict(line.split("\t") for line in content[1:])
    assert lines_map["S1-1"] == "S2-10"
    assert lines_map["S1-2"] == "S3-20"
    assert lines_map["S1-3"] == ""


def test_evaluate_blocking_metrics():
    candidate_map = {
        "S1-1": ["S2-10", "S2-11"],
        "S1-2": ["S3-20"],
        "S1-3": [],
    }
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S3-20,S3-21"},
        {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
    ])

    metrics = evaluate_blocking_metrics(
        candidate_map,
        gt,
        total_s1_count=3,
        total_target_count=10,
    )

    # True pairs: (S1-1, S2-10), (S1-2, S3-20), (S1-2, S3-21) -> total 3 true pairs
    # Recovered: (S1-1, S2-10), (S1-2, S3-20) -> 2 recovered
    assert metrics["total_true_pairs"] == 3.0
    assert metrics["recovered_true_pairs"] == 2.0
    assert round(metrics["recall_ceiling"], 4) == round(2.0 / 3.0, 4)
    # Total candidates = 2 + 1 + 0 = 3
    # All possible pairs = 3 * 10 = 30
    assert metrics["total_candidate_pairs"] == 3.0
    assert metrics["all_possible_pairs"] == 30.0
    assert round(metrics["reduction_ratio"], 4) == 0.90


def test_blocking_works_for_unseen_country_france():
    """
    Test that blocking logic does not branch, filter, or fail on France
    (which is added only in the test set).
    """
    s1_raw = pd.DataFrame([
        {
            "entity_id": "S1-FR-1",
            "business_name": "Boulangerie Patisserie Paris",
            "business_address": "15 Rue de Rivoli, Paris",
            "country": "France",
        }
    ])
    s2_raw = pd.DataFrame([
        {
            "entity_id": "S2-FR-1",
            "business_name": "Boulangerie Patisserie Parisienne",
            "business_address": "15 Rue de Rivoli, 75001 Paris",
            "country": "France",
        }
    ])

    s1_norm = normalize_source(s1_raw, "source1")
    s2_norm = normalize_source(s2_raw, "source2")

    result = generate_candidates(s1_norm, s2_norm)
    recovered_pairs = set(zip(result["source1_entity_id"], result["source2_entity_id"]))
    assert ("S1-FR-1", "S2-FR-1") in recovered_pairs


def test_blocking_handles_address_number_drift_and_fractions():
    """
    Real pattern from reports/eda_summary.md:
    Address numbers drift or include fractions (e.g. '3928 1/2 14TH AVE').
    Address keys handle fractions and token matching without errors.
    """
    s1_raw = pd.DataFrame([
        {
            "entity_id": "S1-NUM-1",
            "business_name": "Apex Auto Care",
            "business_address": "3928 1/2 14th Avenue, Brooklyn, NY",
            "country": "US",
        }
    ])
    s2_raw = pd.DataFrame([
        {
            "entity_id": "S2-NUM-1",
            "business_name": "Apex Auto Care Services",
            "business_address": "3928 14th Avenue, Brooklyn, NY",
            "country": "US",
        }
    ])

    s1_norm = normalize_source(s1_raw, "source1")
    s2_norm = normalize_source(s2_raw, "source2")

    result = generate_candidates(s1_norm, s2_norm)
    recovered_pairs = set(zip(result["source1_entity_id"], result["source2_entity_id"]))
    assert ("S1-NUM-1", "S2-NUM-1") in recovered_pairs

      