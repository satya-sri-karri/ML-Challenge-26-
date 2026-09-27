import pandas as pd

from src.blocking.blocker import (
    generate_candidates,
    generate_candidates_multi_source,
    measure_blocking_performance,
    measure_candidate_volume,
    measure_blocking_recall,
    evaluate_blocking_metrics,
    write_candidate_pairs_tsv,
    write_candidate_pairs_streaming,
    build_target_index,
    query_source1_chunk,
    stream_candidate_pairs,
    stream_candidates_s1_chunked,
    find_target_pruned_keys,
    evaluate_candidate_file_recall,
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


# ---------------------------------------------------------------------------
# Streaming Blocking Architecture Tests
# ---------------------------------------------------------------------------

def test_streaming_chunked_source1_processing(tmp_path):
    """
    Verify Source 1 chunked streaming works correctly when chunk_size is smaller
    than the total number of Source 1 rows (e.g. 5 rows with chunk_size=2).
    """
    s1_df = pd.DataFrame([
        {"entity_id": f"S1-{i}", "business_name": f"Company {i}", "business_address": f"{100+i} Main St, Austin, TX", "country": "US"}
        for i in range(5)
    ])
    s2_df = pd.DataFrame([
        {"entity_id": f"S2-{i}", "business_name": f"Company {i} Inc", "business_address": f"{100+i} Main St, Austin, TX", "country": "US"}
        for i in range(5)
    ])

    out_file = tmp_path / "streaming_chunks.tsv"
    stats = stream_candidate_pairs(
        s1_df,
        [("source2", s2_df)],
        str(out_file),
        chunk_size=2,
    )

    assert stats["total_source1_entities"] == 5
    assert stats["entities_with_candidates"] == 5
    assert stats["singleton_count"] == 0

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 6  # 1 header + 5 rows
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    for i in range(5):
        s1_id, cands = lines[i + 1].split("\t")
        assert s1_id == f"S1-{i}"
        assert f"S2-{i}" in cands


def test_streaming_incremental_writing(tmp_path):
    """
    Verify write_candidate_pairs_streaming writes incrementally without holding
    the entire dataset in memory.
    """
    out_file = tmp_path / "incremental_writer.tsv"
    with open(out_file, "w", encoding="utf-8") as f:
        # Chunk 1 with header
        w1 = write_candidate_pairs_streaming(
            {"S1-1": ["S2-10", "S2-11"], "S1-2": ["S2-20"]},
            f,
            write_header=True,
        )
        assert w1 == 2

        # Chunk 2 without header
        w2 = write_candidate_pairs_streaming(
            {"S1-3": []},
            f,
            write_header=False,
        )
        assert w2 == 1

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 4
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    assert lines[1] == "S1-1\tS2-10,S2-11"
    assert lines[2] == "S1-2\tS2-20"
    assert lines[3] == "S1-3\t"


def test_streaming_one_output_row_per_s1_entity(tmp_path):
    """
    Verify the final output contains EXACTLY one row per Source 1 entity,
    preserving exact input sequence.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-A", "business_name": "Alpha Corp", "business_address": "100 Pine St, Boston, MA", "country": "US"},
        {"entity_id": "S1-B", "business_name": "Beta Tech", "business_address": "200 Elm St, Boston, MA", "country": "US"},
        {"entity_id": "S1-C", "business_name": "Gamma Logic", "business_address": "300 Cedar St, Boston, MA", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-A1", "business_name": "Alpha Corp", "business_address": "100 Pine St, Boston, MA", "country": "US"},
    ])

    out_file = tmp_path / "one_row_per_s1.tsv"
    stream_candidate_pairs(s1_df, [("source2", s2_df)], str(out_file), chunk_size=2)

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 4  # header + 3 rows
    row_ids = [l.split("\t")[0] for l in lines[1:]]
    assert row_ids == ["S1-A", "S1-B", "S1-C"]


def test_streaming_union_of_s2_and_s3_candidates(tmp_path):
    """
    Verify that candidates from sequentially processed Source-2 and Source-3
    are unioned together for every Source 1 entity.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Nexus Robotics", "business_address": "500 High St, Seattle, WA", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-101", "business_name": "Nexus Robotics Inc", "business_address": "500 High St, Seattle, WA", "country": "US"},
    ])
    s3_df = pd.DataFrame([
        {"entity_id": "S3-201", "business_name": "Nexus Robotics LLC", "business_address": "500 High St, Seattle, WA", "country": "US"},
    ])

    out_file = tmp_path / "union_s2_s3.tsv"
    stream_candidate_pairs(
        s1_df,
        [("source2", s2_df), ("source3", s3_df)],
        str(out_file),
    )

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    s1_id, cand_str = lines[1].split("\t")
    cands = cand_str.split(",")
    assert s1_id == "S1-1"
    assert "S2-101" in cands
    assert "S3-201" in cands


def test_streaming_no_duplicate_candidate_ids(tmp_path):
    """
    Verify candidate IDs per entity are strictly deduplicated and sorted.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Apex Group", "business_address": "10 Center Way, Denver, CO", "country": "US"},
    ])
    # Target with multiple rows that match on different signals (exact + prefix + token)
    s2_df = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Apex Group", "business_address": "10 Center Way, Denver, CO", "country": "US"},
        {"entity_id": "S2-2", "business_name": "Apex Group International", "business_address": "10 Center Way, Denver, CO", "country": "US"},
    ])

    out_file = tmp_path / "no_duplicates.tsv"
    stream_candidate_pairs(s1_df, [("source2", s2_df)], str(out_file))

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    cand_str = lines[1].split("\t")[1]
    cand_list = cand_str.split(",")
    assert len(cand_list) == len(set(cand_list))
    assert cand_list == sorted(cand_list)


def test_streaming_empty_candidate_rows(tmp_path):
    """
    Verify that entities with zero candidates produce an empty candidate string
    with the exact format 'source1_entity_id<TAB>'.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-EMPTY", "business_name": "UniqueXyz12345", "business_address": "999 Nowhere Rd, Lost, AK", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-OTHER", "business_name": "Standard Company", "business_address": "100 Main St, Boston, MA", "country": "US"},
    ])

    out_file = tmp_path / "empty_candidate_row.tsv"
    stats = stream_candidate_pairs(s1_df, [("source2", s2_df)], str(out_file))

    assert stats["total_candidate_pairs"] == 0
    assert stats["singleton_count"] == 1

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    parts = lines[1].split("\t")
    assert parts[0] == "S1-EMPTY"
    assert parts[1] == ""


def test_streaming_empty_business_address_safety(tmp_path):
    """
    Verify streaming pipeline safely handles rows where business_address is empty
    without crashing, generating candidates via name signals.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-NOADDR", "business_name": "Pacific BioSciences", "business_address": "", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-NOADDR", "business_name": "Pacific BioSciences Inc", "business_address": "", "country": "US"},
    ])

    out_file = tmp_path / "empty_address.tsv"
    stream_candidate_pairs(s1_df, [("source2", s2_df)], str(out_file))

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert "S2-NOADDR" in lines[1].split("\t")[1]


def test_streaming_name_only_and_address_only_candidate_paths(tmp_path):
    """
    Verify that both name-only candidate matching (different addresses) and
    address-only candidate matching (different names) succeed in streaming mode.
    """
    s1_df = pd.DataFrame([
        # Name match, address completely different
        {"entity_id": "S1-NAME", "business_name": "Starlight Ventures", "business_address": "100 Broadway, NY", "country": "US"},
        # Address match, name completely different
        {"entity_id": "S1-ADDR", "business_name": "Green Valley", "business_address": "85 Wayne Ave, Ticonderoga, NY", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-NAME", "business_name": "Starlight Ventures", "business_address": "999 Pacific Hwy, San Diego, CA", "country": "US"},
        {"entity_id": "S2-ADDR", "business_name": "Unrelated Firm Name", "business_address": "85 Wayne Ave, Ticonderoga, NY", "country": "US"},
    ])

    out_file = tmp_path / "name_addr_paths.tsv"
    stream_candidate_pairs(s1_df, [("source2", s2_df)], str(out_file))

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    mapping = dict(line.split("\t") for line in lines[1:])
    assert "S2-NAME" in mapping["S1-NAME"]
    assert "S2-ADDR" in mapping["S1-ADDR"]


def test_streaming_unrelated_name_address_match(tmp_path):
    """
    Verify real noise case from reports/eda_summary.md:
    Dréxkor vs Atlantic (unrelated names, typo in address: Wanye vs Wayne)
    recovered through the streaming architecture.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-DREX", "business_name": "Atlantic", "business_address": "85 Wayne Avenue, Ticonderoga, NY", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-DREX", "business_name": "Dréxkor", "business_address": "85 Wanye Avenue, Ticonderoga Townshiip, New York", "country": "US"},
    ])

    out_file = tmp_path / "drexkor_streaming.tsv"
    stream_candidate_pairs(s1_df, [("source2", s2_df)], str(out_file))

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert "S2-DREX" in lines[1].split("\t")[1]


def test_streaming_scrambled_address_order(tmp_path):
    """
    Verify scrambled address order works properly in streaming mode.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-SCRAMBLE", "business_name": "Surprise Tools", "business_address": "17437 Guthrie Street, 1st Floor, Surprise, AZ", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-SCRAMBLE", "business_name": "Surprise Tools", "business_address": "AZ, Fl 1st Floor, Surprise, 17437 Guthrie Street", "country": "US"},
    ])

    out_file = tmp_path / "scramble_streaming.tsv"
    stream_candidate_pairs(s1_df, [("source2", s2_df)], str(out_file))

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert "S2-SCRAMBLE" in lines[1].split("\t")[1]


def test_streaming_no_country_filtering(tmp_path):
    """
    Verify that unseen countries like France or Japan pass through without any
    country gating or filtering in streaming mode.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-FR", "business_name": "Boulangerie Paris", "business_address": "15 Rue de Rivoli, Paris", "country": "France"},
        {"entity_id": "S1-JP", "business_name": "Tokyo Electronics", "business_address": "1-1 Chiyoda, Tokyo", "country": "Japan"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-FR", "business_name": "Boulangerie Parisienne", "business_address": "15 Rue de Rivoli, 75001 Paris", "country": "France"},
        {"entity_id": "S2-JP", "business_name": "Tokyo Electronics Corp", "business_address": "Chiyoda, Tokyo", "country": "Japan"},
    ])

    out_file = tmp_path / "countries_streaming.tsv"
    stream_candidate_pairs(s1_df, [("source2", s2_df)], str(out_file))

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    mapping = dict(line.split("\t") for line in lines[1:])
    assert "S2-FR" in mapping["S1-FR"]
    assert "S2-JP" in mapping["S1-JP"]


def test_streaming_evaluate_candidate_file_recall(tmp_path):
    """
    Verify evaluate_candidate_file_recall computes exact metrics directly from
    the candidate TSV file against ground truth.
    """
    cand_file = tmp_path / "test_eval_cands.tsv"
    cand_file.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-1\tS2-10,S2-11\n"
        "S1-2\tS3-20\n"
        "S1-3\t\n",
        encoding="utf-8",
    )
    gt = pd.DataFrame([
        {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10"},
        {"source1_entity_id": "S1-2", "matched_entity_ids": "S3-20,S3-21"},
        {"source1_entity_id": "S1-3", "matched_entity_ids": ""},
    ])

    metrics = evaluate_candidate_file_recall(str(cand_file), gt, total_target_count=10)
    assert metrics["total_true_pairs"] == 3.0
    assert metrics["recovered_true_pairs"] == 2.0
    assert round(metrics["recall_ceiling"], 4) == round(2.0 / 3.0, 4)
    assert metrics["total_candidate_pairs"] == 3.0
    assert metrics["all_possible_pairs"] == 30.0
    assert round(metrics["reduction_ratio"], 4) == 0.90


def test_streaming_target_partitioning_equivalence(tmp_path):
    """
    Verify that target partitioning produces candidate sets mathematically
    equivalent to unpartitioned target execution.
    """
    s1_df = pd.DataFrame([
        {"entity_id": f"S1-{i}", "business_name": f"TargetCorp {i}", "business_address": f"{100+i} Main St, Austin, TX", "country": "US"}
        for i in range(6)
    ])
    s2_df = pd.DataFrame([
        {"entity_id": f"S2-{i}", "business_name": f"TargetCorp {i} Inc", "business_address": f"{100+i} Main St, Austin, TX", "country": "US"}
        for i in range(10)
    ])
    s3_df = pd.DataFrame([
        {"entity_id": f"S3-{i}", "business_name": f"TargetCorp {i} LLC", "business_address": f"{100+i} Main St, Austin, TX", "country": "US"}
        for i in range(10)
    ])

    out_unpart = tmp_path / "unpartitioned.tsv"
    out_part = tmp_path / "partitioned.tsv"

    # Unpartitioned run
    stream_candidate_pairs(
        s1_df,
        [("source2", s2_df), ("source3", s3_df)],
        str(out_unpart),
        target_partition_size=None,
    )

    # Partitioned run with small partitions (3 rows per partition -> 4 partitions per source)
    stream_candidate_pairs(
        s1_df,
        [("source2", s2_df), ("source3", s3_df)],
        str(out_part),
        target_partition_size=3,
        chunk_size=2,
    )

    lines_unpart = [l.rstrip("\r\n") for l in out_unpart.read_text(encoding="utf-8").splitlines() if l.strip()]
    lines_part = [l.rstrip("\r\n") for l in out_part.read_text(encoding="utf-8").splitlines() if l.strip()]

    assert lines_unpart[0] == "source1_entity_id\tcandidate_entity_ids"
    assert lines_part[0] == "source1_entity_id\tcandidate_entity_ids"
    assert len(lines_unpart) == len(lines_part) == 7  # header + 6 S1 rows

    map_unpart = dict(l.split("\t") for l in lines_unpart[1:])
    map_part = dict(l.split("\t") for l in lines_part[1:])

    for sid in map_unpart:
        c_unpart = set(map_unpart[sid].split(",")) if map_unpart[sid] else set()
        c_part = set(map_part[sid].split(",")) if map_part[sid] else set()
        assert c_unpart == c_part, f"Partition mismatch for {sid}: {c_unpart} vs {c_part}"


def test_streaming_target_partitioning_cross_partition_matches(tmp_path):
    """
    Verify that an S1 entity matching targets in MULTIPLE different partitions
    correctly unions all of them together, proving that partitioning does not
    prevent an entity from seeing matching targets in other partitions.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-MULTI", "business_name": "Acme Global", "business_address": "100 Broadway, NY", "country": "US"},
    ])
    # Target S2 where Acme Global targets are scattered in rows 0, 3, and 6
    s2_df = pd.DataFrame([
        {"entity_id": "S2-P0", "business_name": "Acme Global Inc", "business_address": "100 Broadway, NY", "country": "US"},   # Part 0
        {"entity_id": "S2-OTHER1", "business_name": "Unrelated 1", "business_address": "999 Nowhere Rd", "country": "US"},     # Part 0
        {"entity_id": "S2-OTHER2", "business_name": "Unrelated 2", "business_address": "998 Nowhere Rd", "country": "US"},     # Part 0
        {"entity_id": "S2-P1", "business_name": "Acme Global LLC", "business_address": "100 Broadway, NY", "country": "US"},   # Part 1
        {"entity_id": "S2-OTHER3", "business_name": "Unrelated 3", "business_address": "997 Nowhere Rd", "country": "US"},     # Part 1
        {"entity_id": "S2-OTHER4", "business_name": "Unrelated 4", "business_address": "996 Nowhere Rd", "country": "US"},     # Part 1
        {"entity_id": "S2-P2", "business_name": "Acme Global Corp", "business_address": "100 Broadway, NY", "country": "US"},  # Part 2
    ])

    out_file = tmp_path / "cross_partition.tsv"
    stream_candidate_pairs(
        s1_df,
        [("source2", s2_df)],
        str(out_file),
        target_partition_size=3,  # Partitions: rows 0-2 (Part 0), rows 3-5 (Part 1), rows 6 (Part 2)
    )

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    s1_id, cand_str = lines[1].split("\t")
    cands = cand_str.split(",")
    assert s1_id == "S1-MULTI"
    assert "S2-P0" in cands
    assert "S2-P1" in cands
    assert "S2-P2" in cands
    assert len(cands) == len(set(cands))  # no duplicates


def test_streaming_target_partitioning_from_tsv_files(tmp_path):
    """
    Verify that target partitioning works accurately when inputs are real TSV files on disk.
    """
    s1_path = tmp_path / "s1.tsv"
    s2_path = tmp_path / "s2.tsv"
    s3_path = tmp_path / "s3.tsv"
    out_path = tmp_path / "from_tsv_out.tsv"

    pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Vertex Dynamics", "business_address": "100 Pine St, Dallas, TX", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Lone Star Solitary", "business_address": "999 Ghost Rd, Nowhere, TX", "country": "US"},
    ]).to_csv(s1_path, sep="\t", index=False)

    pd.DataFrame([
        {"entity_id": "S2-A", "business_name": "Vertex Dynamics Inc", "business_address": "100 Pine St, Dallas, TX", "country": "US"},
        {"entity_id": "S2-B", "business_name": "Random Corp", "business_address": "500 Other St, Dallas, TX", "country": "US"},
    ]).to_csv(s2_path, sep="\t", index=False)

    pd.DataFrame([
        {"entity_id": "S3-A", "business_name": "Vertex Dynamics LLC", "business_address": "100 Pine St, Dallas, TX", "country": "US"},
        {"entity_id": "S3-B", "business_name": "Another Unrelated", "business_address": "800 Elm St, Dallas, TX", "country": "US"},
    ]).to_csv(s3_path, sep="\t", index=False)

    stream_candidate_pairs(
        str(s1_path),
        [("source2", str(s2_path)), ("source3", str(s3_path))],
        str(out_path),
        chunk_size=1,
        target_partition_size=1,  # Force multiple partitions per source
    )

    lines = [l.rstrip("\r\n") for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    mapping = dict(l.split("\t") for l in lines[1:])
    cands_s1_1 = set(mapping["S1-1"].split(","))
    assert "S2-A" in cands_s1_1
    assert "S3-A" in cands_s1_1
    assert mapping["S1-2"] == ""


def test_streaming_empty_target_source(tmp_path):
    """
    Verify streaming candidate pairs handles empty target datasets gracefully.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Lone Entity", "business_address": "123 Main St", "country": "US"},
    ])
    empty_s2 = pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    out_file = tmp_path / "empty_target.tsv"

    stats = stream_candidate_pairs(s1_df, [("source2", empty_s2)], str(out_file))
    assert stats["total_candidate_pairs"] == 0
    assert stats["singleton_count"] == 1

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[1] == "S1-1\t"


def test_s1_chunked_streaming_exact_equivalence_to_batch(tmp_path):
    """
    Verify stream_candidates_s1_chunked produces candidate sets 100% equivalent
    to generate_candidates_multi_source.
    """
    s1_df = pd.DataFrame([
        {"entity_id": f"S1-{i}", "business_name": f"Company {i}", "business_address": f"{100+i} Main St, Austin, TX", "country": "US"}
        for i in range(10)
    ])
    s2_df = pd.DataFrame([
        {"entity_id": f"S2-{i}", "business_name": f"Company {i} Inc", "business_address": f"{100+i} Main St, Austin, TX", "country": "US"}
        for i in range(15)
    ])
    s3_df = pd.DataFrame([
        {"entity_id": f"S3-{i}", "business_name": f"Company {i} LLC", "business_address": f"{100+i} Main St, Austin, TX", "country": "US"}
        for i in range(15)
    ])

    s1_norm = normalize_source(s1_df, "source1")
    s2_norm = normalize_source(s2_df, "source2")
    s3_norm = normalize_source(s3_df, "source3")

    batch_map = generate_candidates_multi_source(
        s1_norm,
        {"source2": s2_norm, "source3": s3_norm},
    )

    out_file = tmp_path / "s1_chunked_test.tsv"
    stream_candidates_s1_chunked(
        s1_norm,
        [("source2", s2_norm), ("source3", s3_norm)],
        str(out_file),
        s1_chunk_size=3,  # force multiple chunks
    )

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines[0] == "source1_entity_id\tcandidate_entity_ids"
    assert len(lines) == 11  # header + 10 S1 rows

    stream_map = dict(l.split("\t") for l in lines[1:])
    for sid in batch_map:
        b_cands = set(batch_map[sid])
        s_cands = set(stream_map[sid].split(",")) if stream_map[sid] else set()
        assert b_cands == s_cands, f"Mismatch for {sid}: {b_cands} vs {s_cands}"


def test_s1_chunked_streaming_chunking_and_singletons(tmp_path):
    """
    Verify chunked S1 processing correctly outputs singletons (empty candidate rows)
    in the exact order of S1 without duplicate candidate IDs.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-A", "business_name": "Unique Alpha Corp", "business_address": "100 Pine St, Boston, MA", "country": "US"},
        {"entity_id": "S1-B", "business_name": "Solo Entity 999", "business_address": "999 Nowhere Rd, Lost, AK", "country": "US"},
        {"entity_id": "S1-C", "business_name": "Unique Gamma LLC", "business_address": "300 Cedar St, Boston, MA", "country": "US"},
    ])
    s2_df = pd.DataFrame([
        {"entity_id": "S2-A1", "business_name": "Unique Alpha Corp Inc", "business_address": "100 Pine St, Boston, MA", "country": "US"},
        {"entity_id": "S2-C1", "business_name": "Unique Gamma LLC Group", "business_address": "300 Cedar St, Boston, MA", "country": "US"},
    ])

    out_file = tmp_path / "chunked_singletons.tsv"
    stats = stream_candidates_s1_chunked(
        s1_df,
        [("source2", s2_df)],
        str(out_file),
        s1_chunk_size=1,  # one entity per chunk
    )

    assert stats["total_source1_entities"] == 3
    assert stats["singleton_count"] == 1
    assert stats["entities_with_candidates"] == 2

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 4
    rows = [l.split("\t") for l in lines[1:]]
    assert rows[0][0] == "S1-A" and "S2-A1" in rows[0][1]
    assert rows[1][0] == "S1-B" and rows[1][1] == ""
    assert rows[2][0] == "S1-C" and "S2-C1" in rows[2][1]


def test_s1_chunked_streaming_from_tsv_files(tmp_path):
    """
    Verify that stream_candidates_s1_chunked works accurately with disk-backed TSV files.
    """
    s1_path = tmp_path / "s1.tsv"
    s2_path = tmp_path / "s2.tsv"
    s3_path = tmp_path / "s3.tsv"
    out_path = tmp_path / "from_tsv_s1chunked.tsv"

    pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Apex Dynamics", "business_address": "100 Pine St, Dallas, TX", "country": "US"},
        {"entity_id": "S1-2", "business_name": "Lone Star Solitary", "business_address": "999 Ghost Rd, Nowhere, TX", "country": "US"},
    ]).to_csv(s1_path, sep="\t", index=False)

    pd.DataFrame([
        {"entity_id": "S2-A", "business_name": "Apex Dynamics Inc", "business_address": "100 Pine St, Dallas, TX", "country": "US"},
        {"entity_id": "S2-B", "business_name": "Random Corp", "business_address": "500 Other St, Dallas, TX", "country": "US"},
    ]).to_csv(s2_path, sep="\t", index=False)

    pd.DataFrame([
        {"entity_id": "S3-A", "business_name": "Apex Dynamics LLC", "business_address": "100 Pine St, Dallas, TX", "country": "US"},
        {"entity_id": "S3-B", "business_name": "Another Unrelated", "business_address": "800 Elm St, Dallas, TX", "country": "US"},
    ]).to_csv(s3_path, sep="\t", index=False)

    stream_candidates_s1_chunked(
        str(s1_path),
        [("source2", str(s2_path)), ("source3", str(s3_path))],
        str(out_path),
        s1_chunk_size=1,
    )

    lines = [l.rstrip("\r\n") for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    mapping = dict(l.split("\t") for l in lines[1:])
    cands_s1_1 = set(mapping["S1-1"].split(","))
    assert "S2-A" in cands_s1_1
    assert "S3-A" in cands_s1_1
    assert mapping["S1-2"] == ""


def test_s1_chunked_target_pruning_preservation(tmp_path):
    """
    Verify that frequent keys in target sources exceeding threshold are pruned
    and do not generate false match candidates.
    """
    s1_df = pd.DataFrame([
        {"entity_id": "S1-1", "business_name": "Business Frequent", "business_address": "100 Elm St", "country": "US"},
    ])
    # Target S2 where 5 rows share token 'frequent' (different prefix 'fre' vs 'bus')
    s2_df = pd.DataFrame([
        {"entity_id": f"S2-{i}", "business_name": f"Frequent Company {i}", "business_address": f"{200+i} Oak Ave", "country": "US"}
        for i in range(5)
    ])

    out_file = tmp_path / "pruned_test.tsv"
    # Setting max_token_block_size=3 means 'name_token:frequent' (count=5) should be pruned
    stream_candidates_s1_chunked(
        s1_df,
        [("source2", s2_df)],
        str(out_file),
        max_token_block_size=3,
    )

    lines = [l.rstrip("\r\n") for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    # Because token 'frequent' is pruned and addresses and prefixes differ, S1-1 should have 0 candidates
    assert lines[1] == "S1-1\t"