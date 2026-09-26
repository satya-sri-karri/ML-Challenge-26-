import pandas as pd

from src.blocking.blocker import (
    generate_candidates,
    measure_blocking_performance,
    measure_candidate_volume,
    measure_blocking_recall,
)


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