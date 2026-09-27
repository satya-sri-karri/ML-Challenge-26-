import random
import pandas as pd


def load_ground_truth(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str
    )


def parse_matched_ids(value):
    if pd.isna(value) or not str(value).strip():
        return set()

    value = str(value).strip()

    for delimiter in ["|", ",", ";"]:
        if delimiter in value:
            return {
                x.strip()
                for x in value.split(delimiter)
                if x.strip()
            }

    return {value}


def get_source_matches(ground_truth, source_prefix):
    """
    Return:
        Source1 ID -> set of matching IDs
    """

    result = {}

    for _, row in ground_truth.iterrows():
        source1_id = str(row["source1_entity_id"])

        matched_ids = parse_matched_ids(
            row["matched_entity_ids"]
        )

        filtered = {
            entity_id
            for entity_id in matched_ids
            if entity_id.startswith(source_prefix + "-")
        }

        result[source1_id] = filtered

    return result


def collect_required_source_ids(ground_truth, source_prefix):
    matches = get_source_matches(
        ground_truth,
        source_prefix
    )

    required_ids = set()

    for ids in matches.values():
        required_ids.update(ids)

    return required_ids


def load_required_source_rows(path, required_ids):
    """
    Scan a TSV in chunks and keep only required entity IDs.
    """

    required_ids = set(required_ids)
    chunks = []
    found_ids = set()

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=100_000
    ):
        selected = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if not selected.empty:
            chunks.append(selected)
            found_ids.update(
                selected["entity_id"].astype(str)
            )

        if len(found_ids) >= len(required_ids):
            break

    if not chunks:
        return pd.DataFrame()

    return pd.concat(
        chunks,
        ignore_index=True
    )


def collect_required_source2_ids(ground_truth):
    return collect_required_source_ids(
        ground_truth,
        "S2"
    )


def load_required_source2_rows(path, required_ids):
    return load_required_source_rows(
        path,
        required_ids
    )


def create_training_pairs(
    source1,
    candidate_source,
    ground_truth,
    source_prefix,
    negatives_per_positive=2,
    random_state=42
):
    """
    Create temporary development pairs for one source.

    Positives come from ground truth.
    Negatives are random non-matching IDs.

    This is for development only. Final training should
    use Person B's blocked candidate pairs.
    """

    rng = random.Random(random_state)

    candidate_ids = (
        candidate_source["entity_id"]
        .astype(str)
        .tolist()
    )

    candidate_id_set = set(candidate_ids)

    gt_matches = get_source_matches(
        ground_truth,
        source_prefix
    )

    rows = []

    for source1_id in source1["entity_id"].astype(str):

        matched_ids = (
            gt_matches.get(source1_id, set())
            & candidate_id_set
        )

        # Positive examples
        for matched_id in matched_ids:
            rows.append({
                "source1_entity_id": source1_id,
                "candidate_entity_id": matched_id,
                "label": 1
            })

        # Negative examples
        negative_pool = list(
            candidate_id_set - matched_ids
        )

        if not negative_pool:
            continue

        number_of_negatives = min(
            len(negative_pool),
            max(
                1,
                len(matched_ids) * negatives_per_positive
            )
        )

        negatives = rng.sample(
            negative_pool,
            number_of_negatives
        )

        for negative_id in negatives:
            rows.append({
                "source1_entity_id": source1_id,
                "candidate_entity_id": negative_id,
                "label": 0
            })

    return pd.DataFrame(rows)


def create_combined_training_pairs(
    source1,
    source2,
    source3,
    ground_truth,
    negatives_per_positive=2,
    random_state=42
):
    """
    Create temporary S1-S2 + S1-S3 development pairs.
    """

    pairs_s2 = create_training_pairs(
        source1=source1,
        candidate_source=source2,
        ground_truth=ground_truth,
        source_prefix="S2",
        negatives_per_positive=negatives_per_positive,
        random_state=random_state
    )

    pairs_s3 = create_training_pairs(
        source1=source1,
        candidate_source=source3,
        ground_truth=ground_truth,
        source_prefix="S3",
        negatives_per_positive=negatives_per_positive,
        random_state=random_state + 1
    )

    return pd.concat(
        [pairs_s2, pairs_s3],
        ignore_index=True
    )
