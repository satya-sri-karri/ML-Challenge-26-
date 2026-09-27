import pandas as pd


REQUIRED_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_ids",
]


def load_candidate_pairs(path):
    """
    Load Person-B's blocking output.

    Input format:
        source1_entity_id    candidate_entity_ids

    Example:
        S1-001    S2-001,S2-002,S3-001

    Returns pair-level rows:
        source1_entity_id    candidate_entity_id
    """

    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    pairs = []

    for _, row in df.iterrows():
        source1_id = str(
            row["source1_entity_id"]
        ).strip()

        candidate_ids = str(
            row["candidate_entity_ids"]
        ).strip()

        if not source1_id or not candidate_ids:
            continue

        for candidate_id in candidate_ids.split(","):
            candidate_id = candidate_id.strip()

            if not candidate_id:
                continue

            pairs.append({
                "source1_entity_id": source1_id,
                "candidate_entity_id": candidate_id,
            })

    result = pd.DataFrame(
        pairs,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    )

    if result.empty:
        return result

    return (
        result
        .drop_duplicates()
        .reset_index(drop=True)
    )