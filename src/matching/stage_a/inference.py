import os
import joblib
import pandas as pd

from src.features.pair_features import build_pair_features


MODEL_PATH = r"models\stage_a\stage_a_lightgbm.joblib"


def load_stage_a_model(model_path=MODEL_PATH):
    return joblib.load(model_path)


def build_candidate_features(
    source1,
    candidate_source,
    candidate_pairs
):
    source1_records = (
        source1
        .set_index("entity_id")
        .to_dict("index")
    )

    candidate_records = (
        candidate_source
        .set_index("entity_id")
        .to_dict("index")
    )

    rows = []

    for _, pair in candidate_pairs.iterrows():

        source1_id = str(
            pair["source1_entity_id"]
        )

        candidate_id = str(
            pair["candidate_entity_id"]
        )

        if source1_id not in source1_records:
            continue

        if candidate_id not in candidate_records:
            continue

        features = build_pair_features(
            source1_records[source1_id],
            candidate_records[candidate_id]
        )

        features["source1_entity_id"] = source1_id
        features["candidate_entity_id"] = candidate_id

        rows.append(features)

    return pd.DataFrame(rows)


def predict_stage_a(
    source1,
    candidate_source,
    candidate_pairs,
    model_package
):
    feature_df = build_candidate_features(
        source1,
        candidate_source,
        candidate_pairs
    )

    if feature_df.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "calibrated_score",
            ]
        )

    feature_columns = model_package[
        "feature_columns"
    ]

    model = model_package["model"]
    calibrator = model_package["calibrator"]

    probabilities = model.predict_proba(
        feature_df[feature_columns]
    )[:, 1]

    calibrated_scores = calibrator.predict(
        probabilities
    )

    output = feature_df[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].copy()

    output["calibrated_score"] = calibrated_scores

    return output


def save_stage_a_output(
    predictions,
    output_path
):
    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    predictions.to_csv(
        output_path,
        sep="\t",
        index=False
    )
