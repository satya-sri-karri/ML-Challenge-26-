import joblib
import pandas as pd

from src.matching.stage_a.inference import (
    predict_stage_a,
)


MODEL_PATH = (
    r"models\stage_a\stage_a_lightgbm.joblib"
)

S1_PATH = (
    r"dataset\sample\train_source1_gt_sample.parquet"
)

S2_PATH = (
    r"dataset\sample\train_source2_gt_sample.parquet"
)

FEATURE_PATH = (
    r"data\stage_a\training_features_sample.parquet"
)


model_package = joblib.load(
    MODEL_PATH
)

source1 = pd.read_parquet(
    S1_PATH
)

source2 = pd.read_parquet(
    S2_PATH
)

features = pd.read_parquet(
    FEATURE_PATH
)


candidate_pairs = features[
    [
        "source1_entity_id",
        "candidate_entity_id",
    ]
].drop_duplicates()


predictions = predict_stage_a(
    source1,
    source2,
    candidate_pairs,
    model_package
)


output_path = (
    r"data\stage_a\stage_a_predictions_sample.tsv"
)


predictions.to_csv(
    output_path,
    sep="\t",
    index=False
)


print("Candidate pairs:", len(candidate_pairs))
print("Predictions:", len(predictions))

print()
print(
    predictions.head(20).to_string(
        index=False
    )
)

print()
print(
    "Score range:",
    round(
        predictions["calibrated_score"].min(),
        4
    ),
    "to",
    round(
        predictions["calibrated_score"].max(),
        4
    )
)

print()
print("Saved:", output_path)
