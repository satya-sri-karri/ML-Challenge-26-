import pandas as pd

from src.matching.stage_a.training_pairs import (
    load_ground_truth,
    create_combined_training_pairs,
)

from src.features.pair_features import build_pair_features


GT_PATH = r"dataset\sample\train_ground_truth_sample.tsv"

S1_PATH = r"dataset\sample\train_source1_gt_sample.parquet"
S2_PATH = r"dataset\sample\train_source2_gt_sample.parquet"
S3_PATH = r"dataset\sample\train_source3_gt_sample.parquet"


gt = load_ground_truth(GT_PATH)

source1 = pd.read_parquet(S1_PATH)
source2 = pd.read_parquet(S2_PATH)
source3 = pd.read_parquet(S3_PATH)


pairs = create_combined_training_pairs(
    source1=source1,
    source2=source2,
    source3=source3,
    ground_truth=gt,
    negatives_per_positive=2,
    random_state=42
)


source1_records = source1.set_index(
    "entity_id"
).to_dict("index")

source2_records = source2.set_index(
    "entity_id"
).to_dict("index")

source3_records = source3.set_index(
    "entity_id"
).to_dict("index")


feature_rows = []


for _, pair in pairs.iterrows():

    source1_id = pair["source1_entity_id"]
    candidate_id = pair["candidate_entity_id"]

    record1 = source1_records[source1_id]

    if candidate_id in source2_records:
        record2 = source2_records[candidate_id]
    else:
        record2 = source3_records[candidate_id]

    features = build_pair_features(
        record1,
        record2
    )

    features["source1_entity_id"] = source1_id
    features["candidate_entity_id"] = candidate_id
    features["label"] = pair["label"]

    feature_rows.append(features)


features_df = pd.DataFrame(feature_rows)


output_path = (
    r"data\stage_a\training_features_s2_s3_sample.parquet"
)

features_df.to_parquet(
    output_path,
    index=False
)


print("S2 pairs:", len(pairs[pairs["candidate_entity_id"].str.startswith("S2-")]))
print("S3 pairs:", len(pairs[pairs["candidate_entity_id"].str.startswith("S3-")]))

print("Training pairs:", len(pairs))
print("Feature rows:", len(features_df))
print("Feature columns:", len(features_df.columns))

print()
print("Labels:")
print(features_df["label"].value_counts())

print()
print("Source pair distribution:")

print(
    features_df[
        ["is_source2_pair", "is_source3_pair"]
    ].value_counts()
)

print()
print("Saved:", output_path)
