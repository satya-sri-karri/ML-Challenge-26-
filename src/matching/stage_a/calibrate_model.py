import os
import joblib
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import fbeta_score
from sklearn.isotonic import IsotonicRegression


FEATURE_PATH = r"data\stage_a\training_features_s2_s3_sample.parquet"
MODEL_DIR = r"models\stage_a"

os.makedirs(MODEL_DIR, exist_ok=True)


FEATURE_COLUMNS = [
    "name_jaro_winkler",
    "name_levenshtein",
    "name_token_jaccard",
    "name_containment",
    "name_prefix_similarity",
    "name_char_ngram_cosine",
    "name_phonetic_match",
    "address_token_jaccard",
    "address_number_tolerant_match",
    "address_string_similarity",
    "country_match",
    "name1_empty",
    "name2_empty",
    "address1_empty",
    "address2_empty",
    "is_source2_pair",
    "is_source3_pair",
]


def find_best_threshold(y_true, probabilities):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in [i / 100 for i in range(1, 100)]:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = fbeta_score(
            y_true,
            predictions,
            beta=0.5,
            zero_division=0
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


# --------------------------------------------------
# Load combined S2 + S3 development data
# --------------------------------------------------

df = pd.read_parquet(FEATURE_PATH)

X = df[FEATURE_COLUMNS]
y = df["label"]
groups = df["source1_entity_id"]


# --------------------------------------------------
# Grouped cross-validation
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


oof_probabilities = pd.Series(
    index=df.index,
    dtype=float
)


for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx]
    )

    probabilities = model.predict_proba(
        X.iloc[valid_idx]
    )[:, 1]

    oof_probabilities.iloc[valid_idx] = probabilities

    print(
        f"Calibration fold {fold} completed "
        f"({len(valid_idx)} validation rows)"
    )


# --------------------------------------------------
# Raw OOF threshold
# --------------------------------------------------

threshold, f05 = find_best_threshold(
    y,
    oof_probabilities.values
)


# --------------------------------------------------
# Isotonic calibration
# --------------------------------------------------

calibrator = IsotonicRegression(
    out_of_bounds="clip"
)

calibrator.fit(
    oof_probabilities.values,
    y.values
)


calibrated_oof = calibrator.predict(
    oof_probabilities.values
)


# --------------------------------------------------
# Calibrated threshold
# --------------------------------------------------

calibrated_threshold, calibrated_f05 = (
    find_best_threshold(
        y,
        calibrated_oof
    )
)


# --------------------------------------------------
# Source-specific calibrated metrics
# --------------------------------------------------

s2_mask = df["is_source2_pair"] == 1
s3_mask = df["is_source3_pair"] == 1


s2_threshold, s2_f05 = find_best_threshold(
    y[s2_mask],
    calibrated_oof[s2_mask]
)


s3_threshold, s3_f05 = find_best_threshold(
    y[s3_mask],
    calibrated_oof[s3_mask]
)


# --------------------------------------------------
# Train final model on all development data
# --------------------------------------------------

final_model = LGBMClassifier(
    objective="binary",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)


final_model.fit(
    X,
    y
)


# --------------------------------------------------
# Save model package
# --------------------------------------------------

model_package = {
    "model": final_model,
    "calibrator": calibrator,
    "feature_columns": FEATURE_COLUMNS,
    "threshold": calibrated_threshold,
    "training_rows": len(df),
    "positive_rows": int(y.sum()),
    "negative_rows": int((y == 0).sum()),
    "random_state": 42,
}


model_path = os.path.join(
    MODEL_DIR,
    "stage_a_lightgbm.joblib"
)


joblib.dump(
    model_package,
    model_path
)


# --------------------------------------------------
# Save combined OOF predictions
# --------------------------------------------------

oof_output = df[
    [
        "source1_entity_id",
        "candidate_entity_id",
        "label",
        "is_source2_pair",
        "is_source3_pair",
    ]
].copy()


oof_output["raw_probability"] = (
    oof_probabilities.values
)

oof_output["calibrated_score"] = (
    calibrated_oof
)


oof_output.to_parquet(
    r"data\stage_a\oof_predictions_s2_s3_sample.parquet",
    index=False
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("===================================")
print("STAGE-A CALIBRATION")
print("===================================")

print()
print("Rows:", len(df))
print("Positive:", int(y.sum()))
print("Negative:", int((y == 0).sum()))

print()
print("Raw OOF F0.5:", round(f05, 4))
print("Raw threshold:", round(threshold, 4))

print()
print("Calibrated OOF F0.5:", round(calibrated_f05, 4))
print(
    "Calibrated threshold:",
    round(calibrated_threshold, 4)
)

print()
print("S2 calibrated F0.5:", round(s2_f05, 4))
print("S2 threshold:", round(s2_threshold, 4))

print()
print("S3 calibrated F0.5:", round(s3_f05, 4))
print("S3 threshold:", round(s3_threshold, 4))

print()
print("Model saved:")
print(model_path)

print()
print("OOF predictions saved:")
print(
    r"data\stage_a\oof_predictions_s2_s3_sample.parquet"
)