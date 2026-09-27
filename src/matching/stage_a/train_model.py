import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import fbeta_score


FEATURE_PATH = r"data\stage_a\training_features_s2_s3_sample.parquet"


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
        predictions = (probabilities >= threshold).astype(int)

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


df = pd.read_parquet(FEATURE_PATH)

X = df[FEATURE_COLUMNS]
y = df["label"]
groups = df["source1_entity_id"]


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


oof_probabilities = pd.Series(
    index=df.index,
    dtype=float
)


fold_results = []


for fold, (train_idx, valid_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

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
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_valid
    )[:, 1]

    oof_probabilities.iloc[valid_idx] = probabilities

    threshold, score = find_best_threshold(
        y_valid,
        probabilities
    )

    fold_results.append({
        "fold": fold,
        "threshold": threshold,
        "f0.5": score,
        "validation_rows": len(valid_idx),
        "validation_positive": int(y_valid.sum()),
    })

    print(
        f"Fold {fold}: "
        f"F0.5={score:.4f}, "
        f"threshold={threshold:.2f}, "
        f"rows={len(valid_idx)}, "
        f"positive={int(y_valid.sum())}"
    )


# ---------------------------------------------------------
# OVERALL OOF METRIC
# ---------------------------------------------------------

global_threshold, global_f05 = find_best_threshold(
    y,
    oof_probabilities.values
)


# ---------------------------------------------------------
# SOURCE-SPECIFIC METRICS
# ---------------------------------------------------------

s2_mask = df["is_source2_pair"] == 1
s3_mask = df["is_source3_pair"] == 1


s2_threshold, s2_f05 = find_best_threshold(
    y[s2_mask],
    oof_probabilities[s2_mask]
)

s3_threshold, s3_f05 = find_best_threshold(
    y[s3_mask],
    oof_probabilities[s3_mask]
)


print()
print("===================================")
print("STAGE-A LIGHTGBM CROSS VALIDATION")
print("===================================")

print()
print("Rows:", len(df))
print("Features:", len(FEATURE_COLUMNS))
print("Positive:", int(y.sum()))
print("Negative:", int((y == 0).sum()))

print()
print("Overall OOF F0.5:", round(global_f05, 4))
print("Overall OOF threshold:", round(global_threshold, 4))

print()
print("S2 OOF F0.5:", round(s2_f05, 4))
print("S2 threshold:", round(s2_threshold, 4))
print("S2 rows:", int(s2_mask.sum()))
print("S2 positives:", int(y[s2_mask].sum()))

print()
print("S3 OOF F0.5:", round(s3_f05, 4))
print("S3 threshold:", round(s3_threshold, 4))
print("S3 rows:", int(s3_mask.sum()))
print("S3 positives:", int(y[s3_mask].sum()))

print()
print("Fold results:")

for result in fold_results:
    print(result)


# ---------------------------------------------------------
# SAVE OOF PREDICTIONS
# ---------------------------------------------------------

oof_output = df[
    [
        "source1_entity_id",
        "candidate_entity_id",
        "label",
        "is_source2_pair",
        "is_source3_pair",
    ]
].copy()

oof_output["oof_probability"] = oof_probabilities.values

oof_output.to_parquet(
    r"data\stage_a\oof_predictions_s2_s3_sample.parquet",
    index=False
)

print()
print(
    "Saved OOF predictions: "
    r"data\stage_a\oof_predictions_s2_s3_sample.parquet"
)