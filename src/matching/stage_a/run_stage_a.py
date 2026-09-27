import argparse
import joblib
import pandas as pd

from src.matching.stage_a.candidate_loader import (
    load_candidate_pairs,
)

from src.matching.stage_a.inference import (
    predict_stage_a,
)


MODEL_PATH = (
    r"models\stage_a\stage_a_lightgbm.joblib"
)


def main():

    parser = argparse.ArgumentParser(
        description="Run Stage-A pairwise matcher"
    )

    parser.add_argument(
        "--source1",
        required=True
    )

    parser.add_argument(
        "--candidate-source",
        required=True
    )

    parser.add_argument(
        "--candidates",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    print("Loading Source1...")
    source1 = pd.read_parquet(
        args.source1
    )

    print("Loading candidate source...")
    candidate_source = pd.read_parquet(
        args.candidate_source
    )

    print("Loading candidate pairs...")
    candidate_pairs = load_candidate_pairs(
        args.candidates
    )

    print(
        "Candidate pairs:",
        len(candidate_pairs)
    )

    print("Loading Stage-A model...")
    model_package = joblib.load(
        MODEL_PATH
    )

    print("Building features and scoring...")

    predictions = predict_stage_a(
        source1,
        candidate_source,
        candidate_pairs,
        model_package
    )

    predictions.to_csv(
        args.output,
        sep="\t",
        index=False
    )

    print()
    print("==============================")
    print("STAGE-A COMPLETE")
    print("==============================")
    print(
        "Input candidates:",
        len(candidate_pairs)
    )
    print(
        "Scored candidates:",
        len(predictions)
    )
    print(
        "Output:",
        args.output
    )

    if not predictions.empty:
        print()
        print(
            "Score statistics:"
        )
        print(
            predictions[
                "calibrated_score"
            ].describe()
        )


if __name__ == "__main__":
    main()
