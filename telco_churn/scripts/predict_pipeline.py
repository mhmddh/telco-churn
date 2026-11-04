"""Predict from a trusted fitted artifact without training or training labels."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from telco_churn.scripts.train_pipeline import parse_features


def predict_batch(artifact, customers):
    """Parse original feature columns, then apply the saved preprocessing and model."""
    pipeline = joblib.load(artifact)  # Load only artifacts from a trusted source.
    features = parse_features(customers)
    positive_column = list(pipeline.classes_).index(1)
    scores = pipeline.predict_proba(features)[:, positive_column]
    return pd.DataFrame({"score": scores, "predicted": (scores >= 0.5).astype(int)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="CSV with the six feature columns")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = predict_batch(args.artifact, pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
