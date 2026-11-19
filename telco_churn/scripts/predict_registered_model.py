"""Load the TP03 candidate alias and predict without training or labels."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from telco_churn.scripts.train_pipeline import TRACKING_URI, parse_features


def predict_batch(model_uri, customers):
    """Apply input parsing and predict from a registered MLflow model alias."""
    mlflow.set_tracking_uri(TRACKING_URI)
    pipeline = mlflow.sklearn.load_model(model_uri)
    features = parse_features(customers)
    positive_column = list(pipeline.classes_).index(1)
    scores = pipeline.predict_proba(features)[:, positive_column]
    return pd.DataFrame({"score": scores, "predicted": (scores >= 0.5).astype(int)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri", default="models:/telco-churn@candidate",
        help="Registered MLflow model URI to load",
    )
    parser.add_argument("--input", required=True, type=Path, help="CSV containing the six input columns")
    parser.add_argument("--output", required=True, type=Path, help="Destination predictions CSV")
    args = parser.parse_args()
    customers = pd.read_csv(args.input)
    predictions = predict_batch(args.model_uri, customers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Loaded: {args.model_uri}")
    print(f"Saved {len(predictions)} predictions to: {args.output}")


if __name__ == "__main__":
    main()
