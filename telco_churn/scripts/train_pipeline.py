"""TP02 milestone 2: six-feature preprocessing and classifier in one sklearn Pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MISSING_CATEGORY = "missing"


def parse_features(df):
    """Return the six features with typed numerics. Stateless: learns nothing from the data.

    Whitespace-only TotalCharges becomes missing; any other non-numeric text raises an
    error, so malformed values are never silently accepted as missing. Inference must
    call this too. A missing required column raises a KeyError.
    """
    features = df[FEATURES].copy()
    features["TotalCharges"] = features["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise")
    return features


def build_pipeline(classifier):
    """Return a new unfitted pipeline. It reads no data and trains nothing."""
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    # A missing category is not replaced by the most frequent one, which would claim a
    # real contract or payment type. It encodes as all zeros, the same as an unseen category.
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def load_split(df, allocations, split):
    """Return one split's rows using the customer allocation saved by milestone 1."""
    ids = allocations.loc[allocations["split"] == split, "customerID"]
    return df[df["customerID"].isin(ids)]


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "telco.csv")
    if not df["Churn"].isin(["No", "Yes"]).all():
        raise ValueError("Churn must contain only No and Yes.")

    # Reuse the milestone 1 allocation rather than splitting again.
    report_dir = PROJECT_ROOT / "reports"
    allocations = pd.read_csv(report_dir / "customer_splits.csv")
    if (
        set(allocations["customerID"]) != set(df["customerID"])
        or allocations["customerID"].duplicated().any()
    ):
        raise ValueError("customer_splits.csv must list each source customer exactly once.")
    train = load_split(df, allocations, "train")
    valid = load_split(df, allocations, "validation")

    # Churn = Yes is the positive class. customerID and Churn are not inputs.
    X_train, X_valid = parse_features(train), parse_features(valid)
    y_train = train["Churn"].map({"No": 0, "Yes": 1})
    y_valid = valid["Churn"].map({"No": 0, "Yes": 1})

    # One fit call learns imputation, scaling, encoding and the classifier from training only.
    pipe = build_pipeline(LogisticRegression(max_iter=1000, random_state=SEED))
    pipe.fit(X_train, y_train)
    predicted = pipe.predict(X_valid)

    metrics = {
        "accuracy": float(accuracy_score(y_valid, predicted)),
        "confusion_matrix": confusion_matrix(y_valid, predicted, labels=[0, 1]).tolist(),
    }
    report = {
        "seed": SEED,
        "allocation_source": "reports/customer_splits.csv",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "preprocessing": {
            "parsing": "whitespace-only TotalCharges -> missing; malformed numeric text raises",
            "numeric": "median imputation, then StandardScaler",
            "categorical": f"missing -> '{MISSING_CATEGORY}', then one-hot encoding",
            "unknown_category": "ignored: all-zero encoding for that feature",
        },
        "classifier": "LogisticRegression(max_iter=1000)",
        "positive_class": "Yes (1); No = 0",
        "confusion_matrix_order": "rows = actual, columns = predicted; labels = [0, 1]",
        "split_sizes": {"train": len(train), "validation": len(valid)},
        "evaluation_split": "validation; test reserved and not evaluated",
        "metrics": metrics,
    }
    (report_dir / "pipeline_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    valid[["customerID"]].assign(actual=y_valid, predicted=predicted).to_csv(
        report_dir / "pipeline_validation_predictions.csv", index=False
    )

    print("Positive class: Yes = 1. Matrix rows: actual; columns: predicted; order: [No, Yes].")
    print(f"Split sizes: {report['split_sizes']}; seed: {SEED}")
    print(f"Validation accuracy = {metrics['accuracy']:.4f}")
    print(metrics["confusion_matrix"])
    print("Reports saved in reports/. Test set has not been evaluated.")


if __name__ == "__main__":
    main()
