"""TP02 milestones 2-4: train, compare and save complete sklearn pipelines."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
    # real contract or payment type. If seen during fitting, "missing" gets an indicator;
    # otherwise it is an unknown category and encodes as all zeros for that feature.
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


def evaluate_model(candidate, X_valid, y_valid):
    """Use class-1 probabilities for AP and a fixed 0.5 threshold for class metrics."""
    positive_column = list(candidate.classes_).index(1)
    scores = candidate.predict_proba(X_valid)[:, positive_column]
    predicted = (scores >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_valid, predicted, average="binary", pos_label=1, zero_division=0
    )
    return {
        "average_precision": float(average_precision_score(y_valid, scores)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy_score(y_valid, predicted)),
        "confusion_matrix": confusion_matrix(y_valid, predicted, labels=[0, 1]).tolist(),
    }, scores, predicted


def compare_models(X_train, y_train, X_valid, y_valid, customer_ids, report_dir):
    """Part 3: compare fresh pipelines without accessing test data."""
    # Policy chosen before evaluation: highest validation AP among learned models.
    # Exact ties favor the earlier, simpler model in this dictionary.
    # AP measures churn ranking; class metrics use the same 0.5 threshold throughout.
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=1
        ),
    }
    results = []
    fitted = {}
    predictions = pd.DataFrame({"customerID": customer_ids, "actual": y_valid})
    for name, classifier in models.items():
        # Each candidate owns a fresh preprocessor; no fitted state is shared.
        candidate = build_pipeline(classifier)
        start = perf_counter()
        candidate.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        fitted[name] = candidate
        metrics, scores, predicted = evaluate_model(candidate, X_valid, y_valid)
        results.append({
            "model": name,
            "fit_seconds": fit_seconds,
            "parameters": classifier.get_params(),
            **metrics,
        })
        predictions[f"{name}_score"] = scores
        predictions[f"{name}_predicted"] = predicted

    learned = [row for row in results if row["model"] != "majority_baseline"]
    selected = max(learned, key=lambda row: row["average_precision"])
    decision = (
        f"Selected {selected['model']} with validation average precision "
        f"{selected['average_precision']:.4f}, the highest among the learned candidates. "
        "The test set was not evaluated."
    )
    report = {
        "primary_metric": "validation average precision (higher is better)",
        "justification": "Evaluate churn ranking despite the majority non-churn class.",
        "threshold": 0.5,
        "positive_class": "Yes = 1; No = 0",
        "score_api": "predict_proba, column corresponding to class 1 in classes_",
        "zero_division": 0,
        "tie_policy": "exact AP ties favor logistic regression, then tree, then forest",
        "confusion_matrix_order": "rows actual, columns predicted; labels [0, 1]",
        "seed": SEED,
        "features": FEATURES,
        "allocation_source": "reports/customer_splits.csv",
        "split_sizes": {"train": len(X_train), "validation": len(X_valid)},
        "fit_timing": "wall-clock pipeline fit seconds on this machine; descriptive only",
        "selected_model": selected["model"],
        "decision": decision,
        "results": results,
    }
    (report_dir / "model_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    table = pd.DataFrame(results)
    # Serialize nested cells explicitly for a usable CSV comparison table.
    table["parameters"] = table["parameters"].map(json.dumps)
    table["confusion_matrix"] = table["confusion_matrix"].map(json.dumps)
    table.to_csv(report_dir / "model_comparison.csv", index=False)
    predictions.to_csv(report_dir / "model_comparison_predictions.csv", index=False)
    (report_dir / "model_selection.txt").write_text(
        "Criterion chosen in advance: highest validation average precision.\n"
        + decision + "\n", encoding="utf-8"
    )
    print(table[["model", "average_precision", "precision", "recall", "f1", "accuracy",
                 "fit_seconds"]].to_string(index=False))
    print(decision)
    return fitted[selected["model"]], report


def load_training_data():
    """Validate the source and recover the fixed allocations without resplitting."""
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "telco.csv")
    if not df["Churn"].isin(["No", "Yes"]).all():
        raise ValueError("Churn must contain only No and Yes.")

    # Reuse the milestone 1 allocation rather than splitting again.
    report_dir = PROJECT_ROOT / "reports"
    allocations = pd.read_csv(report_dir / "customer_splits.csv")
    if (
        set(allocations["customerID"]) != set(df["customerID"])
        or allocations["customerID"].duplicated().any()
        or df["customerID"].isna().any()
        or df["customerID"].duplicated().any()
        or allocations["split"].isna().any()
        or set(allocations["split"]) != {"train", "validation", "test"}
    ):
        raise ValueError("customer_splits.csv must list each source customer exactly once.")
    train = load_split(df, allocations, "train")
    valid = load_split(df, allocations, "validation")

    # Churn = Yes is the positive class. customerID and Churn are not inputs.
    X_train, X_valid = parse_features(train), parse_features(valid)
    y_train = train["Churn"].map({"No": 0, "Yes": 1})
    y_valid = valid["Churn"].map({"No": 0, "Yes": 1})
    return X_train, y_train, X_valid, y_valid, valid["customerID"], report_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Run the part 3 comparison")
    parser.add_argument("--save", action="store_true", help="Compare, save and verify the winner")
    args = parser.parse_args()
    X_train, y_train, X_valid, y_valid, customer_ids, report_dir = load_training_data()

    if args.compare or args.save:
        selected, comparison = compare_models(
            X_train, y_train, X_valid, y_valid, customer_ids, report_dir
        )
        if args.save:
            from telco_churn.scripts.save_pipeline import save_and_verify

            save_and_verify(selected, comparison, X_valid.head(5))
        return

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
        "split_sizes": {"train": len(X_train), "validation": len(X_valid)},
        "evaluation_split": "validation; test reserved and not evaluated",
        "metrics": metrics,
    }
    (report_dir / "pipeline_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame({"customerID": customer_ids, "actual": y_valid, "predicted": predicted}).to_csv(
        report_dir / "pipeline_validation_predictions.csv", index=False
    )

    print("Positive class: Yes = 1. Matrix rows: actual; columns: predicted; order: [No, Yes].")
    print(f"Split sizes: {report['split_sizes']}; seed: {SEED}")
    print(f"Validation accuracy = {metrics['accuracy']:.4f}")
    print(metrics["confusion_matrix"])
    print("Reports saved in reports/. Test set has not been evaluated.")


if __name__ == "__main__":
    main()
