"""TP02 milestone 1: two-feature model and training-only majority baseline."""

# Import data handling, classifiers, splitting and evaluation tools.
import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Configure the project path, reproducible seed and two input features.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED = 42
FEATURES = ["tenure", "MonthlyCharges"]


def main():
    # 1. Load the original CSV without modifying the source file.
    source = PROJECT_ROOT / "data" / "raw" / "telco.csv"
    df = pd.read_csv(source)
    # 2. Validate customer identifiers, target labels and numeric inputs.
    if df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique.")
    if df["Churn"].isna().any() or not df["Churn"].isin(["No", "Yes"]).all():
        raise ValueError("Churn must contain only No and Yes.")
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="raise")
    if df[FEATURES].isna().any().any():
        raise ValueError("The first model requires nonmissing numeric features.")

    # 3. Allocate 60% to training, then split the remaining 40% equally.
    # Stratification preserves class proportions; the seed fixes the allocation.
    train, remaining = train_test_split(
        df, test_size=0.4, stratify=df["Churn"], random_state=SEED
    )
    valid, test = train_test_split(
        remaining, test_size=0.5, stratify=remaining["Churn"], random_state=SEED
    )
    # 4. Check that no customer appears in more than one split.
    splits = {"train": train, "validation": valid, "test": test}
    ids = {name: set(frame["customerID"]) for name, frame in splits.items()}
    if any(ids[a] & ids[b] for a, b in [("train", "validation"),
                                      ("train", "test"), ("validation", "test")]):
        raise ValueError("Customer allocations overlap.")

    # 5. Prepare inputs and labels: churn (Yes) is the positive class, encoded as 1.
    # customerID and Churn are excluded from inputs; test data stays reserved.
    X_train, X_valid = train[FEATURES], valid[FEATURES]
    y_train = train["Churn"].map({"No": 0, "Yes": 1})
    y_valid = valid["Churn"].map({"No": 0, "Yes": 1})
    # Keep customer IDs alongside actual labels and each model's predictions.
    predictions = valid[["customerID"]].copy()
    predictions["actual"] = y_valid
    results = {}
    # 6. Compare an always-majority baseline with a learned logistic classifier.
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
    }
    # 7. Fit only on training data and evaluate predictions only on validation.
    # Confusion matrix rows = actual, columns = predicted, ordered [No, Yes].
    for name, model in models.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_valid)
        predictions[name] = predicted
        results[name] = {
            "accuracy": float(accuracy_score(y_valid, predicted)),
            "confusion_matrix": confusion_matrix(y_valid, predicted, labels=[0, 1]).tolist(),
        }

    # 8. Save customer allocations and validation predictions as local CSV reports.
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    # Record the fixed customer allocation required by part 1.
    pd.concat([
        frame[["customerID"]].assign(split=name) for name, frame in splits.items()
    ]).to_csv(report_dir / "customer_splits.csv", index=False)
    predictions.to_csv(report_dir / "first_model_validation_predictions.csv", index=False)
    # 9. Record metrics and their interpretation in a machine-readable JSON report.
    report = {
        "seed": SEED,
        "features": FEATURES,
        "positive_class": "Yes (1); No = 0",
        "confusion_matrix_order": "rows = actual, columns = predicted; labels = [0, 1]",
        "split_sizes": {name: len(frame) for name, frame in splits.items()},
        "customer_ids_disjoint": True,
        "evaluation_split": "validation; test reserved and not evaluated",
        "models": results,
    }
    (report_dir / "first_model_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    # 10. Display validation results and matrix labels in the terminal.
    print("Positive class: Yes = 1. Matrix rows: actual; columns: predicted; order: [No, Yes].")
    print(f"Split sizes: {report['split_sizes']}; seed: {SEED}")
    for name, metrics in results.items():
        print(f"\n{name}: validation accuracy = {metrics['accuracy']:.4f}")
        print(metrics["confusion_matrix"])
    print("\nReports saved in reports/. Test set has not been evaluated.")


# Run training only when this module is executed, not when it is imported.
if __name__ == "__main__":
    main()
