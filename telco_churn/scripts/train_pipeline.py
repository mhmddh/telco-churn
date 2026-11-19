"""TP02 milestones 2-4: train, compare and save complete sklearn pipelines."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
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
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "telco-churn-tp03"


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


def candidate_classifiers():
    """Return new, unfitted candidate estimators used by TP02 and TP03."""
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=SEED, n_jobs=1
        ),
    }


def source_sha256(path):
    """Return the SHA-256 fingerprint of an immutable source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_identity():
    """Return the commit and whether tracked files were edited during this run."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            check=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True,
            check=True, text=True
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", "unavailable"
    return commit, str(dirty).lower()


def track_candidate(name, classifier, train, valid, X_train, y_train, X_valid, y_valid):
    """Fit and record one validation run with MLflow; test data is never loaded here."""
    import mlflow

    source = PROJECT_ROOT / "data" / "raw" / "telco.csv"
    allocations = PROJECT_ROOT / "reports" / "customer_splits.csv"
    commit, dirty = git_identity()
    classifier_params = classifier.get_params()
    # MLflow parameters are flat, so prefix estimator settings and record their actual values.
    parameters = {
        "model_type": type(classifier).__name__,
        "split_seed": SEED,
        "decision_threshold": 0.5,
        **{f"estimator__{key}": str(value) for key, value in classifier_params.items()},
    }
    provenance = {
        "source": {"path": "data/raw/telco.csv", "sha256": source_sha256(source)},
        "customer_allocations": {"path": "reports/customer_splits.csv",
                                 "sha256": source_sha256(allocations)},
        "feature_roles": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
                          "identifier_excluded": "customerID", "target_excluded": "Churn"},
        "class_mapping": {"No": 0, "Yes": 1},
        "evaluation": {
            "split": "validation", "primary_metric": "average precision",
            "scores": "predict_proba column associated with class 1", "threshold": 0.5,
            "confusion_matrix_order": "rows actual; columns predicted; labels [0, 1]",
            "test_set_evaluated": False,
        },
        "preprocessing": {
            "numeric": "median imputation then StandardScaler",
            "categorical": "constant 'missing' imputation then OneHotEncoder(handle_unknown='ignore')",
            "unknown_category": "ignored by encoder",
        },
        "git": {"commit": commit, "working_tree_dirty": dirty},
    }
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    train_dataset = mlflow.data.from_pandas(train, name="telco_train", targets="Churn")
    valid_dataset = mlflow.data.from_pandas(valid, name="telco_validation", targets="Churn")
    with mlflow.start_run(run_name=f"{name}-tp03") as run:
        mlflow.log_params(parameters)
        mlflow.set_tags({
            "candidate_name": name,
            "run_purpose": "candidate_comparison",
            "git_commit": commit,
            "git_working_tree_dirty": dirty,
            "data_sha256": provenance["source"]["sha256"],
            "allocation_sha256": provenance["customer_allocations"]["sha256"],
        })
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(valid_dataset, context="validation")
        candidate = build_pipeline(classifier)
        start = perf_counter()
        candidate.fit(X_train, y_train)
        fit_seconds = perf_counter() - start
        metrics, _, _ = evaluate_model(candidate, X_valid, y_valid)
        mlflow.log_metrics({
            **{f"val_{metric}": value for metric, value in metrics.items()
               if metric != "confusion_matrix"},
            "fit_seconds": fit_seconds,
        })
        with TemporaryDirectory() as temporary:
            matrix_path = Path(temporary) / "validation_confusion_matrix.json"
            matrix_path.write_text(json.dumps({
                "labels": [0, 1], "rows": "actual", "columns": "predicted",
                "matrix": metrics["confusion_matrix"],
            }, indent=2) + "\n", encoding="utf-8")
            mlflow.log_artifact(str(matrix_path))
        mlflow.log_dict(provenance, "provenance.json")
        print(f"Run ID: {run.info.run_id}")
        print(f"Validation average precision: {metrics['average_precision']:.4f}")
        return {"run_id": run.info.run_id, "candidate": name, "fit_seconds": fit_seconds, **metrics}


def track_candidate_comparison(train, valid, X_train, y_train, X_valid, y_valid, report_dir):
    """Create three fair MLflow candidate runs and save their validation-only decision."""
    names = ["logistic_regression", "decision_tree", "random_forest"]
    classifiers = candidate_classifiers()
    records = [
        track_candidate(name, classifiers[name], train, valid, X_train, y_train, X_valid, y_valid)
        for name in names
    ]
    selected = max(records, key=lambda record: record["average_precision"])
    table = pd.DataFrame(records)
    table["decision"] = "not selected"
    table.loc[table["run_id"] == selected["run_id"], "decision"] = "selected: highest validation AP"
    table.to_csv(report_dir / "tp03_candidate_comparison.csv", index=False)
    selection = (
        f"Selected {selected['candidate']} (run {selected['run_id']}) because its validation "
        f"average precision was highest at {selected['average_precision']:.4f}. "
        "All candidates used the same six features, saved customer allocation, validation "
        "population and threshold 0.5. The test set was not evaluated. This single validation "
        "split does not establish a statistically significant difference."
    )
    (report_dir / "tp03_candidate_selection.md").write_text(
        "# TP03 candidate selection\n\n" + selection + "\n", encoding="utf-8"
    )
    print("\nMLflow candidate comparison:")
    print(table[["candidate", "run_id", "average_precision", "recall", "f1", "decision"]]
          .to_string(index=False))
    print(selection)
    return selected


def log_selected_model(X_train, y_train, X_valid, y_valid, report_dir):
    """Log the fitted selected TP03 candidate in the exact MLflow run that selected it."""
    import mlflow
    import mlflow.sklearn

    comparison_path = report_dir / "tp03_candidate_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Run --track-all before logging the selected model.")
    comparison = pd.read_csv(comparison_path)
    selected_rows = comparison.loc[
        comparison["decision"].eq("selected: highest validation AP")
    ]
    if len(selected_rows) != 1:
        raise ValueError("Comparison must contain exactly one selected candidate.")
    selected = selected_rows.iloc[0]
    name, run_id = selected["candidate"], selected["run_id"]
    classifier = candidate_classifiers()[name]
    candidate = build_pipeline(classifier)
    candidate.fit(X_train, y_train)
    metrics, _, _ = evaluate_model(candidate, X_valid, y_valid)
    for metric in ["average_precision", "precision", "recall", "f1", "accuracy"]:
        np.testing.assert_allclose(metrics[metric], selected[metric], rtol=0, atol=1e-12)

    mlflow.set_tracking_uri(TRACKING_URI)
    existing_run = mlflow.MlflowClient().get_run(run_id)
    if existing_run.data.tags.get("candidate_name") != name:
        raise ValueError("Selected comparison row does not match the source MLflow run.")
    # Resume this exact source run: the freshly fitted pipeline is verified above to
    # reproduce its stored validation metrics before model logging happens.
    with mlflow.start_run(run_id=run_id):
        model_info = mlflow.sklearn.log_model(
            sk_model=candidate,
            name="churn_pipeline",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )
        mlflow.set_tags({"logged_model_name": "churn_pipeline", "model_logged": "true"})
    handover = {
        "selected_candidate": name,
        "source_run_id": run_id,
        "model_uri": model_info.model_uri,
        "input_example": "first three typed, unencoded training feature rows",
        "input_columns": FEATURES,
        "class_mapping": {"No": 0, "Yes": 1},
        "threshold": 0.5,
        "selection_metric": "validation average precision",
        "validation_metrics_reproduced_before_logging": metrics,
        "registered_model": None,
        "alias": None,
    }
    handover_path = report_dir / "tp03_model_handover.json"
    handover_path.write_text(json.dumps(handover, indent=2) + "\n", encoding="utf-8")
    print(f"Source run ID: {run_id}")
    print(f"Logged model URI: {model_info.model_uri}")
    print(f"Handover record: {handover_path}")
    return handover


def register_selected_model(report_dir):
    """Register the logged selected model and point the candidate alias at its version."""
    import mlflow

    handover_path = report_dir / "tp03_model_handover.json"
    if not handover_path.exists():
        raise FileNotFoundError("Run --log-selected-model before model registration.")
    handover = json.loads(handover_path.read_text(encoding="utf-8"))
    model_uri = handover["model_uri"]
    registered_name = "telco-churn"
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    existing = handover.get("registered_model")
    if existing and existing.get("name") == registered_name:
        version = client.get_model_version(registered_name, existing["version"])
        if version.source != model_uri:
            raise ValueError("Handover version does not point to the recorded logged model URI.")
    else:
        version = mlflow.register_model(model_uri, registered_name)
    client.set_registered_model_alias(registered_name, "candidate", version.version)
    handover["registered_model"] = {"name": registered_name, "version": version.version}
    handover["alias"] = {"name": "candidate", "uri": f"models:/{registered_name}@candidate"}
    handover_path.write_text(json.dumps(handover, indent=2) + "\n", encoding="utf-8")
    print(f"Registered model: {registered_name}")
    print(f"Registered version: {version.version}")
    print(f"Alias URI: models:/{registered_name}@candidate")
    return handover


def verify_registered_model(X_train, y_train, X_valid, report_dir):
    """Compare original scores with alias inference in a separate Python process."""
    handover_path = report_dir / "tp03_model_handover.json"
    if not handover_path.exists():
        raise FileNotFoundError("Run --register-selected-model before alias verification.")
    handover = json.loads(handover_path.read_text(encoding="utf-8"))
    alias = handover.get("alias")
    if not alias:
        raise ValueError("No registered model alias is recorded in the handover file.")
    classifier = candidate_classifiers()[handover["selected_candidate"]]
    original_pipeline = build_pipeline(classifier).fit(X_train, y_train)
    examples = X_valid.head(5).copy()
    positive_column = list(original_pipeline.classes_).index(1)
    original_scores = original_pipeline.predict_proba(examples)[:, positive_column]
    original = pd.DataFrame({
        "score": original_scores, "predicted": (original_scores >= 0.5).astype(int),
    })
    examples_path = report_dir / "tp03_inference_examples.csv"
    original_path = report_dir / "tp03_original_predictions.csv"
    reloaded_path = report_dir / "tp03_alias_predictions.csv"
    examples.to_csv(examples_path, index=False)
    original.to_csv(original_path, index=False)
    # The child command imports no training entry point, calls no fit method and has no labels.
    subprocess.run([
        sys.executable, "-m", "telco_churn.scripts.predict_registered_model",
        "--model-uri", alias["uri"], "--input", str(examples_path), "--output", str(reloaded_path),
    ], cwd=PROJECT_ROOT, check=True)
    reloaded = pd.read_csv(reloaded_path)
    np.testing.assert_array_equal(original["predicted"], reloaded["predicted"])
    np.testing.assert_allclose(original["score"], reloaded["score"], rtol=0, atol=1e-12)
    max_difference = float(np.max(np.abs(original["score"] - reloaded["score"])))
    handover["fresh_process_check"] = {
        "command": "uv run python -m telco_churn.scripts.predict_registered_model "
                   "--input reports/tp03_inference_examples.csv "
                   "--output reports/tp03_alias_predictions.csv",
        "examples": len(examples), "labels_provided": False, "fit_called": False,
        "class_predictions_equal": True, "probability_atol": 1e-12,
        "probability_rtol": 0, "max_probability_difference": max_difference,
    }
    handover_path.write_text(json.dumps(handover, indent=2) + "\n", encoding="utf-8")
    print("Fresh-process alias load verified: equal classes; probabilities agree within 1e-12.")
    return handover


def compare_models(X_train, y_train, X_valid, y_valid, customer_ids, report_dir):
    """Part 3: compare fresh pipelines without accessing test data."""
    # Policy chosen before evaluation: highest validation AP among learned models.
    # Exact ties favor the earlier, simpler model in this dictionary.
    # AP measures churn ranking; class metrics use the same 0.5 threshold throughout.
    models = candidate_classifiers()
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
    return X_train, y_train, X_valid, y_valid, valid["customerID"], report_dir, train, valid


def main():
    # MLflow 3 prints a Unicode run link when it closes a run. Use UTF-8 on Windows
    # terminals so that display-only message cannot turn a successful run into an error.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Run the part 3 comparison")
    parser.add_argument("--save", action="store_true", help="Compare, save and verify the winner")
    parser.add_argument(
        "--track", choices=candidate_classifiers().keys(),
        help="Track one candidate in MLflow for TP03 parts 2 and 3",
    )
    parser.add_argument(
        "--track-all", action="store_true",
        help="Track the three learned candidate configurations for TP03 part 3",
    )
    parser.add_argument(
        "--log-selected-model", action="store_true",
        help="Log the selected pipeline in its source MLflow run for TP03 part 4",
    )
    parser.add_argument(
        "--register-selected-model", action="store_true",
        help="Register the logged selected model and assign the candidate alias",
    )
    parser.add_argument(
        "--verify-registered-model", action="store_true",
        help="Load the candidate alias in a new process and compare predictions",
    )
    args = parser.parse_args()
    X_train, y_train, X_valid, y_valid, customer_ids, report_dir, train, valid = load_training_data()

    if args.track:
        track_candidate(
            args.track, candidate_classifiers()[args.track], train, valid,
            X_train, y_train, X_valid, y_valid,
        )
        return

    if args.track_all:
        track_candidate_comparison(train, valid, X_train, y_train, X_valid, y_valid, report_dir)
        return

    if args.log_selected_model:
        log_selected_model(X_train, y_train, X_valid, y_valid, report_dir)
        return

    if args.register_selected_model:
        register_selected_model(report_dir)
        return

    if args.verify_registered_model:
        verify_registered_model(X_train, y_train, X_valid, report_dir)
        return

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
