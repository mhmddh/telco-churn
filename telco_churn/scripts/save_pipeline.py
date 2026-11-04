"""Save the fitted winner and verify fresh-process inference for TP02 part 4."""

from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from telco_churn.scripts.train_pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    SEED,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_and_verify(selected, comparison, examples):
    """Persist the already-fitted object, then compare inference in another process."""
    destination = PROJECT_ROOT / "models" / "selected_pipeline"
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / "pipeline.joblib"
    joblib.dump(selected, artifact)

    # Keep a label-free batch and the original scores for reproducible reload evidence.
    batch_path = destination / "example_customers.csv"
    examples[FEATURES].to_csv(batch_path, index=False)
    positive_column = list(selected.classes_).index(1)
    original_scores = selected.predict_proba(examples)[:, positive_column]
    original = pd.DataFrame({
        "score": original_scores, "predicted": (original_scores >= 0.5).astype(int)
    })
    original.to_csv(destination / "original_predictions.csv", index=False)
    loaded_path = destination / "reloaded_predictions.csv"
    # The child command loads only the artifact and feature CSV: no source CSV or labels.
    subprocess.run([
        sys.executable, "-m", "telco_churn.scripts.predict_pipeline",
        "--artifact", str(artifact), "--input", str(batch_path), "--output", str(loaded_path),
    ], cwd=PROJECT_ROOT, check=True)
    reloaded = pd.read_csv(loaded_path)
    np.testing.assert_allclose(original_scores, reloaded["score"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(original["predicted"], reloaded["predicted"])

    allocations = pd.read_csv(PROJECT_ROOT / "reports" / "customer_splits.csv")
    allocations.to_csv(destination / "customer_splits.csv", index=False)
    source_files = sorted((PROJECT_ROOT / "telco_churn").rglob("*.py"))
    source_files += [PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "uv.lock"]
    selected_result = next(
        row for row in comparison["results"] if row["model"] == comparison["selected_model"]
    )
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "artifact_sha256": sha256(artifact),
        "source": {"path": "data/raw/telco.csv",
                   "sha256": sha256(PROJECT_ROOT / "data/raw/telco.csv")},
        "seed": SEED,
        "customer_allocations": {
            name: group["customerID"].tolist() for name, group in allocations.groupby("split")
        },
        "feature_roles": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
                          "identifier": "customerID", "target": "Churn"},
        "input_schema": {
            "numeric": "numbers or numeric text; missing accepted; blank TotalCharges -> missing",
            "categorical": "text or missing; unknown values ignored by encoder",
            "boundary": "apply parse_features before prediction; all six columns required",
        },
        "preprocessing": {
            "numeric": "training median imputation then StandardScaler",
            "categorical": "constant 'missing' imputation then OneHotEncoder(handle_unknown='ignore')",
        },
        "model": selected_result,
        "selection_rule": comparison["primary_metric"],
        "decision": comparison["decision"],
        "positive_class": "Churn = Yes (1); No = 0",
        "threshold": 0.5,
        "confusion_matrix_order": comparison["confusion_matrix_order"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "packages": {name: version(name) for name in
                                     ["scikit-learn", "pandas", "numpy", "scipy", "joblib"]}},
        "code_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): sha256(p) for p in source_files},
        "reproduction_command": "uv run python -m telco_churn.scripts.train_pipeline --save",
        "reload_check": {"fresh_process": True, "rows": len(examples),
                         "class_predictions_equal": True, "probability_atol": 1e-12,
                         "probability_rtol": 0,
                         "max_probability_difference": float(
                             np.max(np.abs(original_scores - reloaded["score"].to_numpy()))
                         )},
        "test_evaluated": False,
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved fitted pipeline and metadata to {destination}")
    print("Fresh-process reload verified: equal classes; probabilities agree within 1e-12.")
