"""TP02 part 5: focused checks on the saved fitted pipeline, without retraining."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd

from telco_churn.scripts.train_pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    parse_features,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    bundle = PROJECT_ROOT / "models/selected_pipeline"
    metadata = json.loads((bundle / "metadata.json").read_text())
    artifact = bundle / "pipeline.joblib"
    require(hashlib.sha256(artifact.read_bytes()).hexdigest() == metadata["artifact_sha256"],
            "Artifact hash does not match metadata")
    source = PROJECT_ROOT / metadata["source"]["path"]
    require(hashlib.sha256(source.read_bytes()).hexdigest() == metadata["source"]["sha256"],
            "Source hash does not match metadata")
    pipe = joblib.load(artifact)  # This project's trusted artifact only.
    checks = {}

    def passed(name, evidence):
        checks[name] = {"passed": True, "evidence": evidence}
        print(f"PASS {name}: {evidence}")

    # 1. Check all allocations; do not evaluate the reserved test customers.
    df = pd.read_csv(source)
    allocations = pd.read_csv(bundle / "customer_splits.csv")
    require(not df.customerID.isna().any() and not df.customerID.duplicated().any(),
            "Source IDs must be present and unique")
    require(not allocations.customerID.isna().any()
            and not allocations.customerID.duplicated().any(), "Repeated or missing allocation ID")
    require(set(allocations.split) == {"train", "validation", "test"}, "Invalid split names")
    require(set(allocations.customerID) == set(df.customerID), "Incomplete allocations")
    groups = {name: set(group.customerID) for name, group in allocations.groupby("split")}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        require(not groups[a] & groups[b], f"Overlap: {a}, {b}")
    require(all(groups[name] == set(ids)
                for name, ids in metadata["customer_allocations"].items()),
            "Allocations differ from metadata")
    passed("disjoint_splits", {name: len(ids) for name, ids in groups.items()})

    # 2. Compare fitted numeric statistics with independently computed training statistics.
    train = parse_features(df[df.customerID.isin(groups["train"])])
    numeric = pipe.named_steps["preprocessor"].named_transformers_["num"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    medians = train[NUMERIC_FEATURES].median().to_numpy()
    np.testing.assert_allclose(imputer.statistics_, medians, rtol=0, atol=1e-12)
    imputed_train = train[NUMERIC_FEATURES].fillna(dict(zip(NUMERIC_FEATURES, medians)))
    np.testing.assert_allclose(scaler.mean_, imputed_train.mean(), rtol=0, atol=1e-10)
    full_medians = parse_features(df)[NUMERIC_FEATURES].median().to_numpy()
    require(not np.allclose(medians, full_medians),
            "Training/full medians coincide; this dataset cannot distinguish the two")
    passed("training_only_statistics", {
        "features": NUMERIC_FEATURES, "fitted_medians": imputer.statistics_.tolist(),
        "training_medians": medians.tolist(), "full_dataset_medians": full_medians.tolist(),
        "scaler_mean_matches_imputed_training": True,
    })

    # 3. Check numeric and categorical fitted statistics before and after prediction.
    encoder = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    before = deepcopy([imputer.statistics_, scaler.mean_, scaler.scale_, encoder.categories_])
    examples = pd.read_csv(bundle / "example_customers.csv")
    pipe.predict(parse_features(examples))
    for old, current in zip(before[:3], [imputer.statistics_, scaler.mean_, scaler.scale_]):
        np.testing.assert_array_equal(old, current)
    for old, current in zip(before[3], encoder.categories_):
        np.testing.assert_array_equal(old, current)
    passed("prediction_preserves_statistics", "Imputer, scaler and encoder state unchanged")

    # 4. Test each supported input policy separately on copies of a label-free example.
    missing = examples.head(1).copy()
    missing["TotalCharges"] = "   "
    parsed = parse_features(missing)
    require(parsed.TotalCharges.isna().all(), "Blank charge was not parsed as missing")
    transformed = numeric.transform(parsed[NUMERIC_FEATURES])
    expected = (medians[2] - scaler.mean_[2]) / scaler.scale_[2]
    np.testing.assert_allclose(transformed[0, 2], expected, rtol=0, atol=1e-12)
    require(np.isfinite(pipe.predict_proba(parsed)).all(), "Missing-input prediction failed")
    passed("missing_numeric", "Blank TotalCharges uses the training median")

    unknown = examples.head(1).copy()
    novel = "__part5_unseen_contract__"
    require(novel not in encoder.categories_[0], "Test category is not novel")
    unknown["Contract"] = novel
    cat = pipe.named_steps["preprocessor"].named_transformers_["cat"]
    encoded = cat.transform(parse_features(unknown)[CATEGORICAL_FEATURES])
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    require(np.all(encoded[0, :len(encoder.categories_[0])] == 0),
            "Unknown Contract was not encoded as zeros")
    require(np.isfinite(pipe.predict_proba(parse_features(unknown))).all(),
            "Unknown-category prediction failed")
    passed("unknown_category", "Unseen Contract encodes as zeros; reliability is not established")

    # 5. A required column cannot disappear silently at the inference boundary.
    try:
        parse_features(examples.drop(columns="Contract"))
    except KeyError as error:
        passed("missing_required_feature", f"Visible KeyError: {error}")
    else:
        raise AssertionError("Missing required feature was silently accepted")

    # 6. Run inference in an empty working directory, blocking fit and source CSV access.
    # The child can read only the label-free input CSV through pandas.
    child = '''
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from telco_churn.scripts.predict_pipeline import predict_batch
artifact, input_path, output_path = map(Path, sys.argv[2:])
read_csv = pd.read_csv
def guarded_read(path, *args, **kwargs):
    if Path(path).resolve() != input_path.resolve():
        raise AssertionError("Inference tried to read training data or labels")
    return read_csv(path, *args, **kwargs)
with patch.object(pd, "read_csv", side_effect=guarded_read):
    customers = pd.read_csv(input_path)
    assert "Churn" not in customers.columns
    with patch.object(Pipeline, "fit", side_effect=AssertionError("Refitting forbidden")), \\
         patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("Refitting forbidden")), \\
         patch.object(SimpleImputer, "fit", side_effect=AssertionError("Refitting forbidden")), \\
         patch.object(StandardScaler, "fit", side_effect=AssertionError("Refitting forbidden")), \\
         patch.object(OneHotEncoder, "fit", side_effect=AssertionError("Refitting forbidden")):
        predict_batch(artifact, customers).to_csv(output_path, index=False)
'''
    with TemporaryDirectory() as temporary:
        output = Path(temporary) / "predictions.csv"
        subprocess.run([
            sys.executable, "-c", child, str(PROJECT_ROOT), str(artifact),
            str(bundle / "example_customers.csv"), str(output),
        ], cwd=temporary, check=True, capture_output=True, text=True)
        loaded = pd.read_csv(output)
    original = pd.read_csv(bundle / "original_predictions.csv")
    np.testing.assert_array_equal(original.predicted, loaded.predicted)
    np.testing.assert_allclose(original.score, loaded.score, rtol=0, atol=1e-12)
    passed("fresh_process_reload", {
        "rows": len(loaded), "class_predictions_equal": True,
        "max_probability_difference": float(np.max(np.abs(original.score - loaded.score))),
        "atol": 1e-12, "rtol": 0, "fit_blocked": True, "training_csv_access_blocked": True,
    })

    report = {"checks": checks, "test_evaluated": False,
              "peer_review": "Pending: a classmate must reproduce and report an issue/resolution"}
    report_path = PROJECT_ROOT / "reports/reliability_checks.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"All {len(checks)} checks passed. Evidence: {report_path}")


if __name__ == "__main__":
    main()
