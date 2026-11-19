"""Verify TP03 records and registered alias after a manual MLflow server restart."""

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd

from telco_churn.scripts.train_pipeline import EXPERIMENT_NAME, PROJECT_ROOT, TRACKING_URI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--after-restart", action="store_true",
        help="Confirm that the server was manually restarted before this check",
    )
    args = parser.parse_args()
    if not args.after_restart:
        parser.error("Stop/restart the tracking server first, then pass --after-restart.")

    report_dir = PROJECT_ROOT / "reports"
    handover = json.loads((report_dir / "tp03_model_handover.json").read_text(encoding="utf-8"))
    original = pd.read_csv(report_dir / "tp03_original_predictions.csv")
    input_path = report_dir / "tp03_inference_examples.csv"
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise AssertionError(f"Experiment {EXPERIMENT_NAME!r} is unavailable after restart.")
    run = client.get_run(handover["source_run_id"])
    if run.info.status != "FINISHED":
        raise AssertionError(f"Source run status changed to {run.info.status!r}.")
    registered = handover["registered_model"]
    alias = handover["alias"]
    version = client.get_model_version_by_alias(registered["name"], alias["name"])
    if version.version != registered["version"]:
        raise AssertionError("Alias no longer points to the recorded registered version.")
    if version.source != handover["model_uri"] or version.run_id != handover["source_run_id"]:
        raise AssertionError("Registered version no longer matches the recorded model/run.")

    # Use an empty working directory and the inference-only module in a child process.
    with TemporaryDirectory() as temporary:
        output = Path(temporary) / "alias_predictions.csv"
        subprocess.run([
            sys.executable, "-m", "telco_churn.scripts.predict_registered_model",
            "--model-uri", alias["uri"], "--input", str(input_path), "--output", str(output),
        ], cwd=temporary, check=True)
        reloaded = pd.read_csv(output)
    np.testing.assert_array_equal(original["predicted"], reloaded["predicted"])
    np.testing.assert_allclose(original["score"], reloaded["score"], rtol=0, atol=1e-12)
    persistence = {
        "observed_utc": datetime.now(UTC).isoformat(),
        "manual_restart_confirmed": True,
        "tracking_uri": TRACKING_URI,
        "experiment": {"name": EXPERIMENT_NAME, "id": experiment.experiment_id},
        "source_run": {"id": run.info.run_id, "status": run.info.status},
        "registered_model": {"name": version.name, "version": version.version,
                             "alias": alias["name"], "uri": alias["uri"]},
        "model_source_uri": version.source,
        "fresh_process_inference": {"rows": len(reloaded), "classes_equal": True,
                                    "probability_atol": 1e-12, "probability_rtol": 0,
                                    "max_probability_difference": float(
                                        np.max(np.abs(original["score"] - reloaded["score"]))
                                    )},
        "test_set_evaluated": False,
    }
    (report_dir / "tp03_persistence_check.json").write_text(
        json.dumps(persistence, indent=2) + "\n", encoding="utf-8"
    )
    summary = f"""# TP03 summary

## Evidence to submit

- Comparison screenshot: add your MLflow screenshot filename here.
- Selected-run screenshot: add your MLflow screenshot filename here.
- Registry screenshot: add your MLflow screenshot filename here.

## Selected model

- Source run ID: `{run.info.run_id}`
- Logged model URI: `{handover['model_uri']}`
- Registered model/version: `{version.name}` version `{version.version}`
- Alias: `{alias['uri']}`
- Selection: random forest had the highest validation average precision under the
  predefined TP02/TP03 metric rule. The test set was not evaluated.

## Fresh-process inference

```powershell
uv run python -m telco_churn.scripts.predict_registered_model --input reports/tp03_inference_examples.csv --output reports/tp03_alias_predictions.csv
```

After the manual server restart, the experiment, source run, registered version
and `candidate` alias remained available. The inference-only child process loaded
the alias, reproduced all {len(reloaded)} class predictions, and had maximum
probability difference {persistence['fresh_process_inference']['max_probability_difference']:.1f}
within absolute tolerance `1e-12`.

## Local storage and handover

`mlflow.db` stores experiment/run metadata, parameters, metrics, tags, dataset
records and registry/alias metadata. `mlartifacts/` stores logged files and model
artifacts. If the database survived but artifacts were lost, metadata would remain
but the recorded pipeline could not be loaded. If only artifacts survived, the
run history, registry name/version and alias mapping would be missing. A colleague
on another machine needs the compatible environment and project package, plus
access to a shared tracking/artifact server or exported tracking metadata and
artifacts; a local URI alone is not enough.
"""
    (report_dir / "tp03_summary.md").write_text(summary, encoding="utf-8")
    print("Persistence verified after restart: experiment, run, version, alias and inference work.")
    print(f"Evidence: {report_dir / 'tp03_persistence_check.json'}")
    print(f"Submission summary: {report_dir / 'tp03_summary.md'}")


if __name__ == "__main__":
    main()
