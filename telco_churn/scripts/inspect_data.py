from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
source = PROJECT_ROOT / "data" / "raw" / "telco.csv"

df = pd.read_csv(source)

# Inspect the target.
target_counts = df["Churn"].value_counts(dropna=False)
target_proportions = df["Churn"].value_counts(
    normalize=True, dropna=False
)

# Count repeated IDs after their first occurrence.
duplicate_ids = int(df["customerID"].duplicated().sum())

# Pandas missing values and blank strings are separate checks.
missing_counts = {
    column: int(df[column].isna().sum())
    for column in df.columns
}

blank_counts = {
    column: int(
        df[column].astype("string").str.strip().eq("").sum()
    )
    for column in df.columns
}

# Identify the original file without modifying it.
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

report = {
    "source": {
        "path": source.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": source_hash,
    },
    "environment": {
        "python": sys.version,
        "pandas": version("pandas"),
    },
    "dimensions": {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
    },
    "column_types": df.dtypes.astype(str).to_dict(),
    "target_counts": {
        str(label): int(count)
        for label, count in target_counts.items()
    },
    "target_proportions": {
        str(label): float(proportion)
        for label, proportion in target_proportions.items()
    },
    "duplicate_customer_ids": duplicate_ids,
    "missing_values": missing_counts,
    "blank_values": blank_counts,
}

# Save a machine-readable report.
report_path = PROJECT_ROOT / "reports" / "data_overview.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("Dimensions (rows, columns):")
print(df.shape)

print("\nColumn types:")
print(df.dtypes)

print("\nTarget counts:")
print(target_counts)

print("\nTarget proportions:")
print(target_proportions)

print("\nDuplicate customer IDs:")
print(duplicate_ids)

print("\nMissing values:")
print(pd.Series(missing_counts))

print("\nEmpty or whitespace-only values:")
print(pd.Series(blank_counts))

print("\nSHA-256:")
print(source_hash)

print(f"\nReport saved to: {report_path}")