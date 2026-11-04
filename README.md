# Telco Churn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         telco_churn and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── telco_churn   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes telco_churn a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## Setup and data inspection

This project inspects a telecom customer snapshot before building a model
to predict customer churn.

### Install the environment

Install Python 3.12 and uv, then run from the project directory:

```powershell
uv sync --locked
```

### Obtain the dataset

Download the [original CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv)
and save it as `data/raw/telco.csv`. Keep the original file unchanged.

Expected SHA-256:

```text
16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
```

Verify the local file:

```powershell
uv run python -m telco_churn.scripts.hash_data
```

### Run the inspection

```powershell
uv run python -m telco_churn.scripts.inspect_data
```

The script saves `reports/data_overview.json`, containing dataset dimensions,
column types, target counts and proportions, duplicate customer-ID count,
missing and blank-value counts, source hash, and Python/pandas versions.

The raw CSV and local environment are excluded from Git.

## TP02 milestone 1: first model

Install the locked environment as above, then run:

```powershell
uv run python -m telco_churn.scripts.first_model
```

The script uses only `tenure` and `MonthlyCharges` to train logistic regression
and a `DummyClassifier(strategy="most_frequent")`. Both fit on training data.
`customerID` identifies customers and checks split separation; `Churn` is the
target, with **Yes = 1 (positive)** and No = 0. Neither is an input feature.

Two stratified splits with seed 42 allocate 4,225 training, 1,409 validation
and 1,409 test customers (approximately 60/20/20). The script saves customer
allocations in `reports/customer_splits.csv` for reuse in later milestones.
Keep the source CSV, seed and allocations fixed. The test set is reserved;
only validation results may guide model selection.

Validation results on the original CSV:

| Model | Accuracy | Actual No: predicted No / Yes | Actual Yes: predicted No / Yes |
|---|---:|---|---|
| Majority baseline | 0.7346 | 1035 / 0 | 374 / 0 |
| Logistic regression | 0.7828 | 950 / 85 | 221 / 153 |

Confusion matrices use actual classes as rows and predicted classes as columns,
both ordered `[No (0), Yes (1)]`. Always predicting No reaches 73.46% accuracy
while missing every churner. The learned model catches 153 of 374 churners,
so accuracy alone is insufficient to describe its usefulness.

`reports/first_model_metrics.json` records accuracy, labeled confusion matrices,
seed and split sizes. `reports/first_model_validation_predictions.csv`
contains customer IDs, actual labels and both models' validation predictions.
The original CSV remains unchanged. This milestone requires complete numeric
values in its two inputs; preprocessing and the larger feature set come next.

## TP02 milestone 2: preprocessing pipeline

Milestone 1 must have produced `reports/customer_splits.csv`. Then run:

```powershell
uv run python -m telco_churn.scripts.train_pipeline
```

One `fit` call trains the preprocessing and a logistic regression on training
customers only. Validation reuses the fitted transformations. The test set stays
reserved. Pass inputs through `parse_features`, then to the pipeline:

| Column | Input type | Preprocessing |
|---|---|---|
| `tenure`, `MonthlyCharges`, `TotalCharges` | number, or numeric text | Median imputation, then `StandardScaler` |
| `Contract`, `InternetService`, `PaymentMethod` | text | Missing becomes `"missing"`, then `OneHotEncoder(handle_unknown="ignore")` |

`parse_features` learns nothing from the data, and inference must apply the same
rules:

- **Blank `TotalCharges`:** whitespace-only values become missing.
- **Malformed numbers:** any other non-numeric text raises a `ValueError`.
- **Missing required column:** raises an error instead of predicting.
- **Missing or unseen category:** encodes as all zeros for that feature. The model
  still returns a prediction, which is not evidence that it is reliable for that
  customer.

The 11 blank `TotalCharges` values all belong to customers with `tenure` 0, 7 of
them in training. Median imputation gives them the training median of 1373.625,
far above what a new customer has been charged. This is a known simplification.

Validation results on the original CSV (Churn = Yes is positive; rows are actual
and columns are predicted, ordered `[No, Yes]`):

| Model | Accuracy | Actual No: predicted No / Yes | Actual Yes: predicted No / Yes |
|---|---:|---|---|
| Six-feature logistic regression pipeline | 0.8041 | 935 / 100 | 176 / 198 |

These results differ from milestone 1 because the features and preprocessing
changed, so they do not show that one algorithm is better than another.
`reports/pipeline_metrics.json` records the configuration and metrics, and
`reports/pipeline_validation_predictions.csv` holds the validation predictions.

## TP02 milestone 3: compare models fairly

Run the comparison using the same six features and saved customer allocations:

```powershell
uv run python -m telco_churn.scripts.train_pipeline --compare
```

Without `--compare`, the script still runs the part 2 logistic regression pipeline.
Each comparison candidate gets fresh preprocessing fitted only on training data.
The baseline always predicts the most frequent training class. Logistic regression
provides a simple linear model, a depth-5 decision tree captures nonlinear rules,
and a 100-tree random forest averages trees with depth limited to 8. Seeds are 42;
full parameters and approximate pipeline fit times are recorded in the reports.

The selection rule is fixed before evaluation: highest validation **average precision**
among learned models. This measures churn ranking in an imbalanced dataset using
class-1 probabilities rather than class predictions. Exact ties favor logistic
regression, then the tree, then the forest. Precision, recall and F1 use the same
probability threshold of 0.5, with undefined precision set to zero. The test set
remains reserved. Fit timing is descriptive on this machine.

| Model | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.2654 | 0.0000 | 0.0000 | 0.0000 | 0.7346 |
| Logistic regression | 0.6475 | 0.6644 | 0.5294 | 0.5893 | 0.8041 |
| Decision tree | 0.6173 | 0.6356 | 0.6016 | 0.6181 | 0.8027 |
| Random forest | 0.6643 | 0.6569 | 0.4813 | 0.5556 | 0.7956 |

Random forest is selected by the stated average-precision rule. Its higher ranking
quality does not imply higher recall at threshold 0.5: the decision tree detects
more churners at that threshold. This validation result does not establish that
the small AP difference will hold on unseen data. Comparison with part 1 cannot
isolate an algorithm improvement because the features also changed.

Generated files under `reports/` (ignored by Git):

- `model_comparison.csv`: metrics, confusion matrices, parameters and fit times.
- `model_comparison.json`: results, metric definitions and selection policy.
- `model_comparison_predictions.csv`: validation scores and predictions per model.
- `model_selection.txt`: short validation-only selection decision.

This milestone compares candidates; saving the fitted pipeline belongs to part 4.

## TP02 milestone 4: save and reload the fitted pipeline

Run the comparison, save its fitted winner, and verify inference in a fresh process:

```powershell
uv run python -m telco_churn.scripts.train_pipeline --save
```

This reuses the part 1 customer allocations and fits candidates only on training
data. Validation selects the winner; it is saved directly, without refitting on
validation or evaluating test. The artifact contains fitted imputers, scaler,
encoder and classifier together. Building a pipeline returns an unfitted object;
fitting it changes its state.

The generated bundle is `models/selected_pipeline/` (ignored by Git):

- `pipeline.joblib`: complete fitted prediction pipeline.
- `metadata.json`: source and artifact SHA-256, customer allocations, seed,
  feature roles and parsing policy, preprocessing settings, model parameters,
  validation metrics, selection rule, positive class and threshold, Python/library
  versions and hashes of source code, project configuration and lockfile.
- `customer_splits.csv`: copy of the customer allocations.
- `example_customers.csv`: five validation examples with six features and no labels.
- `original_predictions.csv` and `reloaded_predictions.csv`: reload evidence.

Predict in a separate process using only the artifact and an input feature CSV:

```powershell
uv run python -m telco_churn.scripts.predict_pipeline --artifact models/selected_pipeline/pipeline.joblib --input models/selected_pipeline/example_customers.csv --output reports/new_predictions.csv
```

The inference command does not train, access the original training CSV or require
labels. It applies the shared `parse_features` rules, then calls the loaded
pipeline. Provide all six original columns in the part 2 schema: numeric columns
accept numbers or numeric text and supported missing values; categorical columns
accept text or missing values. Malformed numeric text and absent required columns
fail visibly. Output contains class-1 probabilities (`score`) and predictions
using threshold 0.5 (`predicted`).

The save command checks that original and reloaded class predictions are identical
and probability differences are at most `1e-12` (absolute tolerance; relative
tolerance zero). The check uses the same examples after CSV round-tripping and
records its outcome in metadata. Only load trusted joblib files, which can execute
code when loaded. Recreate the recorded environment with `uv sync --locked`;
loading across library versions is not guaranteed to work. Keep the bundle intact
when sharing it through the course's artifact submission location.

