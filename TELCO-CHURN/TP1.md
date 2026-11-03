# TP01 - Build a Reproducible Churn Project

**ESILV - Machine Learning Operations**  
**Dataset:** IBM Telco Customer Churn  
**Duration:** approximately 90 minutes  
**Starting point:** your own new project and the original CSV  
**Format:** work in groups or individually ; keep individual explanations

> Can another engineer install your project, identify its input data and run it without asking you what to do?

## Your mission

You have joined a telecom ML team. The company provides a customer snapshot and wants to predict churn. Before training a model, establish a project that a teammate can understand, run and reproduce.

You receive the dataset and this brief. Create your own repository, environment, source files and documentation. Use a project template or the example layout below as a structural guide; neither supplies the completed exercise. TP02 will build training and preprocessing Pipelines inside this same project.

Use Python 3.11 or 3.12 and Git. uv is the recommended environment/dependency manager; agree another approach with the instructor if necessary. Tools should be installed and GitHub login working before class.

The short hints below identify useful APIs and commands. They are not a complete solution. Replace `...`, add imports where needed and decide how the pieces belong in your project.

## 1. Design a small project and its environment

Start from a recognizable ML/data project structure rather than putting data, notebooks, models and source code into one folder. There is no single universally correct tree: the important properties are clear responsibilities, immutable original inputs, reproducible dependencies and discoverable entry points.

### Option A - Generate a structure with Cookiecutter Data Science

**Cookiecutter** is a project templating tool; **Cookiecutter Data Science (CCDS)** is a template and generator tailored to data projects. A generated structure does not train a model or complete this TP for you.

For the current CCDS v2 route, use its `ccds` command. With uv already available, run the generator as an isolated tool:

```sh
# Run from the parent directory in which you want the project created.
uvx --from cookiecutter-data-science ccds
```

If the instructor has installed CCDS separately, `ccds` is sufficient. The generator creates a new project directory; do not run it inside an existing project that you want to keep. Older material using the v1 template can have different commands and a different source layout.

Suggested responses for the course (select labels, not remembered menu numbers):

| Prompt | Example response |
|---|---|
| Project name | Telco Churn |
| Repository name | telco-churn |
| Module name | telco_churn |
| Python version | 3.11 or 3.12, matching your environment |
| Dataset storage | none; use the locally supplied CSV |
| Environment manager | uv |
| Dependency file | pyproject.toml |
| Pydata packages | none; add what you need explicitly |
| Include code scaffold | No; begin with an empty Python module |
| Docs | none for this TP |

For other options, keep the project lightweight and record your choices. Generated development tooling need not all be used today. Read the generated README and inspect `pyproject.toml` before installing dependencies. Do not run `uv init` again inside a generated project.

### Option B - Build a small structure yourself

If the template is unavailable or you choose the manual route, use the example below. A useful initialization hint is:

```sh
# Creates a new project folder and an installable src-layout package.
uv init --package telco-churn
```

Then work inside that new directory and add the needed folders. This is an alternative to Option A, not an extra command to run on top of it.

### A concrete Telco example

The paths below are relative to a project root called `telco-churn/`. This is a **course-adapted example**, not a promise that CCDS generates exactly these files.

| Path | What belongs here | When needed |
|---|---|---|
| `README.md` | Setup, source data, commands and handover instructions | TP01 |
| `pyproject.toml`, `uv.lock` | Package/dependency definition and resolved selections | TP01; lock created during resolution |
| `.gitignore` | Local environments, credentials and output policy | TP01 |
| `data/raw/telco.csv` | Original, unchanged source snapshot | TP01 |
| `data/interim/` | Intermediate transformed datasets | Later, only if useful |
| `data/processed/` | Prepared datasets for downstream use | Later, if you persist them |
| `src/telco_churn/__init__.py` | Makes the project code an importable package | TP01, manual src-layout route |
| `src/telco_churn/dataset.py` | Your inspection/data-reading code | TP01 |
| `scripts/inspect_data.py` | Optional thin command-line entry calling package functions | TP01, if you use a separate entry script |
| `notebooks/01_initial_exploration.ipynb` | Optional exploratory work, not the only runnable workflow | As needed |
| `references/data_source.md` | Data source, hash, field notes | TP01 |
| `reports/data_overview.json` | Generated inspection report | TP01 |
| `evidence/tp01/` | Small submitted records, review and recreation evidence | TP01 |
| `configs/train.json` | Training configuration | TP02 |
| `src/telco_churn/pipeline.py` | Unfitted preprocessing/classifier builders | TP02 |
| `src/telco_churn/training.py` | Training and candidate comparison logic | TP02 |
| `src/telco_churn/prediction.py` | Load a fitted pipeline and predict | TP02 |
| `models/` | Saved fitted pipelines and associated metadata | TP02 |
| `tests/` | Focused behavior checks | As the project develops |

CCDS v2 commonly puts the importable package directly at `telco_churn/` rather than `src/telco_churn/`. Keeping that generated layout is acceptable. Both layouts work when packaging is configured correctly. Do not move the package into `src/` without checking the build configuration. Likewise, CCDS may use `dataset.py`, `features.py` and `modeling/`; these names can fulfill the same responsibilities as your own modules.

Today, establish the TP01 locations and explain the later ones in your README. You do not need to implement TP02 code or create meaningless empty files. Generated models belong in `models/`; Python model-building code belongs in the importable package. A package directory called `modeling/` contains code, not serialized training outputs.

### Decide where each new file belongs

Discuss these examples before writing code:

| New item | Appropriate location in the example |
|---|---|
| Unmodified CSV received from the company | `data/raw/` |
| A cleaned copy or intermediate transformation | `data/interim/` or `data/processed/`, depending on its purpose |
| Function that fills missing values as part of a training Pipeline | Package source, later `pipeline.py` |
| A fitted pipeline saved with joblib | `models/` |
| A plot showing the label distribution | `reports/figures/` |
| An exploratory notebook | `notebooks/` |
| Hyperparameters | `configs/` |
| A check that predictions survive save/reload | `tests/` |

A local folder hierarchy does not itself implement Bronze/Silver/Gold storage, data versioning or access control. Use the folder names to communicate purpose; explain the actual behavior of your program.

### Create the environment and verify imports

After generating or initializing the project, work from its root:

```sh
uv add pandas
uv run python --version
uv tree
```

Record the interpreter version and dependency choices. Start with TP01 dependencies; training packages can be added in TP02. With an installable package configured and installed by uv, check that your own module imports:

```sh
uv run python -c "import telco_churn; print(telco_churn.__file__)"
```

Adapt the module name if you chose another one. Do not copy `.venv/` between projects or solve imports by hardcoding your laptop's directories into `sys.path`.

**Questions:** How do `pyproject.toml`, `uv.lock` and `.venv/` differ? Why separate notebooks from reusable source code? Where should the fitted model go, and where should the code that builds it go?

**Checkpoint:** show your layout, explain five file-location choices, import pandas and your package through the project environment, and document the setup. If using CCDS, record its generator version/options in your README; generated output depends on the chosen version and options.

## 2. Identify the data you received

Keep the original CSV unchanged. Record its source, local location and SHA-256 hash. The source repository and commit-pinned download are listed at the end of this brief; you may use the copy supplied by the instructor.

Understand the distinction between:

| Role | Examples |
|---|---|
| Customer identifier | `customerID` |
| Potential numeric inputs | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Potential categorical inputs | `Contract`, `InternetService`, `PaymentMethod` |
| Target | `Churn`, with labels `Yes` and `No` |

There are additional fields in the source. This is a customer snapshot, not a dated event history or a randomized retention experiment.

### Hint

```python
from pathlib import Path
import hashlib

source = Path(...)  # Your input path.
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
```

**Questions:** Can two files with the same name contain different data? Does a matching hash prove that the business data are correct? Why should the target and customer ID not become ordinary model inputs later?

**Checkpoint:** a teammate can identify and obtain the same source file. Do not change the file in a spreadsheet application and assume its original byte identity is preserved.

## 3. Implement a small inspection program

Write a script that runs from a terminal and saves a machine-readable report. Choose your filename, report format and project locations.

Your report must include:

- Dataset dimensions and field types.
- Target counts and proportions.
- Duplicate customer-ID count.
- Missing-value counts, with an explicit check for empty or whitespace-only values.
- Source identity/hash and the environment information needed to interpret the run.

Keep the work focused on inspection. Do not clean away records, impute values, split data or train models in this TP.

### Hint

```python
import pandas as pd

df = pd.read_csv(source)
print(df.shape)
print(df.dtypes)
print(df["Churn"].value_counts(dropna=False))
# TODO: compute the remaining checks and save a report.
```

For whitespace inspection, research `astype("string")`, `str.strip()` and `eq("")`. Compare that result with `isna()`; they do not necessarily identify the same records.

For environment evidence, `sys.version` and `importlib.metadata.version("pandas")` are useful starting points. You decide how to record and associate them with the report.

**Questions:** Why might a charge column be read as text? Why should the script save a report rather than only print output? Would always predicting the majority class be a meaningful performance baseline in TP02?

**Checkpoint:** one documented command generates the report. You can explain its fields. Discover the results yourself; no expected count table is supplied.

## 4. Prove that the environment can be recreated

Preserve the first environment record. Close processes using the project environment, set the existing environment aside and recreate it from your declared dependencies. Avoid discarding unrelated work.

Run the same inspection again with the same code and original input. Compare interpreter/package versions, source hash and data statistics. Explain any differences; timestamps and machine details need not be identical.

### Hint

Once a uv project already has a consistent lock file, investigate:

```sh
uv sync --locked
```

This is a recreation/consistency check, not the first command for a brand-new project without a lock. Do not bypass a mismatch by silently resolving a new dependency set. A lock records package selections; it does not freeze every hardware, operating-system or future training detail.

**Checkpoint:** the inspection still runs in the recreated environment, and you retain evidence from before and after.

## 5. Version and review a useful change

Create your own GitHub repository and use Git to record meaningful changes. Inspect `.gitignore` before staging files. Exclude environments and their backups, credentials, and generated outputs according to your documented project policy.

For this exercise, keep the raw CSV outside Git and record how to obtain it and verify its hash. Keep small, non-sensitive submission evidence in a deliberately tracked location. Do not ignore every report and then accidentally omit the report needed for assessment.

Work on a branch to improve your README: a new teammate should understand the business question, setup, source data, inspection command and generated result. Open a pull request and ask your partner to review it. Arrange the repository access needed for review.

### Hint

```sh
git status
git switch -c docs/project-onboarding
git diff
# TODO: stage a focused change, commit, push and open a pull request.
```

The branch example assumes the repository has an initial commit. If you are unsure whether a repository already exists, inspect it before running `git init` or adding a remote. Never push to a repository you do not own or have permission to modify.

**Questions:** What does a commit identify? What is still unrecorded if your working tree has edits? What information belongs in a helpful PR description?

**Checkpoint:** a focused PR includes a verification statement, one substantive review question and your response. A merge alone is not evidence that the instructions worked.

## 6. Hand over the project

Ask another pair to obtain your project in a clean directory, follow your README, install the environment and run the inspection. Let them try without your oral instructions first.

Record one issue they found, how you addressed it and whether the revised instructions worked. Keep the working directory for TP02.

**Final question:** what can another engineer now reproduce, and what evidence would still be needed to reproduce a trained model?

## Deliverables and assessment

| Deliverable | Evidence | Points |
|---|---|---:|
| Own project structure and environment definition | Clear data/code/output boundaries; importable code; dependencies/interpreter documented; recreation comparison | 3 |
| Inspection code and report | Requested checks, source provenance and hash; original data unchanged | 3 |
| Git workflow and reviewed change | Meaningful commits, appropriate ignore rules and PR discussion | 2 |
| README and handover note | Peer execution, one issue/resolution and explanation of remaining limits | 2 |
| **Total** | | **10** |

Submit a repository link or project export with accessible evidence. File names and directory choices are yours. Do not submit virtual environments, credentials or someone else's reference implementation.

### Suggested checkpoints

| Around | Aim to have |
|---|---|
| 20 minutes | Your project and environment established |
| 45 minutes | Inspection report and data-source record |
| 60 minutes | Recreation evidence |
| 75 minutes | README change and peer review |
| Final 15 minutes | Handover, one fix and submission check |

If login or network problems block the remote PR, keep local commits and a written peer review as interim evidence; record the blocker and complete the PR after access is restored.

## Optional extensions

1. **Git recovery:** on a disposable branch, make a temporary README edit. Research `git restore` versus `git restore --staged`. Predict which operation discards the edit and which only unstages it; inspect the diff before applying either command.
2. **Different source bytes:** change a temporary copy of the CSV and compare its hash. Do not alter the original or describe the modified file as the original snapshot.
3. **Cross-machine recreation:** compare your partner's environment and results. Distinguish package selections, data content and machine-specific details.

## Documentation starting points

- [IBM Telco source repository](https://github.com/IBM/telco-customer-churn-on-icp4d)
- [Commit-pinned CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/1fd6fd70906479be1712e64cfc0ea89f58466629/data/Telco-Customer-Churn.csv)
- [uv project guide](https://docs.astral.sh/uv/guides/projects/)
- [Cookiecutter Data Science: quickstart and structure](https://cookiecutter-data-science.drivendata.org/)
- [Cookiecutter Data Science: generator options](https://cookiecutter-data-science.drivendata.org/all-options/)
- [uv: running isolated tools](https://docs.astral.sh/uv/concepts/tools/)
- [Python package version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/)
- [Pro Git](https://git-scm.com/book/en/v2)
- [Git restore](https://git-scm.com/docs/git-restore)
- [GitHub pull requests](https://docs.github.com/en/pull-requests)

Consult the docs and ask focused questions when blocked. You should be able to explain the purpose of each command and each check in your implementation.