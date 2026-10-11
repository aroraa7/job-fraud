# Job Fraud Detection

Compare OpenAI and frozen BERT embeddings for fraudulent job-posting detection, using logistic regression and random forest on shared evaluation splits.

## Dataset

The original raw CSV is at `data/raw/fake_job_postings.csv`, copied unchanged from the existing local dataset. Source: [Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).

The label is `fraudulent` (`0` = real, `1` = fraudulent). Preserve `job_id` throughout the workflow. The raw-data audit is complete; no cleaning, filtering, splitting, or embedding generation has been performed yet.

## Folders

| Folder | Purpose |
| --- | --- |
| `data/raw/` | Immutable source data |
| `data/processed/` | Prepared data and persisted split assignments |
| `data/embeddings/` | Embeddings, posting IDs, and generation metadata |
| `notebooks/` | Examples and explanations importing shared source code |
| `src/job_fraud/` | All reusable source code and pipeline implementations |
| `configs/` | Experiment settings |
| `tests/` | Data integrity and workflow checks |
| `reports/` | Data audit and evaluation results |
| `models/` | Saved classifiers and inference configuration |

## Environment

Use the `job-fraud` Mamba/Conda environment defined in [environment.yml](environment.yml). It contains Python 3.11, pandas, scikit-learn, and pytest for the initial data preparation stage.

Verified on October 11, 2026: Python 3.11.17, pandas 3.0.6, scikit-learn 1.9.1, and pytest 9.1.1. The environment specification allows compatible updates; these versions record the initial installation.

On a new machine, create the environment after authorizing installation:

```sh
mamba env create -f environment.yml
```

The environment has already been created on this machine. Activate it with:

```sh
mamba activate job-fraud
```

Alternatively, target it explicitly without activation:

```sh
mamba run -n job-fraud python --version
```

All reusable source code belongs in `src/job_fraud/`. Notebooks contain examples and explanations that import this code. Notebook execution tools and OpenAI/BERT dependencies will be added separately with permission when needed.

## Raw-data audit

From the project root, run the shared source code:

```sh
PYTHONPATH=src mamba run -n job-fraud python -m job_fraud.audit
mamba run -n job-fraud python -m pytest -q
```

The audit resolves the project root from its source location; `--project-root PATH` can override it explicitly. It writes a timestamped directory under `reports/data_audit/` containing `audit.md`, `audit.json`, and `duplicate_memberships.csv`. Reports remain local under the current Git exclusions. Each report records the raw CSV checksum, audit version, run time, and dependency versions.

`src/job_fraud/data.py` validates required columns, positive unique posting IDs, and binary nonmissing labels. `src/job_fraud/audit.py` measures class balance, empty fields, text lengths, duplicate candidates, and conflicting duplicate labels without altering the raw file. Missing descriptive fields remain empty strings, and literal text such as `NA` is retained.

The initial audit found 17,880 postings (866 fraudulent), valid posting IDs and labels, and one empty description. Normalized matching across the five proposed text fields identified 747 duplicate groups involving 2,755 postings, with no conflicting labels within those groups. Matching ignores case and repeated whitespace; it does not detect all near duplicates. Review related postings before finalizing shared splits.

Combined text has a median of 349 and a 95th percentile of 758 whitespace-separated words. These are not tokenizer counts and do not establish model token limits. HTML handling, final input construction, long-text coverage, and splitting remain pending.

See `notebooks/01_raw_data_audit_example.ipynb` for an example importing the shared implementation. Its seven code cells were checked in a fresh Python process, but it has not been run in a Jupyter kernel; notebook tooling and kernel registration are still pending. The notebook keeps outputs empty. The source package is imported through `PYTHONPATH=src` or the example's explicit project-root setup; no editable package installation is required at this stage. `pyproject.toml` currently configures pytest only.

## Status

The project scaffold, Python environment, and read-only raw-data audit are complete. All 16 focused checks passed. No cleaned dataset, splits, embeddings, trained models, or paid API calls have been produced.

The project is tracked on `main` in the private [GitHub repository](https://github.com/aroraa7/job-fraud). Raw data and generated artifacts remain local and are excluded from Git. Always ask the user before each push.
