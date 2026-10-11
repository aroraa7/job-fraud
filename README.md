# Job Fraud Detection

Compare OpenAI and frozen BERT embeddings for fraudulent job-posting detection, using logistic regression and random forest on shared evaluation splits.

## Dataset

The original raw CSV is at `data/raw/fake_job_postings.csv`, copied unchanged from the existing local dataset. Source: [Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction).

The label is `fraudulent` (`0` = real, `1` = fraudulent). Preserve `job_id` throughout the workflow. No cleaning, filtering, splitting, or embedding generation has been performed in this project yet.

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

## Status

The folder scaffold, raw data, and initial Python environment are prepared. No data processing, embedding generation, model training, or paid API calls have been performed yet.

The project is tracked on `main` in the private [GitHub repository](https://github.com/aroraa7/job-fraud). Raw data and generated artifacts remain local and are excluded from Git. Always ask the user before each push.
