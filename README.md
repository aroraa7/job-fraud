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
| `notebooks/` | Exploration and explanations |
| `src/job_fraud/` | Reusable pipeline code |
| `configs/` | Experiment settings |
| `tests/` | Data integrity and workflow checks |
| `reports/` | Data audit and evaluation results |
| `models/` | Saved classifiers and inference configuration |

## Status

Only the folder scaffold and raw data are prepared. Select a Mamba/Conda environment before running Python. No dependencies have been installed and no API calls have been made.

The project is tracked on `main` in the private [GitHub repository](https://github.com/aroraa7/job-fraud). Raw data and generated artifacts remain local and are excluded from Git. Always ask the user before each push.
