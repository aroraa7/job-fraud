# Job Fraud Detection

Compare OpenAI and frozen BERT embeddings for fraudulent job-posting detection, using logistic regression and random forest on shared evaluation splits.

The current stage generates embedding input directly from all 16 raw posting-information fields. Every original row is retained. `job_id` and the `fraudulent` target remain separate from the embedding passage. Cleaning, deduplication, splitting, and added vector normalization are deferred.

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

This completed audit describes an earlier five-text-field proposal. Its findings are preserved for reference; they do not filter rows or determine the current 16-field embedding inputs.

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

The project scaffold, initial Python environment, raw-data audit, and raw-field embedding input construction are complete. All 28 focused checks passed. A full input snapshot was constructed and verified; no actual embeddings, splits, trained models, or paid API calls have been produced. BERT and OpenAI execution code is prepared but has not been verified against actual models. Embedding dependencies and model resources are not installed yet.

## Embedding inputs and generation

`src/job_fraud/embedding_inputs.py` renders each posting as 16 labeled sections in raw column order. Text values, HTML, punctuation, and whitespace are preserved. The three binary fields are rendered as:

```text
Remote work offered: Yes / No
Company logo present: Yes / No
Screening questions present: Yes / No
```

Missing binary values become `Unknown`; unexpected values are rejected. Other missing values remain empty sections. The binary values in the raw CSV and snapshot's original columns stay `0`/`1`; only the generated passage uses Yes/No. The snapshot retains all original columns and adds `embedding_text` and `text_sha256`.

Create a new local snapshot without model packages, downloads, or API calls:

```sh
PYTHONPATH=src mamba run -n job-fraud python -m job_fraud.embeddings --backend prepare
```

Each run writes `inputs.csv`, `input_manifest.json`, and `preview.txt` into a new timestamped directory under `data/embeddings/raw_all_fields_yes_no_v1/`. The manifest records field labels, raw/input checksums, row count, and input construction version. Snapshot loading verifies those checksums, posting IDs, labels, and the relationship between source fields and rendered passages.

`configs/embeddings.json` contains proposed generation settings: `google-bert/bert-base-uncased` with content-token mean pooling, and `text-embedding-3-small` with its default output dimensions. Long passages are handled as nonoverlapping token chunks; chunk vectors are combined with a content-token-weighted mean so text is not silently truncated. Both providers receive the same raw passage, although tokenizers and chunk boundaries differ. No additional vector normalization is applied; OpenAI's provider-returned normalization is retained, and averaging chunks can change vector length.

The following are future commands, not completed runs. Packages `torch`, `transformers`, `openai`, and `tiktoken` and any model/tokenizer resources require explicit installation/download permission. Configure `OPENAI_API_KEY` through the environment, never tracked files. Run a small BERT check before a full run:

```sh
PYTHONPATH=src mamba run -n job-fraud python -m job_fraud.embeddings --backend bert --input-dir PATH_TO_SNAPSHOT --limit 5
```

BERT uses local cached resources by default. Only add `--allow-model-download` after download approval. It runs frozen in evaluation/inference mode and selects MPS when available, otherwise CPU. Omit `--limit` for all postings.

Estimate OpenAI input tokens without paid API calls (the tokenizer resource may need a separately approved download):

```sh
PYTHONPATH=src mamba run -n job-fraud python -m job_fraud.embeddings --backend openai --input-dir PATH_TO_SNAPSHOT
```

After approval of cost and token budget, add `--execute-openai --max-api-tokens APPROVED_TOKEN_CAP`. Without the execution flag this backend only plans token usage. Requests respect per-input and per-request limits; automatic paid retries are disabled. A failed run does not automatically resume, and API calls already made can still incur charges. Review failures before rerunning.

Successful generation saves `embeddings.npy`, `posting_ids_and_labels.csv`, and `manifest.json` in a new provider-specific directory under the snapshot. The manifest records configuration, input identity, dependency versions, chunk counts, available provider/model revision information, output checksums, and whether the run used all rows or an explicit sample. Embedding row order matches the saved posting IDs and labels. Output files remain excluded from Git. These save/load checks have passed with fixtures, not real model vectors.

The project is tracked on `main` in the private [GitHub repository](https://github.com/aroraa7/job-fraud). Raw data and generated artifacts remain local and are excluded from Git. Always ask the user before each push.
