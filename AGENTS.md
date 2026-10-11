# Project instructions

Read README.md for project scope and layout.

- Do not initialize Git, create a remote repository, or upload files until the user requests it. Always obtain explicit user approval before every push to GitHub.
- Never install dependencies or tools without explicit permission. Before running Python, select an appropriate Mamba/Conda environment with the user; do not use system Python or base.
- Keep raw data immutable. Preserve job_id and validate the binary fraudulent label.
- Put reusable code in src/job_fraud/, settings in configs/, focused checks in tests/, and generated findings in reports/.
- All reusable source code lives in src/job_fraud/. Notebooks are examples and explanations that import shared source code; do not implement duplicate pipeline logic in notebooks.
- Use the job-fraud Mamba/Conda environment specified in environment.yml. Invoke commands with mamba run -n job-fraud or activate that environment first.
- Use identical posting text and persisted split membership for the four experiments. Keep duplicate and related postings in the same split.
- Fit learned preprocessing and calculate class weights on training data only. Tune hyperparameters and thresholds on validation data; reserve test data for final evaluation.
- Save embeddings with ordered posting IDs, source/text identity, model/tokenizer identity, and generation settings. Verify alignment before training.
- Report fraud precision, recall, F1, average precision, confusion matrices, and class support, with configuration and split identity.
- Save classifiers with input construction, embedding settings, label mapping, and threshold. Verify save/load prediction consistency.
- Keep credentials out of source and outputs. Use environment variables. Obtain permission before model downloads or paid API execution.
- Before Git tracking, exclude credentials, datasets, embeddings, checkpoints, caches, and local environments. Stage files selectively.
- Do not copy historical experiments into this fresh project unless requested. Explain changes, checks, and remaining limitations.
