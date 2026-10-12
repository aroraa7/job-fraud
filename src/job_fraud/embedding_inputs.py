"""Construct embedding passages from raw fields without cleaning or filtering."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from job_fraud.audit import file_sha256
from job_fraud.data import EXPECTED_COLUMNS, load_postings

INPUT_VERSION = "raw_all_fields_yes_no_v1"
INPUT_FIELDS = tuple(field for field in EXPECTED_COLUMNS if field not in ("job_id", "fraudulent"))
FIELD_LABELS = {
    "title": "Title", "location": "Location", "department": "Department",
    "salary_range": "Salary range", "company_profile": "Company profile",
    "description": "Description", "requirements": "Requirements", "benefits": "Benefits",
    "telecommuting": "Remote work offered", "has_company_logo": "Company logo present",
    "has_questions": "Screening questions present", "employment_type": "Employment type",
    "required_experience": "Required experience", "required_education": "Required education",
    "industry": "Industry", "function": "Function",
}
BINARY_FIELDS = ("telecommuting", "has_company_logo", "has_questions")


def posting_text(record: dict) -> str:
    """Render exactly 16 fields; preserve nonbinary text including HTML/spacing.

    IDs and the target never enter the passage. Empty binary values are Unknown;
    unexpected values fail instead of silently being treated as No.
    """
    sections = []
    for field in INPUT_FIELDS:
        value = record[field]
        value = "" if pd.isna(value) else str(value)
        if field in BINARY_FIELDS:
            key = value.strip()
            if key not in ("0", "1", ""):
                raise ValueError(f"{field} must be 0, 1, or empty.")
            value = {"0": "No", "1": "Yes", "": "Unknown"}[key]
        sections.append(f"{FIELD_LABELS[field]}: {value}")
    return "\n\n".join(sections)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_inputs(project_root: str | Path) -> Path:
    """Save a new input snapshot, retaining all original rows and columns."""
    root = Path(project_root).resolve()
    source = root / "data/raw/fake_job_postings.csv"
    digest = file_sha256(source)
    frame = load_postings(source)
    frame["embedding_text"] = [posting_text(record) for record in frame.to_dict("records")]
    frame["text_sha256"] = frame["embedding_text"].map(text_sha256)
    if file_sha256(source) != digest:
        raise RuntimeError("Raw source changed while constructing inputs.")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "data/embeddings" / INPUT_VERSION / timestamp
    if not output.resolve().is_relative_to(root / "data/embeddings"):
        raise ValueError("Input snapshots must stay inside data/embeddings/.")
    output.mkdir(parents=True, exist_ok=False)
    csv = output / "inputs.csv"
    frame.to_csv(csv, index=False)
    manifest = {
        "input_version": INPUT_VERSION, "source_relative_path": source.relative_to(root).as_posix(),
        "source_sha256": digest, "inputs_sha256": file_sha256(csv), "rows": len(frame),
        "input_fields": list(INPUT_FIELDS), "field_labels": FIELD_LABELS,
        "binary_mapping": {"0": "No", "1": "Yes", "empty": "Unknown"},
        "cleaning": "none", "deduplication": "none", "splitting": "none",
        "ordering": "original CSV order", "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (output / "preview.txt").write_text(
        "Embedding input for the first raw posting (ID and target excluded):\n\n"
        + frame.iloc[0]["embedding_text"] + "\n", encoding="utf-8")
    return output


def load_inputs(folder: str | Path) -> tuple[pd.DataFrame, dict]:
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "input_manifest.json").read_text())
    if manifest["input_version"] != INPUT_VERSION or manifest["input_fields"] != list(INPUT_FIELDS):
        raise ValueError("Input construction does not match the supported version.")
    if file_sha256(folder / "inputs.csv") != manifest["inputs_sha256"]:
        raise ValueError("Input CSV checksum does not match its manifest.")
    frame = load_postings(folder / "inputs.csv")
    if len(frame) != manifest["rows"] or not frame["embedding_text"].map(text_sha256).eq(frame["text_sha256"]).all():
        raise ValueError("Input row count or text hashes do not match.")
    if [posting_text(record) for record in frame.to_dict("records")] != frame["embedding_text"].tolist():
        raise ValueError("Embedding passages do not match their source fields.")
    return frame, manifest
