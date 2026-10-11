"""Read and validate the original job-posting CSV without modifying it."""

from pathlib import Path

import pandas as pd

TEXT_FIELDS = ("title", "company_profile", "description", "requirements", "benefits")
EXPECTED_COLUMNS = (
    "job_id", "title", "location", "department", "salary_range",
    "company_profile", "description", "requirements", "benefits",
    "telecommuting", "has_company_logo", "has_questions", "employment_type",
    "required_experience", "required_education", "industry", "function", "fraudulent",
)


def validate_postings(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy with integer IDs/labels; retain all other fields.

    Read raw values as strings first so missing or malformed labels cannot be
    silently converted to a valid class. Missing descriptive fields stay empty.
    """
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The dataset contains no postings.")
    result = frame.copy()
    ids = result["job_id"].astype("string").str.strip()
    if not ids.str.fullmatch(r"[0-9]+").fillna(False).all():
        raise ValueError("job_id must contain nonmissing positive integers.")
    try:
        result["job_id"] = ids.astype("int64")
    except (ValueError, OverflowError) as exc:
        raise ValueError("job_id must fit in a signed 64-bit integer.") from exc
    if result["job_id"].le(0).any():
        raise ValueError("job_id must contain positive integers.")
    if result["job_id"].duplicated().any():
        raise ValueError("Duplicate job_id values found.")
    labels = result["fraudulent"].astype("string").str.strip()
    if not labels.isin(["0", "1"]).all():
        raise ValueError("fraudulent must contain only nonmissing binary labels 0 or 1.")
    result["fraudulent"] = labels.astype("int8")
    return result


def load_postings(path: str | Path) -> pd.DataFrame:
    """Load strings without interpreting text such as 'NA' as missing data."""
    frame = pd.read_csv(Path(path), dtype="string", keep_default_na=False)
    return validate_postings(frame)
