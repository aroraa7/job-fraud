"""Small fixtures exercise integrity failures and leakage-relevant duplicate checks."""

import json

import pandas as pd
import pytest

from job_fraud.audit import audit_project, build_audit
from job_fraud.data import EXPECTED_COLUMNS, load_postings, validate_postings


def postings():
    records = []
    for job_id, label in [(1, 0), (2, 1), (3, 0)]:
        row = dict.fromkeys(EXPECTED_COLUMNS, "")
        row.update(job_id=str(job_id), fraudulent=str(label), title=f"Job {job_id}")
        records.append(row)
    return pd.DataFrame(records)


@pytest.mark.parametrize("value", ["", "2", "-1", "fraud", "0.0", None])
def test_invalid_labels_rejected(value):
    frame = postings()
    frame.loc[0, "fraudulent"] = value
    with pytest.raises(ValueError, match="binary labels"):
        validate_postings(frame)


@pytest.mark.parametrize("value", ["", "0", "-1", "1.5", None])
def test_invalid_ids_rejected(value):
    frame = postings()
    frame.loc[0, "job_id"] = value
    with pytest.raises(ValueError, match="job_id"):
        validate_postings(frame)


def test_duplicate_ids_and_missing_columns_rejected():
    frame = postings()
    frame.loc[1, "job_id"] = "1"
    with pytest.raises(ValueError, match="Duplicate job_id"):
        validate_postings(frame)
    with pytest.raises(ValueError, match="description"):
        validate_postings(postings().drop(columns="description"))
    with pytest.raises(ValueError, match="no postings"):
        validate_postings(postings().iloc[:0])


def test_loader_preserves_text_and_posting_order(tmp_path):
    frame = postings().iloc[[2, 0, 1]].copy()
    frame.loc[0, "location"] = "NA"
    frame.loc[0, "description"] = "<p>Hello, world!</p>\nSecond line"
    path = tmp_path / "raw.csv"
    frame.to_csv(path, index=False)
    loaded = load_postings(path)
    assert loaded.job_id.tolist() == [3, 1, 2]
    assert loaded.loc[1, "location"] == "NA"
    assert loaded.loc[1, "description"] == frame.loc[0, "description"]


def test_normalized_text_conflicts_ignore_metadata_and_preserve_input():
    frame = postings()
    frame.loc[0, ["title", "description", "location"]] = ["Engineer", "Build systems", "US"]
    frame.loc[1, ["title", "description", "location"]] = [" ENGINEER ", "build   systems", "UK"]
    original = frame.copy(deep=True)
    report, memberships = build_audit(frame)
    assert report["duplicates"]["exact_posting_fields"]["groups"] == 0
    assert report["duplicates"]["exact_text_fields"]["groups"] == 0
    assert report["duplicates"]["normalized_text_fields"]["groups_with_conflicting_labels"] == 1
    rows = memberships[memberships.duplicate_kind.eq("normalized_text_fields")]
    assert set(rows.job_id) == {1, 2}
    assert rows.label_conflict.all()
    pd.testing.assert_frame_equal(frame, original)


def test_exact_duplicates_and_empty_text_are_distinguished():
    frame = postings()
    frame.loc[1, "title"] = frame.loc[0, "title"]
    frame.loc[2, "title"] = "  "
    report, _ = build_audit(frame)
    assert report["duplicates"]["exact_posting_fields"]["groups_with_conflicting_labels"] == 1
    assert report["duplicates"]["exact_text_fields"]["postings_in_groups"] == 2
    assert report["all_text_fields_empty_count"] == 1
    assert report["missingness"]["title"]["empty_count"] == 1


def test_audit_artifacts_leave_raw_bytes_unchanged(tmp_path):
    source = tmp_path / "data/raw/fake_job_postings.csv"
    source.parent.mkdir(parents=True)
    postings().to_csv(source, index=False)
    original = source.read_bytes()
    paths = audit_project(tmp_path)
    report = json.loads(paths["json"].read_text())
    assert source.read_bytes() == original
    assert report["label_counts"] == {"0": 2, "1": 1}
    assert report["schema_checks"]["binary_nonmissing_labels"] == "passed"
    assert all(path.is_relative_to(tmp_path / "reports") for path in paths.values())
    assert list(pd.read_csv(paths["duplicates"]).columns) == [
        "duplicate_kind", "group_id", "job_id", "fraudulent", "label_conflict"]
    assert "Raw job-posting data audit" in paths["markdown"].read_text()
