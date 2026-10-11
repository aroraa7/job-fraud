"""Read-only dataset audit. Run with ``python -m job_fraud.audit``."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re

import pandas as pd

from job_fraud.data import TEXT_FIELDS, load_postings, validate_postings

MEMBERSHIP_COLUMNS = ["duplicate_kind", "group_id", "job_id", "fraudulent", "label_conflict"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _normalize(value: object) -> str:
    # Used ONLY for duplicate detection, not for model input preprocessing.
    return re.sub(r"\s+", " ", _text(value)).strip().casefold()


def _length_summary(values: pd.Series) -> dict:
    return {
        "min": int(values.min()), "median": float(values.median()),
        "p90": float(values.quantile(0.90)), "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)), "max": int(values.max()),
    }


def _duplicate_audit(frame: pd.DataFrame, fields: list[str], kind: str,
                     normalized: bool = False, skip_empty: bool = False) -> tuple[dict, list[dict]]:
    groups = defaultdict(list)
    for record in frame[["job_id", "fraudulent", *fields]].to_dict("records"):
        values = [(_normalize if normalized else _text)(record[field]) for field in fields]
        if skip_empty and not any(value.strip() for value in values):
            continue
        # JSON encoding keeps field boundaries unambiguous.
        key = hashlib.sha256(json.dumps(values, ensure_ascii=False).encode("utf-8")).hexdigest()
        groups[key].append((int(record["job_id"]), int(record["fraudulent"])))
    duplicates = {key: rows for key, rows in groups.items() if len(rows) > 1}
    conflicts = 0
    memberships = []
    for key, rows in sorted(duplicates.items()):
        conflict = len({label for _, label in rows}) > 1
        conflicts += int(conflict)
        for job_id, label in rows:
            memberships.append({"duplicate_kind": kind, "group_id": key,
                                "job_id": job_id, "fraudulent": label,
                                "label_conflict": conflict})
    return {
        "fields": fields, "groups": len(duplicates),
        "postings_in_groups": len(memberships),
        "extra_postings_beyond_one_per_group": sum(len(rows) - 1 for rows in duplicates.values()),
        "groups_with_conflicting_labels": conflicts,
        "postings_in_conflicting_groups": sum(row["label_conflict"] for row in memberships),
    }, memberships


def build_audit(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Describe data and duplicate candidates without filtering or relabeling rows."""
    frame = validate_postings(frame)
    count = len(frame)
    class_counts = {str(label): int(frame["fraudulent"].eq(label).sum()) for label in (0, 1)}
    missingness = {}
    for field in frame.columns:
        empty = frame[field].map(_text).str.strip().eq("")
        missingness[field] = {
            "empty_count": int(empty.sum()), "empty_percent": float(empty.mean() * 100),
            "empty_by_label": {str(label): int((empty & frame["fraudulent"].eq(label)).sum())
                               for label in (0, 1)},
        }
    text = frame[list(TEXT_FIELDS)].map(_text)
    combined = text.apply(lambda row: "\n".join(value for value in row if value.strip()), axis=1)
    lengths = {}
    for field, values in [(field, text[field]) for field in TEXT_FIELDS] + [("combined_text", combined)]:
        lengths[field] = {"characters": _length_summary(values.str.len()),
                          "whitespace_words": _length_summary(values.str.split().str.len())}
    combined_words = combined.str.split().str.len()
    by_label = {}
    for label in (0, 1):
        subset = combined_words[frame["fraudulent"].eq(label)]
        by_label[str(label)] = _length_summary(subset) if len(subset) else None
    duplicates = {}
    memberships = []
    specs = [
        ("exact_posting_fields", [c for c in frame.columns if c not in ("job_id", "fraudulent")], False, False),
        ("exact_text_fields", list(TEXT_FIELDS), False, True),
        ("normalized_text_fields", list(TEXT_FIELDS), True, True),
    ]
    for kind, fields, normalized, skip_empty in specs:
        duplicates[kind], rows = _duplicate_audit(frame, fields, kind, normalized, skip_empty)
        memberships.extend(rows)
    result = {
        "audit_version": 1, "rows": count, "columns": frame.columns.tolist(),
        "schema_checks": {"required_columns": "passed", "unique_positive_job_ids": "passed",
                          "binary_nonmissing_labels": "passed"},
        "label_counts": class_counts,
        "fraud_percent": class_counts["1"] / count * 100,
        "majority_class_accuracy_percent": max(class_counts.values()) / count * 100,
        "text_fields": list(TEXT_FIELDS), "missingness": missingness,
        "all_text_fields_empty_count": int(combined.str.strip().eq("").sum()),
        "text_lengths": lengths, "combined_word_lengths_by_label": by_label,
        "duplicates": duplicates,
        "limitations": [
            "Whitespace word counts are not BERT or OpenAI tokenizer counts.",
            "Normalized duplicates only ignore case and repeated whitespace; fuzzy/near duplicates are not detected.",
            "All-empty text rows are excluded from text duplicate groups and counted separately.",
            "Duplicate memberships overlap across definitions; do not add their counts.",
            "No rows have been removed, labels changed, input text finalized, or splits constructed.",
            "Raw text, including HTML and punctuation, is retained. Only IDs and labels are parsed as integers.",
        ],
    }
    return result, pd.DataFrame(memberships, columns=MEMBERSHIP_COLUMNS)


def render_markdown(report: dict) -> str:
    lines = ["# Raw job-posting data audit", "",
             f"Source: `{report['source']['relative_path']}`",
             f"SHA-256: `{report['source']['sha256']}`", "",
             f"Postings: **{report['rows']:,}**; columns: **{len(report['columns'])}**.",
             "Required columns, positive unique IDs, and binary labels passed validation.", "",
             "| Label | Meaning | Count |", "| --- | --- | ---: |",
             f"| 0 | Legitimate | {report['label_counts']['0']:,} |",
             f"| 1 | Fraudulent | {report['label_counts']['1']:,} |", "",
             f"Fraud share: **{report['fraud_percent']:.2f}%**. Always predicting the majority class gives "
             f"**{report['majority_class_accuracy_percent']:.2f}%** accuracy.", "",
             "## Missing fields", "", "Blank and whitespace-only fields count as empty.", "",
             "| Field | Empty count | Empty % |", "| --- | ---: | ---: |"]
    for field, stats in report["missingness"].items():
        lines.append(f"| {field} | {stats['empty_count']:,} | {stats['empty_percent']:.2f} |")
    lines += ["", f"Postings with all five proposed text fields empty: {report['all_text_fields_empty_count']:,}.",
              "", "## Text lengths", "",
              "Combined text joins nonblank raw fields with newlines for measurement only; it is not finalized model input.",
              "Whitespace words include HTML fragments and must not be interpreted as tokenizer counts.", "",
              "| Field | Median words | 95th percentile words | Maximum words |", "| --- | ---: | ---: | ---: |"]
    for field, stats in report["text_lengths"].items():
        words = stats["whitespace_words"]
        lines.append(f"| {field} | {words['median']:.1f} | {words['p95']:.1f} | {words['max']:,} |")
    lines += ["", "## Duplicate candidates", "",
              "IDs and labels are excluded from duplicate matching so label conflicts remain visible.",
              "Text matching uses all five text fields; normalized matching ignores case and repeated whitespace.", "",
              "| Definition | Groups | Postings in groups | Groups with conflicting labels |",
              "| --- | ---: | ---: | ---: |"]
    for kind, stats in report["duplicates"].items():
        lines.append(f"| {kind} | {stats['groups']:,} | {stats['postings_in_groups']:,} | {stats['groups_with_conflicting_labels']:,} |")
    lines += ["", "See `duplicate_memberships.csv` for posting IDs, group identities, and conflicts.",
              "Counts overlap across definitions. These groups are candidates for keeping together during splitting.",
              "", "## Limits and next decisions", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += ["", "Review conflicts and related duplicates before splitting; decide on HTML handling, input fields, and long-text coverage.",
              "", "## Run metadata", "", f"Generated (UTC): {report['generated_at_utc']}",
              f"Versions: `{json.dumps(report['versions'], sort_keys=True)}`", ""]
    return "\n".join(lines)


def audit_project(project_root: str | Path) -> dict[str, Path]:
    """Write a new timestamped report under reports/; leave raw data untouched."""
    root = Path(project_root).resolve()
    source = root / "data/raw/fake_job_postings.csv"
    digest = file_sha256(source)
    report, memberships = build_audit(load_postings(source))
    if file_sha256(source) != digest:
        raise RuntimeError("Source CSV changed during the audit; no report was written.")
    now = datetime.now(timezone.utc)
    report["source"] = {"relative_path": source.relative_to(root).as_posix(), "sha256": digest}
    report["generated_at_utc"] = now.isoformat()
    report["versions"] = {"python": platform.python_version(),
                          **{name: version(name) for name in ("pandas", "scikit-learn", "pytest")}}
    output = root / "reports/data_audit" / now.strftime("%Y%m%dT%H%M%S%fZ")
    # Reject symlinks that would redirect report output into raw data or outside reports.
    resolved = output.resolve()
    if not resolved.is_relative_to(root / "reports") or resolved.is_relative_to(root / "data/raw"):
        raise ValueError("Report output must remain inside the project's reports directory.")
    output.mkdir(parents=True, exist_ok=False)
    paths = {"json": output / "audit.json", "markdown": output / "audit.md",
             "duplicates": output / "duplicate_memberships.csv"}
    paths["json"].write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    memberships.to_csv(paths["duplicates"], index=False)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    for kind, path in audit_project(args.project_root).items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
