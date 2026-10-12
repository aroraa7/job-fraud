"""Fit logistic regression with full-data or persisted train/test evaluation."""

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_fscore_support)

from job_fraud.audit import file_sha256
from job_fraud.embedding_inputs import load_inputs


def split_membership(identity: pd.DataFrame, path: Path, settings: dict) -> pd.DataFrame:
    """Persist randomized stratified membership; reject changed row identity."""
    if path.exists():
        membership = pd.read_csv(path)
        if len(membership) != len(identity) or any(
            not np.array_equal(membership[c].to_numpy(), identity[c].to_numpy())
            for c in identity.columns
        ):
            raise ValueError("Saved split does not match posting identity.")
    else:
        train, test = train_test_split(
            np.arange(len(identity)), test_size=settings["test_size"],
            random_state=settings["random_state"], stratify=identity["fraudulent"])
        membership = identity.copy()
        membership["split"] = "train"
        membership.loc[test, "split"] = "test"
        path.parent.mkdir(parents=True, exist_ok=True)
        membership.to_csv(path, index=False)
    if set(membership["split"]) != {"train", "test"}:
        raise ValueError("Saved split must contain train and test only.")
    expected_train, expected_test = train_test_split(
        np.arange(len(identity)), test_size=settings["test_size"],
        random_state=settings["random_state"], stratify=identity["fraudulent"])
    if set(np.flatnonzero(membership["split"].eq("test"))) != set(expected_test):
        raise ValueError("Saved split does not match configured seed and fraction.")
    return membership


def run_baseline(root: Path, config_path: str = "configs/baseline.json") -> Path:
    config = json.loads((root / config_path).read_text())
    snapshot = root / config["snapshot"]
    frame, input_manifest = load_inputs(snapshot)
    if file_sha256(root / input_manifest["source_relative_path"]) != input_manifest["source_sha256"]:
        raise ValueError("Raw source checksum changed.")
    identity = frame[["job_id", "fraudulent", "text_sha256"]]
    y = frame["fraudulent"].to_numpy(dtype=np.int64)
    split = config.get("split")
    train_indices = test_indices = np.arange(len(y))
    split_identity = None
    if split:
        membership = split_membership(identity, root / split["path"], split)
        train_indices = np.flatnonzero(membership["split"].eq("train"))
        test_indices = np.flatnonzero(membership["split"].eq("test"))
        split_identity = {"path": split["path"], "sha256": file_sha256(root / split["path"]),
                          "settings": split, "train_rows": len(train_indices), "test_rows": len(test_indices),
                          "train_support": {str(k): int((y[train_indices] == k).sum()) for k in (0, 1)},
                          "test_support": {str(k): int((y[test_indices] == k).sum()) for k in (0, 1)},
                          "shared_text_hashes": len(set(identity.iloc[train_indices]["text_sha256"]) &
                                                    set(identity.iloc[test_indices]["text_sha256"]))}
    evaluation = "held-out test" if split else "training"
    category = "train_test" if split else "baselines"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_dir = root / "reports" / category / stamp
    model_dir = root / "models" / category / stamp
    report_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    results = {
        "evaluation": "held-out randomized stratified test; no tuning" if split else "training data; all rows used for fit and scoring; no split",
        "split_identity": split_identity,
        "config": config, "source_identity": input_manifest,
        "sklearn_version": sklearn.__version__, "providers": {},
    }
    for provider, directory in config["providers"].items():
        folder = snapshot / directory
        manifest = json.loads((folder / "manifest.json").read_text())
        if manifest["status"] != "complete" or manifest["generation"]["row_selection"] != "all":
            raise ValueError("Expected a completed all-row matrix.")
        for filename, key in (("embeddings.npy", "embeddings_sha256"),
                              ("posting_ids_and_labels.csv", "alignment_sha256")):
            if file_sha256(folder / filename) != manifest[key]:
                raise ValueError(f"Checksum mismatch: {provider}/{filename}")
        alignment = pd.read_csv(folder / "posting_ids_and_labels.csv")
        if len(alignment) != len(identity) or any(
            not np.array_equal(alignment[column].to_numpy(), identity[column].to_numpy())
            for column in identity.columns
        ):
            raise ValueError(f"Posting identity mismatch: {provider}")
        X = np.load(folder / "embeddings.npy", allow_pickle=False)
        if X.shape != (len(y), 768) or not np.isfinite(X).all():
            raise ValueError("Expected finite 768-feature matrix.")
        print(f"Fitting {provider}: {X.shape}", flush=True)
        classifier = LogisticRegression(**config["logistic_regression"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(X[train_indices], y[train_indices])
        probability = classifier.predict_proba(X[test_indices])[:, list(classifier.classes_).index(1)]
        evaluated_y = y[test_indices]
        prediction = (probability >= config["threshold"]).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            evaluated_y, prediction, average="binary", zero_division=0)
        artifact = model_dir / f"{provider}_logistic_regression.joblib"
        joblib.dump({"classifier": classifier, "threshold": config["threshold"],
                     "label_mapping": {0: "legitimate", 1: "fraudulent"},
                     "embedding_manifest": manifest, "config": config,
                     "evaluation": results["evaluation"], "split_identity": split_identity}, artifact)
        restored = joblib.load(artifact)
        if not np.array_equal(probability, restored["classifier"].predict_proba(X[test_indices])[:, 1]):
            raise ValueError("Save/load probabilities differ.")
        metrics = {
            "shape": list(X.shape), "fraud_precision": float(precision),
            "fraud_recall": float(recall), "fraud_f1": float(f1),
            "average_precision": float(average_precision_score(evaluated_y, probability)),
            "accuracy": float(accuracy_score(evaluated_y, prediction)),
            "confusion_matrix_true_rows_predicted_columns_0_1": confusion_matrix(evaluated_y, prediction, labels=[0, 1]).tolist(),
            "class_support": {str(label): int((evaluated_y == label).sum()) for label in (0, 1)},
            "iterations": classifier.n_iter_.tolist(),
            "converged": not any(isinstance(w.message, ConvergenceWarning) for w in caught),
            "warnings": [str(w.message) for w in caught],
            "embedding_manifest_sha256": file_sha256(folder / "manifest.json"),
            "model_path": str(artifact.relative_to(root)),
            "model_sha256": file_sha256(artifact), "save_load_consistent": True,
        }
        results["providers"][provider] = metrics
        predictions = identity.iloc[test_indices].copy()
        predictions["fraud_probability"] = probability
        predictions["prediction"] = prediction
        predictions.to_csv(report_dir / f"{provider}_{evaluation.replace(' ', '_')}_predictions.csv", index=False)
        (report_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Completed {provider}: {json.dumps(metrics)}", flush=True)
    print(f"Results: {report_dir}", flush=True)
    return report_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    run_baseline(Path(__file__).resolve().parents[2], args.config)
