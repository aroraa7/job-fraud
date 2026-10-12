"""Fit and describe an explicitly in-sample logistic regression baseline."""

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
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, precision_recall_fscore_support)

from job_fraud.audit import file_sha256
from job_fraud.embedding_inputs import load_inputs


def run_baseline(root: Path) -> Path:
    config = json.loads((root / "configs/baseline.json").read_text())
    snapshot = root / config["snapshot"]
    frame, input_manifest = load_inputs(snapshot)
    if file_sha256(root / input_manifest["source_relative_path"]) != input_manifest["source_sha256"]:
        raise ValueError("Raw source checksum changed.")
    identity = frame[["job_id", "fraudulent", "text_sha256"]]
    y = frame["fraudulent"].to_numpy(dtype=np.int64)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_dir = root / "reports/baselines" / stamp
    model_dir = root / "models/baselines" / stamp
    report_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    results = {
        "evaluation": "training data; all rows used for fit and scoring; no split",
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
            classifier.fit(X, y)
        probability = classifier.predict_proba(X)[:, list(classifier.classes_).index(1)]
        prediction = (probability >= config["threshold"]).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, prediction, average="binary", zero_division=0)
        artifact = model_dir / f"{provider}_logistic_regression.joblib"
        joblib.dump({"classifier": classifier, "threshold": config["threshold"],
                     "label_mapping": {0: "legitimate", 1: "fraudulent"},
                     "embedding_manifest": manifest, "config": config,
                     "evaluation": results["evaluation"]}, artifact)
        restored = joblib.load(artifact)
        if not np.array_equal(probability, restored["classifier"].predict_proba(X)[:, 1]):
            raise ValueError("Save/load probabilities differ.")
        metrics = {
            "shape": list(X.shape), "fraud_precision": float(precision),
            "fraud_recall": float(recall), "fraud_f1": float(f1),
            "average_precision": float(average_precision_score(y, probability)),
            "accuracy": float(accuracy_score(y, prediction)),
            "confusion_matrix_true_rows_predicted_columns_0_1": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
            "class_support": {str(label): int((y == label).sum()) for label in (0, 1)},
            "iterations": classifier.n_iter_.tolist(),
            "converged": not any(isinstance(w.message, ConvergenceWarning) for w in caught),
            "warnings": [str(w.message) for w in caught],
            "embedding_manifest_sha256": file_sha256(folder / "manifest.json"),
            "model_path": str(artifact.relative_to(root)),
            "model_sha256": file_sha256(artifact), "save_load_consistent": True,
        }
        results["providers"][provider] = metrics
        predictions = identity.copy()
        predictions["fraud_probability"] = probability
        predictions["prediction"] = prediction
        predictions.to_csv(report_dir / f"{provider}_training_predictions.csv", index=False)
        (report_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Completed {provider}: {json.dumps(metrics)}", flush=True)
    print(f"Results: {report_dir}", flush=True)
    return report_dir


if __name__ == "__main__":
    run_baseline(Path(__file__).resolve().parents[2])
