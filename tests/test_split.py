"""Check persisted split identity, disjointness, and repeatability."""

import pandas as pd
import pytest

from job_fraud.baseline import split_membership


def test_persisted_split(tmp_path):
    identity = pd.DataFrame({"job_id": range(1, 101), "fraudulent": [0] * 80 + [1] * 20,
                             "text_sha256": [f"hash{i}" for i in range(100)]})
    settings = {"test_size": 0.2, "random_state": 42}
    path = tmp_path / "split.csv"
    membership = split_membership(identity, path, settings)
    train = membership[membership.split.eq("train")]
    test = membership[membership.split.eq("test")]
    assert len(train) == 80 and len(test) == 20
    assert not set(train.job_id) & set(test.job_id)
    assert test.fraudulent.sum() == 4 and train.fraudulent.sum() == 16
    assert membership.equals(split_membership(identity, path, settings))
    changed = identity.copy()
    changed.loc[0, "job_id"] = 999
    with pytest.raises(ValueError, match="identity"):
        split_membership(changed, path, settings)
    with pytest.raises(ValueError, match="seed"):
        split_membership(identity, path, {**settings, "random_state": 43})
