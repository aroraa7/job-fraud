"""Check input preservation, target exclusion, chunk coverage, and saved alignment."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from job_fraud.data import EXPECTED_COLUMNS
from job_fraud.embedding_inputs import INPUT_FIELDS, load_inputs, posting_text, prepare_inputs
from job_fraud.embeddings import bert_chunk_record, generate_openai, load_openai_key, ordered_openai_vectors, save_embeddings, token_chunks, weighted_mean


def record():
    row = dict.fromkeys(EXPECTED_COLUMNS, "")
    row.update(job_id="4321", fraudulent="1", title="Engineer", description="<p>Build &amp; test.</p>  ",
               telecommuting="1", has_company_logo="0", has_questions="1", location="NA")
    return row


def snapshot(tmp_path):
    source = tmp_path / "data/raw/fake_job_postings.csv"
    source.parent.mkdir(parents=True)
    rows = [record(), {**record(), "job_id": "4322"}]
    pd.DataFrame(rows).to_csv(source, index=False)
    original = source.read_bytes()
    folder = prepare_inputs(tmp_path)
    assert source.read_bytes() == original
    return source, folder


def test_all_fields_rendered_without_cleaning_or_target_leakage():
    row = record()
    text = posting_text(row)
    assert len(INPUT_FIELDS) == 16
    assert "Remote work offered: Yes" in text
    assert "Company logo present: No" in text
    assert "Screening questions present: Yes" in text
    assert "Description: <p>Build &amp; test.</p>  " in text
    assert "Location: NA" in text
    assert "4321" not in text
    assert "fraudulent" not in text
    assert posting_text({**row, "job_id": "9999", "fraudulent": "0"}) == text
    assert row["has_company_logo"] == "0"


@pytest.mark.parametrize("field", ["telecommuting", "has_company_logo", "has_questions"])
def test_invalid_binary_values_rejected(field):
    with pytest.raises(ValueError, match=field):
        posting_text({**record(), field: "2"})


def test_missing_binary_is_not_no():
    assert "Remote work offered: Unknown" in posting_text({**record(), "telecommuting": ""})


def test_snapshot_preserves_duplicates_labels_and_raw_bytes(tmp_path):
    source, folder = snapshot(tmp_path)
    original = source.read_bytes()
    frame, manifest = load_inputs(folder)
    assert source.read_bytes() == original
    assert frame.job_id.tolist() == [4321, 4322]
    assert frame.fraudulent.tolist() == [1, 1]
    assert frame.telecommuting.tolist() == ["1", "1"]
    assert frame.embedding_text.iloc[0] == frame.embedding_text.iloc[1]
    assert manifest["deduplication"] == manifest["splitting"] == "none"
    assert manifest["rows"] == 2
    assert set(EXPECTED_COLUMNS).issubset(frame.columns)


def test_changed_input_snapshot_is_rejected(tmp_path):
    _, folder = snapshot(tmp_path)
    with (folder / "inputs.csv").open("a") as output:
        output.write("tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        load_inputs(folder)


def test_token_chunks_preserve_every_token_in_order():
    tokens = list(range(11))
    chunks = token_chunks(tokens, 4)
    assert list(map(len, chunks)) == [4, 4, 3]
    assert [token for chunk in chunks for token in chunk] == tokens
    with pytest.raises(ValueError):
        token_chunks([], 4)


def test_weighted_chunk_average_is_not_renormalized():
    result = weighted_mean([[1, 0], [0, 1]], [3, 1])
    np.testing.assert_allclose(result, [0.75, 0.25])
    assert not np.isclose(np.linalg.norm(result), 1)
    with pytest.raises(ValueError):
        weighted_mean([[np.nan, 0]], [1])


def test_bert_chunk_preserves_content_and_masks_only_added_boundaries():
    tokenizer = SimpleNamespace(cls_token_id=101, sep_token_id=102,
                                num_special_tokens_to_add=lambda pair: 2)
    # A token with the same ID as a boundary can occur in raw content.
    record = bert_chunk_record([22, 102, 33], tokenizer)
    assert record["input_ids"] == [101, 22, 102, 33, 102]
    assert record["special_tokens_mask"] == [1, 0, 0, 0, 1]
    assert record["attention_mask"] == [1] * 5
    assert record["token_type_ids"] == [0] * 5


def test_response_indices_restore_order_and_reject_missing_rows():
    response = SimpleNamespace(data=[SimpleNamespace(index=1, embedding=[2, 3]),
                                     SimpleNamespace(index=0, embedding=[0, 1])])
    np.testing.assert_array_equal(ordered_openai_vectors(response, 2), [[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="indices"):
        ordered_openai_vectors(response, 3)


def test_api_token_cap_checked_before_import_or_request():
    with pytest.raises(ValueError, match="token cap"):
        generate_openai([[[1, 2, 3]]], {}, max_api_tokens=2)


def test_local_key_loader_respects_existing_environment_and_does_not_execute(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text('OTHER=$(touch unsafe)\nOPENAI_API_KEY="fixture-key"\n')
    load_openai_key(tmp_path)
    import os
    assert os.environ["OPENAI_API_KEY"] == "fixture-key"
    assert not (tmp_path / "unsafe").exists()
    monkeypatch.setenv("OPENAI_API_KEY", "existing-key")
    load_openai_key(tmp_path)
    assert os.environ["OPENAI_API_KEY"] == "existing-key"


def test_embedding_save_roundtrip_retains_ids_labels_and_vectors(tmp_path):
    _, folder = snapshot(tmp_path)
    frame, manifest = load_inputs(folder)
    values = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    output = save_embeddings(folder, frame, manifest, "test_fixture", {}, values, {"chunk_counts": [1, 2]})
    np.testing.assert_array_equal(np.load(output / "embeddings.npy", allow_pickle=False), values)
    alignment = pd.read_csv(output / "posting_ids_and_labels.csv")
    assert alignment.job_id.tolist() == [4321, 4322]
    assert alignment.fraudulent.tolist() == [1, 1]
    assert alignment.chunk_count.tolist() == [1, 2]
    assert alignment.text_sha256.tolist() == frame.text_sha256.tolist()
    assert json.loads((output / "manifest.json").read_text())["status"] == "complete"
    with pytest.raises(ValueError, match="row alignment"):
        save_embeddings(folder, frame, manifest, "test_fixture", {}, values[:1], {"chunk_counts": [1]})
