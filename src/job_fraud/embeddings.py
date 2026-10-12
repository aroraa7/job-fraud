"""Embedding generation with local input snapshots and explicit execution flags.

Model packages are imported only when their backend is selected. Defaults do
not download model resources or call the OpenAI API. Actual execution requires
separate user approval; CLI flags are safeguards, not substitutes for approval.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import numpy as np

from job_fraud.audit import file_sha256
from job_fraud.embedding_inputs import INPUT_VERSION, load_inputs, prepare_inputs


def load_openai_key(root: Path) -> None:
    """Read only OPENAI_API_KEY from local .env; never execute file contents."""
    if os.environ.get("OPENAI_API_KEY"):
        return
    path = root / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.strip().partition("=")
        if separator and name.strip() == "OPENAI_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                os.environ["OPENAI_API_KEY"] = value
            return


def token_chunks(tokens: list[int], size: int) -> list[list[int]]:
    if size <= 0 or not tokens:
        raise ValueError("Chunk size must be positive and token input nonempty.")
    return [tokens[start:start + size] for start in range(0, len(tokens), size)]


def weighted_mean(vectors, weights) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or not len(values) or len(weights) != len(values):
        raise ValueError("Chunk vectors and weights must align.")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Chunk vectors must be finite and weights positive.")
    return np.average(values, axis=0, weights=weights).astype(np.float32)


def ordered_openai_vectors(response, count: int) -> np.ndarray:
    """Order response vectors by their explicit input index, not list position."""
    items = sorted(response.data, key=lambda item: item.index)
    if [item.index for item in items] != list(range(count)):
        raise ValueError("OpenAI response indices do not align with the request.")
    values = np.asarray([item.embedding for item in items], dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("OpenAI returned invalid embedding values.")
    return values


def _check_config(config: dict) -> None:
    if config["input_version"] != INPUT_VERSION or config["normalize_vectors"] is not False:
        raise ValueError("This workflow uses raw_all_fields_yes_no_v1 without vector normalization.")
    for provider in ("bert", "openai"):
        settings = config[provider]
        if settings["long_text"] != "nonoverlapping_chunks_token_weighted_mean" or settings["batch_size"] <= 0:
            raise ValueError("Unsupported chunking settings or batch size.")
    if config["bert"]["pooling"] != "mean_content_tokens" or not 3 <= config["bert"]["max_sequence_length"] <= 512:
        raise ValueError("BERT requires content-token mean pooling and a sequence length of 3–512.")
    if not 1 <= config["openai"]["max_chunk_tokens"] <= 8191 or not 1 <= config["openai"]["max_request_tokens"] <= 300000:
        raise ValueError("OpenAI token limits exceed the supported request bounds.")


def openai_plan(texts: list[str], settings: dict) -> tuple[list, dict]:
    import tiktoken
    encoder = tiktoken.get_encoding(settings["encoding"])
    # Special-looking strings in raw advertisements are ordinary input text.
    rows = [token_chunks(encoder.encode(text, disallowed_special=()), settings["max_chunk_tokens"]) for text in texts]
    return rows, {"posting_count": len(rows), "chunk_count": sum(map(len, rows)),
                  "input_tokens": sum(len(chunk) for chunks in rows for chunk in chunks),
                  "tokenizer": settings["encoding"]}


def generate_openai(rows: list, settings: dict, max_api_tokens: int) -> tuple[np.ndarray, dict]:
    tokens = sum(len(chunk) for chunks in rows for chunk in chunks)
    if max_api_tokens < tokens:
        raise ValueError(f"Run requires {tokens} input tokens, exceeding the authorized token cap.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment or ignored .env file.")
    from openai import OpenAI
    # No automatic retries: a failure must be reviewed before another paid call.
    client = OpenAI(max_retries=0, timeout=120)
    flat = [(row, chunk) for row, chunks in enumerate(rows) for chunk in chunks]
    vectors_by_row = [[] for _ in rows]
    weights_by_row = [[] for _ in rows]
    billed_tokens = 0
    response_models = set()
    try:
        start = 0
        while start < len(flat):
            end = start
            request_tokens = 0
            while end < len(flat) and end - start < settings["batch_size"]:
                chunk_tokens = len(flat[end][1])
                if request_tokens + chunk_tokens > settings["max_request_tokens"]:
                    break
                request_tokens += chunk_tokens
                end += 1
            if end == start:
                raise ValueError("A chunk exceeds the per-request token budget.")
            batch = flat[start:end]
            response = client.embeddings.create(model=settings["model"], input=[chunk for _, chunk in batch], encoding_format="float")
            values = ordered_openai_vectors(response, len(batch))
            billed_tokens += response.usage.total_tokens
            response_models.add(response.model)
            for (row, chunk), vector in zip(batch, values, strict=True):
                vectors_by_row[row].append(vector)
                weights_by_row[row].append(len(chunk))
            start = end
            print(f"OpenAI chunks completed: {end}/{len(flat)}", flush=True)
    finally:
        client.close()
    return np.stack([weighted_mean(vectors, weights) for vectors, weights in zip(vectors_by_row, weights_by_row, strict=True)]), {
        "usage_total_tokens": billed_tokens, "response_models": sorted(response_models),
        "tokenizer": settings["encoding"], "chunk_counts": [len(chunks) for chunks in rows],
        "versions": {name: version(name) for name in ("openai", "tiktoken")},
    }


def bert_chunk_record(chunk: list[int], tokenizer) -> dict:
    """Wrap content IDs with BERT's CLS/SEP tokens without retokenizing."""
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError("Expected BERT's two single-sequence boundary tokens.")
    return {"input_ids": [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id],
            "token_type_ids": [0] * (len(chunk) + 2),
            "attention_mask": [1] * (len(chunk) + 2),
            "special_tokens_mask": [1, *([0] * len(chunk)), 1]}


def generate_bert(texts: list[str], settings: dict, allow_download: bool = False) -> tuple[np.ndarray, dict]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    kwargs = {"revision": settings["revision"], "local_files_only": not allow_download}
    tokenizer = AutoTokenizer.from_pretrained(settings["model"], use_fast=False, **kwargs)
    model = AutoModel.from_pretrained(settings["model"], **kwargs)
    if settings["max_sequence_length"] > model.config.max_position_embeddings:
        raise ValueError("Configured sequence length exceeds BERT's position limit.")
    model.eval()
    model.requires_grad_(False)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    size = settings["max_sequence_length"] - tokenizer.num_special_tokens_to_add(pair=False)
    all_vectors = []
    chunk_counts = []
    content_tokens = 0
    for row_index, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
        chunks = token_chunks(tokens, size)
        content_tokens += len(tokens)
        chunk_counts.append(len(chunks))
        vectors = []
        for start in range(0, len(chunks), settings["batch_size"]):
            batch = chunks[start:start + settings["batch_size"]]
            records = [bert_chunk_record(chunk, tokenizer) for chunk in batch]
            encoded = tokenizer.pad(records, padding=True, return_tensors="pt")
            special = encoded.pop("special_tokens_mask").to(device)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                mask = (encoded["attention_mask"] * (1 - special)).unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vectors.extend(pooled.cpu().numpy())
        all_vectors.append(weighted_mean(vectors, [len(chunk) for chunk in chunks]))
        if (row_index + 1) % 100 == 0 or row_index + 1 == len(texts):
            print(f"BERT postings completed: {row_index + 1}/{len(texts)}", flush=True)
    return np.stack(all_vectors), {"device": device, "resolved_model_revision": getattr(model.config, "_commit_hash", None) or settings["revision"],
                                  "tokenizer_class": type(tokenizer).__name__,
                                  "content_tokens": content_tokens, "chunk_counts": chunk_counts,
                                  "versions": {name: version(name) for name in ("torch", "transformers")}}


def save_embeddings(folder: Path, frame, input_manifest: dict, backend: str, config: dict,
                    values: np.ndarray, metadata: dict) -> Path:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(frame) or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Embedding dimensions, row alignment, or numerical validity failed.")
    chunk_counts = metadata.get("chunk_counts")
    if chunk_counts is None or len(chunk_counts) != len(frame) or any(count < 1 for count in chunk_counts):
        raise ValueError("Chunk counts must align with input rows.")
    output = folder / f"{backend}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    output.mkdir(exist_ok=False)
    np.save(output / "embeddings.npy", values, allow_pickle=False)
    alignment = frame[["job_id", "fraudulent", "text_sha256"]].copy()
    alignment["chunk_count"] = chunk_counts
    alignment.to_csv(output / "posting_ids_and_labels.csv", index=False)
    manifest = {"status": "complete", "backend": backend, "shape": list(values.shape), "dtype": "float32",
                "input_manifest": input_manifest, "config": config, "generation": metadata,
                "python": platform.python_version(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "embeddings_sha256": file_sha256(output / "embeddings.npy"),
                "alignment_sha256": file_sha256(output / "posting_ids_and_labels.csv"),
                "normalization": "none; provider-returned vectors and chunk-weighted means are retained"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--backend", choices=("prepare", "bert", "openai"), default="prepare")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Explicit smoke-test subset; omit for every row")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--execute-openai", action="store_true")
    parser.add_argument("--max-api-tokens", type=int)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads((args.config or root / "configs/embeddings.json").read_text())
    _check_config(config)
    if args.backend == "prepare":
        print(prepare_inputs(root))
        return
    if args.input_dir is None:
        parser.error("--input-dir is required; prepare a snapshot first")
    folder = args.input_dir.resolve()
    if not folder.is_relative_to(root / "data/embeddings"):
        parser.error("--input-dir must be inside this project's data/embeddings/")
    frame, manifest = load_inputs(folder)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        frame = frame.iloc[:args.limit].copy()
    texts = frame["embedding_text"].tolist()
    if args.backend == "openai":
        rows, plan = openai_plan(texts, config["openai"])
        print(json.dumps(plan, indent=2))
        if not args.execute_openai:
            return
        if args.max_api_tokens is None or args.max_api_tokens <= 0:
            parser.error("OpenAI execution requires an approved positive --max-api-tokens cap")
        load_openai_key(root)
        values, metadata = generate_openai(rows, config["openai"], args.max_api_tokens)
    else:
        values, metadata = generate_bert(texts, config["bert"], args.allow_model_download)
    metadata["row_selection"] = "all" if args.limit is None else f"first {len(frame)} rows"
    print(save_embeddings(folder, frame, manifest, args.backend, config, values, metadata))


if __name__ == "__main__":
    main()
