"""Local Nomic embeddings with exact input identity and offline cache reads."""

import json
import urllib.request
from pathlib import Path

import ollama
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import settings as cfg
from .readout import unit
from .storage import digest, tensor_hash


def input_key(text: str) -> str:
    return digest(
        {
            "input": text,
            "model_digest": cfg.MODEL_DIGEST,
            "truncate": False,
            "preprocessing": "nfkc-whitespace-lower-l2-v1",
        }
    )


def check_vector(values: list[float]) -> torch.Tensor:
    vector = torch.tensor(values, dtype=torch.float64)
    if vector.shape != (cfg.EMBEDDING_DIM,):
        raise ValueError(
            f"Expected {cfg.EMBEDDING_DIM} embedding coordinates, got {tuple(vector.shape)}"
        )
    unit(vector)  # Validate finite coordinates and nonzero norm without changing the original.
    return vector


def read_cache(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        return {}
    result = {}
    for row in pq.read_table(path).to_pylist():
        if row["model_digest"] != cfg.MODEL_DIGEST or row["cache_key"] != input_key(row["input"]):
            raise ValueError("Embedding cache identity mismatch")
        if row["input"] in result:
            raise ValueError("Duplicate embedding cache input")
        vector = check_vector(row["embedding"])
        if tensor_hash(unit(vector)) != row["normalized_hash"]:
            raise ValueError("Embedding cache vector checksum mismatch")
        result[row["input"]] = vector
    return result


def cached_vectors(path: Path, inputs: list[str]) -> dict[str, torch.Tensor]:
    cache = read_cache(path)
    missing = set(inputs) - cache.keys()
    if missing:
        raise ValueError(
            f"Offline cache miss: {sorted(missing)}. Run 01_embed.py for fixed inputs."
        )
    return {text: unit(cache[text]) for text in inputs}


def write_cache(path: Path, cache: dict[str, torch.Tensor]) -> None:
    schema = pa.schema(
        [
            ("input", pa.string()),
            ("cache_key", pa.string()),
            ("model_digest", pa.string()),
            ("original_norm", pa.float64()),
            ("normalized_hash", pa.string()),
            ("embedding", pa.list_(pa.float64(), cfg.EMBEDDING_DIM)),
        ]
    )
    rows = [
        {
            "input": text,
            "cache_key": input_key(text),
            "model_digest": cfg.MODEL_DIGEST,
            "original_norm": float(torch.linalg.vector_norm(vector)),
            "normalized_hash": tensor_hash(unit(vector)),
            "embedding": vector.tolist(),
        }
        for text, vector in sorted(cache.items())
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary)
    temporary.replace(path)


def verify_local_model(client: ollama.Client) -> None:
    models = client.list().model_dump()["models"]
    match = next((m for m in models if m["model"] == cfg.MODEL), None)
    if match is None or match["digest"] != cfg.MODEL_DIGEST:
        raise ValueError(
            "Local model digest changed or model is missing; do not silently pull or substitute it"
        )


def warm_cache(path: Path, inputs: list[str], client=None) -> dict:
    cache = read_cache(path)
    missing = sorted(set(inputs) - cache.keys())
    if not missing:
        return {"new_inputs": 0, "cache_only": True}
    client = client or ollama.Client(host=cfg.OLLAMA_HOST, timeout=120)
    verify_local_model(client)
    for start in range(0, len(missing), 32):
        batch = missing[start : start + 32]
        response = client.embed(model=cfg.MODEL, input=batch, truncate=False)
        if len(response.embeddings) != len(batch):
            raise ValueError("Ollama returned the wrong embedding row count")
        for text, values in zip(batch, response.embeddings, strict=True):
            cache[text] = check_vector(values)
    verify_local_model(client)  # Detect a tag change during embedding.
    write_cache(path, cache)
    return {"new_inputs": len(missing), "cache_only": False, "request_order": missing}


def local_version() -> str:
    with urllib.request.urlopen(cfg.OLLAMA_HOST + "/api/version", timeout=10) as response:
        return json.load(response)["version"]
