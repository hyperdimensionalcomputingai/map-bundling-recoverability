"""Cache and artifact contracts are checked without a running model."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from experiment import embeddings, storage
from experiment import settings as cfg


@pytest.fixture
def tiny_embeddings(monkeypatch):
    monkeypatch.setattr(cfg, "EMBEDDING_DIM", 3)


class FakeClient:
    def __init__(self):
        self.calls = []

    def list(self):
        return SimpleNamespace(
            model_dump=lambda: {"models": [{"model": cfg.MODEL, "digest": cfg.MODEL_DIGEST}]}
        )

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(embeddings=[[1.0, 2.0, 3.0] for _ in kwargs["input"]])


def test_cache_batches_exact_inputs_and_reuses_offline(tmp_path, tiny_embeddings, monkeypatch):
    path, client = tmp_path / "embeddings.parquet", FakeClient()
    inputs = ["search_document: jazz", "search_query: jazz"]
    details = embeddings.warm_cache(path, inputs, client)
    assert details["new_inputs"] == 2
    assert client.calls[0]["truncate"] is False
    assert len(embeddings.read_cache(path)) == 2
    monkeypatch.setattr(
        embeddings.ollama,
        "Client",
        lambda **kw: pytest.fail("Offline reuse contacted Ollama"),
    )
    assert embeddings.warm_cache(path, inputs)["cache_only"]
    assert torch.allclose(
        torch.linalg.vector_norm(embeddings.cached_vectors(path, inputs)[inputs[0]]),
        torch.tensor(1.0, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="Offline cache miss"):
        embeddings.cached_vectors(path, ["search_query: new text"])


def test_cache_rejects_changed_model(tmp_path, tiny_embeddings, monkeypatch):
    path = tmp_path / "embeddings.parquet"
    embeddings.write_cache(
        path,
        {"search_document: jazz": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)},
    )
    monkeypatch.setattr(cfg, "MODEL_DIGEST", "changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        embeddings.read_cache(path)


def test_cache_rejects_modified_coordinates(tmp_path, tiny_embeddings):
    path = tmp_path / "embeddings.parquet"
    embeddings.write_cache(
        path,
        {"search_document: jazz": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)},
    )
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["embedding"][0] = 7.0
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        embeddings.read_cache(path)


@pytest.mark.parametrize("values", [[1.0, 2.0], [0.0, 0.0, 0.0], [1.0, float("nan"), 3.0]])
def test_bad_model_outputs(values, tiny_embeddings):
    with pytest.raises(ValueError):
        embeddings.check_vector(values)


def test_stage_completion_invalidation_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "contract", lambda: {"test": 1})
    monkeypatch.setattr(storage, "environment", dict)
    monkeypatch.setattr(storage, "source_state", dict)
    output = tmp_path / "test.txt"
    output.write_text("evidence")
    for stage in storage.STAGES:
        storage.begin_stage(stage, (), tmp_path)
        storage.finish_stage(stage, [output], tmp_path)
    assert storage.ensure_run(tmp_path)["complete"]
    storage.begin_stage("search", ("encode", "probe"), tmp_path)
    assert set(storage.ensure_run(tmp_path)["stages"]) == {"embed", "encode", "probe"}
    assert not storage.ensure_run(tmp_path)["complete"]
    output.write_text("corrupt")
    with pytest.raises(ValueError, match="Changed or missing"):
        storage.require_stage("encode", tmp_path)


def test_run_identity_change_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "contract", lambda: {"seed": 1})
    monkeypatch.setattr(storage, "environment", dict)
    storage.ensure_run(tmp_path)
    monkeypatch.setattr(storage, "contract", lambda: {"seed": 2})
    with pytest.raises(ValueError, match="Run identity changed"):
        storage.ensure_run(tmp_path)


def test_arrow_preserves_raw_integer_and_search_float_vectors(tmp_path):
    schema = storage.records_schema(3)
    rows = [
        {
            "person_id": "a",
            "encoder_id": "test",
            "fact_count": 3,
            "raw_norm": 5.0,
            "raw_sum": [3, 0, -4],
            "vector": [0.6, 0.0, -0.8],
        }
    ]
    path = storage.write_table("records", rows, tmp_path, schema)
    restored = pq.read_table(path)
    assert restored.schema == schema
    assert restored.to_pylist()[0]["raw_sum"] == [3, 0, -4]
    assert storage.read_table("records", tmp_path)["person_id"].to_list() == ["a"]


def test_new_query_uses_frozen_records_and_basis(tmp_path, tiny_embeddings, monkeypatch):
    import runpy

    from experiment.projection import make_projection, project
    from experiment.readout import bind, unit

    monkeypatch.setattr(cfg, "RUN_DIR", tmp_path)
    monkeypatch.setattr(cfg, "DIMENSIONS", 32)
    monkeypatch.setattr(storage, "source_state", dict)
    basis = make_projection(3, 32, 2026, cfg.ROLES)
    torch.save(basis, tmp_path / "projection.pt")
    manifest = storage.ensure_run(tmp_path)
    embedding = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    embeddings.write_cache(tmp_path / "embeddings.parquet", {"search_document: jazz": embedding})
    vector = bind(basis["roles"]["interest"], project(embedding, basis)[1])
    records_path = storage.write_table(
        "records",
        [
            {
                "person_id": "a",
                "encoder_id": manifest["encoder_id"],
                "fact_count": 1,
                "raw_norm": float(torch.linalg.vector_norm(vector.double())),
                "raw_sum": vector.tolist(),
                "vector": unit(vector).float().tolist(),
            }
        ],
        tmp_path,
        storage.records_schema(32),
    )
    storage.store_lance_records(tmp_path)
    storage.finish_stage("encode", [records_path, tmp_path / "projection.pt"], tmp_path)
    frozen = [
        records_path,
        tmp_path / "projection.pt",
        tmp_path / "embeddings.parquet",
        tmp_path / "manifest.json",
    ]
    before = {path: storage.file_hash(path) for path in frozen}
    client = FakeClient()
    monkeypatch.setattr(embeddings.ollama, "Client", lambda **kw: client)
    namespace = runpy.run_path(str(cfg.ROOT / "src" / "07_try_query.py"))
    main = namespace["main"]
    main.__globals__.update(
        {
            "QUERY_TEXT": "a previously unseen phrase",
            "ROLE": "interest",
            "require_stage": lambda stage: storage.require_stage(stage, tmp_path),
            "load_basis": lambda: storage.load_basis(tmp_path),
            "load_records": lambda: storage.load_records(tmp_path),
            "lance_scores": lambda query: storage.lance_scores(query, tmp_path),
        }
    )
    main()
    assert client.calls[0]["input"] == ["search_query: a previously unseen phrase"]
    assert before == {path: storage.file_hash(path) for path in frozen}
    results = list((tmp_path / "queries").glob("*.json"))
    assert len(results) == 1
    assert storage.load_json(results[0])["results"][0]["score"] == pytest.approx(1.0)
    main()
    assert len(client.calls) == 1
