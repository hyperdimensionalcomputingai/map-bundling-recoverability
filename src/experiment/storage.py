"""Explicit vector schemas and a small provenance ledger for the script sequence."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import lancedb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from . import settings as cfg

STAGES = ("embed", "encode", "probe", "search", "subtract", "report")
SCRIPTS = dict(
    zip(
        STAGES,
        (
            "01_embed.py",
            "02_encode.py",
            "03_probe.py",
            "04_search.py",
            "05_subtract.py",
            "06_report.py",
        ),
    )
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return digest(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.tolist(),
        }
    )


def load_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def fixtures() -> tuple[list[dict], list[dict]]:
    return load_json(cfg.DATA_DIR / "people.json"), load_json(cfg.DATA_DIR / "queries.json")


def contract() -> dict:
    return {
        "encoder_version": cfg.ENCODER_VERSION,
        "model": cfg.MODEL,
        "model_digest": cfg.MODEL_DIGEST,
        "embedding_dim": cfg.EMBEDDING_DIM,
        "dimensions": cfg.DIMENSIONS,
        "seed": cfg.SEED,
        "roles": list(cfg.ROLES),
        "prefixes": cfg.PREFIXES,
        "truncate": False,
        "preprocessing": "nfkc-whitespace-lower-l2-v1",
        "projection": "iid-rademacher-divide-sqrt-D-then-sign",
        "device": "cpu",
        "dtypes": {
            "embedding": "float64",
            "raw_sum": "int32",
            "dot": "int64",
            "search": "float32",
        },
        "data_hashes": {p.name: file_hash(p) for p in sorted(cfg.DATA_DIR.glob("*.json"))},
    }


def environment() -> dict:
    packages = (
        "torch",
        "torch-hd",
        "ollama",
        "polars",
        "pyarrow",
        "lancedb",
        "matplotlib",
    )
    result = {name: importlib.metadata.version(name) for name in packages}
    result["python"] = platform.python_version()
    result["platform"] = platform.platform()
    return result


def source_state() -> dict:
    paths = list((cfg.ROOT / "src").rglob("*.py")) + [
        cfg.ROOT / "pyproject.toml",
        cfg.ROOT / "uv.lock",
    ]
    hashes = {str(p.relative_to(cfg.ROOT)): file_hash(p) for p in paths if p.exists()}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=cfg.ROOT, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=cfg.ROOT, text=True, capture_output=True, check=False
    )
    return {
        "hashes": hashes,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(status.stdout.strip()),
    }


def ensure_run(run_dir: Path = cfg.RUN_DIR) -> dict:
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    path = run_dir / "manifest.json"
    current = contract()
    if path.exists():
        manifest = load_json(path)
        if manifest["contract"] != current or manifest["environment"] != environment():
            raise ValueError("Run identity changed. Choose a new RUN_DIR in settings.py.")
        return manifest
    manifest = {
        "contract": current,
        "encoder_id": digest(current),
        "environment": environment(),
        "stages": {},
        "complete": False,
    }
    write_json(path, manifest)
    return manifest


def require_stage(stage: str, run_dir: Path = cfg.RUN_DIR) -> dict:
    manifest = ensure_run(run_dir)
    entry = manifest["stages"].get(stage)
    if entry is None:
        raise RuntimeError(
            f"Missing or incomplete {stage} stage; run uv run src/{SCRIPTS[stage]} first"
        )
    for name, expected in entry["outputs"].items():
        path = run_dir / name
        if not path.exists() or file_hash(path) != expected:
            raise ValueError(f"Changed or missing artifact {name}; rerun src/{SCRIPTS[stage]}")
    return manifest


def begin_stage(stage: str, dependencies: tuple[str, ...], run_dir: Path = cfg.RUN_DIR) -> dict:
    manifest = ensure_run(run_dir)
    for dependency in dependencies:
        require_stage(dependency, run_dir)
    for downstream in STAGES[STAGES.index(stage) :]:
        manifest["stages"].pop(downstream, None)
    manifest["complete"] = False
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def finish_stage(
    stage: str,
    outputs: list[Path],
    run_dir: Path = cfg.RUN_DIR,
    details: dict | None = None,
) -> None:
    manifest = ensure_run(run_dir)
    manifest["stages"][stage] = {
        "script": SCRIPTS[stage],
        "source": source_state(),
        "details": details or {},
        "outputs": {str(path.relative_to(run_dir)): file_hash(path) for path in outputs},
    }
    manifest["complete"] = all(name in manifest["stages"] for name in STAGES)
    write_json(run_dir / "manifest.json", manifest)


def write_table(
    name: str,
    rows: list[dict],
    run_dir: Path = cfg.RUN_DIR,
    schema: pa.Schema | None = None,
) -> Path:
    path = run_dir / "tables" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)
    return path


def read_table(name: str, run_dir: Path = cfg.RUN_DIR) -> pl.DataFrame:
    return pl.read_parquet(run_dir / "tables" / f"{name}.parquet")


def records_schema(dimensions: int) -> pa.Schema:
    return pa.schema(
        [
            ("person_id", pa.string()),
            ("encoder_id", pa.string()),
            ("fact_count", pa.int32()),
            ("raw_norm", pa.float64()),
            ("raw_sum", pa.list_(pa.int32(), dimensions)),
            ("vector", pa.list_(pa.float32(), dimensions)),
        ]
    )


def facts_schema(dimensions: int) -> pa.Schema:
    fields = [(name, pa.string()) for name in ("fact_id", "person_id", "role", "text", "input")]
    return pa.schema(
        fields
        + [
            ("value", pa.list_(pa.int32(), dimensions)),
            ("bound", pa.list_(pa.int32(), dimensions)),
        ]
    )


def load_records(run_dir: Path = cfg.RUN_DIR) -> dict[str, torch.Tensor]:
    manifest = require_stage("encode", run_dir)
    rows = read_table("records", run_dir).to_dicts()
    if any(row["encoder_id"] != manifest["encoder_id"] for row in rows):
        raise ValueError("Stored record encoder identity mismatch")
    return {row["person_id"]: torch.tensor(row["raw_sum"], dtype=torch.int32) for row in rows}


def load_basis(run_dir: Path = cfg.RUN_DIR) -> dict:
    require_stage("encode", run_dir)
    return torch.load(run_dir / "projection.pt", map_location="cpu", weights_only=True)


def store_lance_records(run_dir: Path = cfg.RUN_DIR) -> None:
    table = pq.read_table(run_dir / "tables" / "records.parquet")
    lancedb.connect(str(run_dir / "lancedb")).create_table("records", table, mode="overwrite")


def lance_scores(query: torch.Tensor, run_dir: Path = cfg.RUN_DIR) -> dict[str, float]:
    from .readout import unit

    manifest = require_stage("encode", run_dir)
    table = lancedb.connect(str(run_dir / "lancedb")).open_table("records")
    expected = pq.read_table(run_dir / "tables" / "records.parquet")
    # The teaching table is tiny: audit its contents before claiming parity.
    if not table.to_arrow().combine_chunks().equals(expected.combine_chunks()):
        raise ValueError("LanceDB records differ from the frozen Arrow source; rerun 02_encode.py")
    result = (
        table.search(unit(query).to(torch.float32).tolist())
        .distance_type("cosine")
        .bypass_vector_index()
        .limit(table.count_rows())
        .to_arrow()
    )
    rows = pl.from_arrow(result).to_dicts()
    if any(row["encoder_id"] != manifest["encoder_id"] for row in rows):
        raise ValueError("LanceDB encoder identity mismatch")
    return {row["person_id"]: 1.0 - row["_distance"] for row in rows}
