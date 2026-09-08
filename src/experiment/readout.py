"""Small, source-independent tensor operations used by the experiments."""

import torch
import torchhd


def unit(vector: torch.Tensor) -> torch.Tensor:
    """L2-normalize along the last axis, returning float64 CPU tensors."""
    value = vector.as_subclass(torch.Tensor).to(dtype=torch.float64, device="cpu")
    if value.ndim == 0 or value.shape[-1] == 0 or not torch.isfinite(value).all():
        raise ValueError("Expected nonempty, finite vectors")
    norms = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if torch.any(norms == 0):
        raise ValueError("Zero vectors have no cosine similarity")
    return value / norms


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.ndim != 1 or left.shape != right.shape:
        raise ValueError("Cosine expects two vectors of equal width")
    return float(unit(left) @ unit(right))


def integer_dot(left: torch.Tensor, right: torch.Tensor) -> int:
    return int(left.to(torch.int64) @ right.to(torch.int64))


def bind(role: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Bind one bipolar role to a same-width integer MAP value or sum."""
    if role.ndim != 1 or role.shape != value.shape:
        raise ValueError("Binding expects equal-width one-dimensional vectors")
    if not torch.all((role == 1) | (role == -1)):
        raise ValueError("A binding role must be bipolar")
    for vector in (role, value):
        if not torch.isfinite(vector).all() or not torch.equal(vector, vector.round()):
            raise ValueError("This MAP demo binds finite integer coordinates")
    return torchhd.bind(
        role.to(torch.int32).as_subclass(torchhd.MAPTensor),
        value.to(torch.int32).as_subclass(torchhd.MAPTensor),
    )


def bundle(vectors: list[torch.Tensor]) -> torch.Tensor:
    if not vectors or any(v.shape != vectors[0].shape for v in vectors):
        raise ValueError("Bundling needs nonempty, equal-width vectors")
    # MAP multibundling preserves dtype: promote before reducing.
    matrix = torch.stack(vectors).to(torch.int32).as_subclass(torchhd.MAPTensor)
    return torchhd.multiset(matrix)


def rank_records(records: dict[str, torch.Tensor], query: torch.Tensor) -> list[dict]:
    """Rank all records; near ties share a rank rather than imply a preference."""
    if not records:
        raise ValueError("Cannot rank an empty collection")
    ids = sorted(records)
    matrix = torch.stack([unit(records[key]) for key in ids])
    scores = matrix @ unit(query)
    rows = [{"person_id": key, "score": float(score)} for key, score in zip(ids, scores)]
    rows.sort(key=lambda row: (-row["score"], row["person_id"]))
    for row in rows:
        others = [r["score"] for r in rows if r["person_id"] != row["person_id"]]
        row["rank"] = 1 + sum(score > row["score"] + 1e-5 for score in others)
        row["tied"] = any(abs(score - row["score"]) <= 1e-5 for score in others)
        row["margin"] = row["score"] - max(others) if others else 0.0
    return rows


def block_score(
    facts: list[dict],
    values: dict[str, torch.Tensor],
    role: str,
    query_value: torch.Tensor,
    roles: tuple[str, ...],
) -> float:
    """Cosine in explicit per-role slots, retaining within-role correlations."""
    record_slots, query_slots = [], []
    for slot in roles:
        members = [values[f["input"]].to(torch.float64) for f in facts if f["role"] == slot]
        record_slots.append(
            torch.stack(members).sum(0) if members else torch.zeros_like(query_value)
        )
        query_slots.append(query_value if slot == role else torch.zeros_like(query_value))
    return cosine(torch.cat(record_slots), torch.cat(query_slots))
