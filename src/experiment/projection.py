"""One frozen Rademacher basis; independent bipolar roles and explicit sign ties."""

import hashlib
import json
import math

import torch
import torchhd

from .readout import unit


def generator(seed: int, namespace: str) -> torch.Generator:
    key = json.dumps(["seed-v1", seed, namespace], separators=(",", ":")).encode()
    derived = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
    return torch.Generator(device="cpu").manual_seed(derived)


def random_role(dimensions: int, seed: int, name: str) -> torch.Tensor:
    return torchhd.MAPTensor.random(
        1,
        dimensions,
        generator=generator(seed, "role:" + name),
        dtype=torch.int32,
        device="cpu",
    )[0].as_subclass(torch.Tensor)


def make_projection(width: int, dimensions: int, seed: int, roles: tuple[str, ...]) -> dict:
    if width < 1 or dimensions < 1:
        raise ValueError("Projection dimensions must be positive")
    draws = torch.randint(
        0,
        2,
        (width, dimensions),
        generator=generator(seed, "projection"),
        dtype=torch.int32,
    )
    return {
        "matrix": draws * 2 - 1,
        "projection_ties": random_role(dimensions, seed, "projection_ties"),
        "record_ties": random_role(dimensions, seed, "record_ties"),
        "roles": {role: random_role(dimensions, seed, role) for role in roles},
    }


def bipolarize(projected: torch.Tensor, ties: torch.Tensor) -> torch.Tensor:
    if projected.shape[-1] != ties.numel() or not torch.isfinite(projected).all():
        raise ValueError("Sign input must be finite and match the tie vector")
    if not torch.all((ties == 1) | (ties == -1)):
        raise ValueError("Tie vector must be bipolar")
    result = torch.where(projected > 0, 1, torch.where(projected < 0, -1, ties))
    return result.to(torch.int32).as_subclass(torchhd.MAPTensor)


def project(embedding: torch.Tensor, basis: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return linear float64 output and signed MAP values, each [..., D]."""
    matrix = basis["matrix"]
    if embedding.shape[-1] != matrix.shape[0]:
        raise ValueError("Embedding width does not match the projection")
    linear = unit(embedding) @ matrix.to(torch.float64) / math.sqrt(matrix.shape[1])
    return linear, bipolarize(linear, basis["projection_ties"])
