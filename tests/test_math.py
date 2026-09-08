"""Small exact examples test the algebra, never a preferred semantic ranking."""

import math

import pytest
import torch
import torchhd

from experiment.encoding import canonical, person_facts, prefixed
from experiment.projection import (
    bipolarize,
    make_projection,
    project,
    random_role,
)
from experiment.readout import (
    bind,
    block_score,
    bundle,
    cosine,
    integer_dot,
    rank_records,
    unit,
)


def test_binding_inverse_and_probe_expansion():
    r = torch.tensor([1, -1, 1, -1])
    s = torch.tensor([-1, -1, 1, 1])
    x = torch.tensor([1, 1, -1, -1])
    y = torch.tensor([1, -1, -1, 1])
    record = bundle([bind(r, x), bind(s, y)])
    assert torch.equal(bind(r, bind(r, x)), x)
    assert torch.equal(bind(r, record), x + bind(r, bind(s, y)))
    assert cosine(bind(r, record), y) == pytest.approx(cosine(record, bind(r, y)))
    assert torch.equal(record - bind(r, x), bind(s, y))


def test_integer_bundle_does_not_sign_or_overflow_int8():
    result = bundle([torch.ones(4, dtype=torch.int8)] * 200)
    assert result.dtype == torch.int32
    assert result.tolist() == [200] * 4
    assert integer_dot(result, result) == 160000


def test_contributions_share_record_denominator():
    terms = [torch.tensor([1, 1, -1, 1]), torch.tensor([1, -1, 1, 1])]
    record, query = bundle(terms), torch.tensor([1, 1, 1, -1])
    denominator = float(
        torch.linalg.vector_norm(record.double()) * torch.linalg.vector_norm(query.double())
    )
    assert sum(integer_dot(t, query) / denominator for t in terms) == pytest.approx(
        cosine(record, query)
    )


def test_l2_scale_and_whole_record_sign_are_different():
    record = torch.tensor([3, 1, -1, -3], dtype=torch.int32)
    term = torch.tensor([1, 1, 1, -1])
    residual = record - term
    length = torch.linalg.vector_norm(record.double())
    assert torch.allclose(length * unit(record) - term, residual.double())
    assert torch.allclose(unit(unit(record) - term / length), unit(residual))
    assert not torch.allclose(unit(unit(record) - term), unit(residual))
    signed = torchhd.normalize(record.as_subclass(torchhd.MAPTensor))
    assert signed.tolist() == [1, 1, -1, -1]
    assert not torch.allclose(signed.double(), unit(record))


def test_distinct_inventories_have_same_sum():
    a, b = torch.tensor([1, 1, 1, 1]), torch.tensor([1, -1, -1, -1])
    c, d = torch.tensor([1, 1, -1, -1]), torch.tensor([1, -1, 1, 1])
    assert torch.equal(a + b, c + d)
    assert not torch.equal(a, c) and not torch.equal(a, d)


def test_projection_replay_and_independent_namespaces():
    first = make_projection(3, 64, 2026, ("interest", "eye_color"))
    second = make_projection(3, 64, 2026, ("interest", "eye_color"))
    assert torch.equal(first["matrix"], second["matrix"])
    assert set(first["matrix"].flatten().tolist()) == {-1, 1}
    assert not torch.equal(first["roles"]["interest"], first["roles"]["eye_color"])
    assert not torch.equal(first["roles"]["interest"], random_role(64, 2027, "interest"))
    linear, signed = project(torch.tensor([1.0, 2.0, 3.0]), first)
    expected = unit(torch.tensor([1.0, 2.0, 3.0])) @ first["matrix"].double() / math.sqrt(64)
    assert torch.equal(linear, expected)
    assert torch.equal(signed, project(torch.tensor([2.0, 4.0, 6.0]), second)[1])


def test_rademacher_scaling_on_complete_two_dimensional_basis():
    matrix = torch.tensor([[1, 1, -1, -1], [1, -1, 1, -1]])
    basis = {"matrix": matrix, "projection_ties": torch.ones(4, dtype=torch.int32)}
    linear, _ = project(torch.tensor([3.0, 4.0]), basis)
    assert float(torch.linalg.vector_norm(linear)) == pytest.approx(1.0)


def test_explicit_sign_ties():
    assert bipolarize(
        torch.tensor([0.0, 0.0, 2.0, -3.0]), torch.tensor([-1, 1, -1, 1])
    ).tolist() == [-1, 1, 1, -1]


@pytest.mark.parametrize(
    "bad",
    [
        torch.zeros(3),
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        torch.tensor([]),
        torch.tensor(1.0),
    ],
)
def test_invalid_cosine_vectors(bad):
    with pytest.raises(ValueError):
        unit(bad)


@pytest.mark.parametrize(
    "role,value",
    [
        (torch.tensor([0, 1]), torch.ones(2)),
        (torch.ones(2), torch.ones(3)),
        (torch.ones(2), torch.tensor([1.0, 0.5])),
    ],
)
def test_invalid_binding(role, value):
    with pytest.raises(ValueError):
        bind(role, value)


def test_rank_reports_ties_and_keeps_all_records():
    records = {
        "b": torch.tensor([1.0, 0.0]),
        "a": torch.tensor([1.0, 0.0]),
        "c": torch.tensor([-1.0, 0.0]),
    }
    rows = rank_records(records, torch.tensor([1.0, 0.0]))
    assert [r["person_id"] for r in rows] == ["a", "b", "c"]
    assert [r["rank"] for r in rows] == [1, 1, 3]
    assert rows[0]["tied"] and rows[0]["margin"] == 0


def test_block_reference_retains_correlated_terms():
    facts = [
        {"role": "interest", "input": "a"},
        {"role": "interest", "input": "b"},
        {"role": "eye", "input": "c"},
    ]
    values = {
        "a": torch.tensor([1.0, 0.0]),
        "b": torch.tensor([1.0, 0.0]),
        "c": torch.tensor([0.0, 1.0]),
    }
    assert block_score(
        facts, values, "interest", values["a"], ("interest", "eye")
    ) == pytest.approx(2 / math.sqrt(5))


def test_canonical_inputs_and_prefixes():
    assert canonical("  ＪＡＺＺ  \n Music ") == "jazz music"
    assert prefixed("Jazz", "document") != prefixed("Jazz", "query")
    facts = person_facts({"id": "a", "age": 34, "eye_color": "Brown", "interests": ["Tea", "Jazz"]})
    assert [f["text"] for f in facts] == ["34 years old", "brown eyes", "jazz", "tea"]
    with pytest.raises(ValueError):
        person_facts({"id": "a", "interests": ["Jazz", " JAZZ "]})


def test_sign_loss_has_a_constructive_counterexample():
    first, second, fact = (
        torch.tensor([3, 1]),
        torch.tensor([1, 3]),
        torch.tensor([1, 1]),
    )
    ties = torch.tensor([-1, -1])
    assert torch.equal(bipolarize(first, ties), bipolarize(second, ties))
    assert not torch.equal(bipolarize(first - fact, ties), bipolarize(second - fact, ties))


def test_missing_fields_and_invalid_query_role():
    from experiment.encoding import query_input

    assert len(person_facts({"id": "a", "interests": ["tea"]})) == 1
    with pytest.raises(ValueError, match="Unknown query role"):
        query_input({"role": "unknown", "text": "tea"})
    with pytest.raises(ValueError, match="width"):
        project(torch.ones(4), make_projection(3, 16, 1, ("interest",)))


def test_absent_role_block_score_is_zero():
    values = {"a": torch.tensor([1.0, 0.0])}
    assert (
        block_score(
            [{"role": "interest", "input": "a"}],
            values,
            "absent",
            values["a"],
            ("interest", "absent"),
        )
        == 0
    )
