"""Remove the stored jazz contribution and contrast related-query subtraction."""

import torch

from experiment import settings as cfg
from experiment.embeddings import cached_vectors
from experiment.encoding import (
    encode_person,
    prefixed,
    query_input,
    required_inputs,
)
from experiment.projection import bipolarize, project
from experiment.readout import bind, cosine, integer_dot, unit
from experiment.storage import (
    begin_stage,
    finish_stage,
    fixtures,
    load_basis,
    load_records,
    write_json,
    write_table,
)


def main():
    begin_stage("subtract", ("encode", "search"))
    people, queries = fixtures()
    basis, records = load_basis(), load_records()
    embeddings = cached_vectors(
        cfg.RUN_DIR / "embeddings.parquet", required_inputs(people, queries)
    )
    values = {text: project(vector, basis)[1] for text, vector in embeddings.items()}
    maya = next(p for p in people if p["id"] == "maya")
    without_jazz = {**maya, "interests": [v for v in maya["interests"] if v != "jazz"]}
    residual, residual_facts, residual_bound = encode_person(without_jazz, values, basis["roles"])
    original = records["maya"]
    interest = basis["roles"]["interest"]
    jazz = bind(interest, values[prefixed("jazz", "document")])
    lounge = bind(interest, values[prefixed("lounge music", "query")])
    jazz_query = bind(interest, values[prefixed("jazz", "query")])
    assert torch.equal(original - jazz, residual)
    assert torch.equal(residual + jazz, original)

    length = torch.linalg.vector_norm(original.double())
    normalized = unit(original)
    signed_residual = bipolarize(residual, basis["record_ties"])
    signed_original = bipolarize(original, basis["record_ties"])
    variants = {
        "original": (original, original),
        "raw_remove_stored_jazz": (original - jazz, residual),
        "raw_readd_stored_jazz": (residual + jazz, original),
        "restore_scale_then_remove": (length * normalized - jazz, residual),
        "scaled_unit_removal_direction": (
            unit(normalized - jazz / length),
            unit(residual),
        ),
        "naive_unit_removal_direction": (unit(normalized - jazz), unit(residual)),
        "remove_lounge_query": (original - lounge, residual),
        "remove_jazz_query": (original - jazz_query, residual),
        "signed_naive_removal": (
            bipolarize(signed_original - jazz, basis["record_ties"]),
            signed_residual,
        ),
    }
    checks, scores, residual_terms = [], [], []
    music_queries = [q for q in queries if q["id"] in ("jazz", "music", "lounge_music", "r_and_b")]
    for name, (actual, expected) in variants.items():
        error = float((actual.double() - expected.double()).abs().max())
        checks.append(
            {
                "operation": name,
                "exact": torch.equal(actual, expected),
                "within_1e_12": error <= 1e-12,
                "max_coordinate_error": error,
                "expected": "original"
                if name in ("original", "raw_readd_stored_jazz")
                else "unit_residual"
                if "direction" in name
                else "signed_residual"
                if name == "signed_naive_removal"
                else "raw_residual",
                "residual_zero_votes": int((residual == 0).sum()),
            }
        )
        for query in music_queries:
            bound_query = bind(interest, values[query_input(query)])
            scores.append(
                {
                    "operation": name,
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "score": cosine(actual, bound_query),
                }
            )
    assert all(
        row["within_1e_12"]
        for row in checks
        if row["operation"]
        in (
            "raw_remove_stored_jazz",
            "raw_readd_stored_jazz",
            "restore_scale_then_remove",
            "scaled_unit_removal_direction",
        )
    )
    for query in music_queries:
        bound_query = bind(interest, values[query_input(query)])
        denominator = float(
            torch.linalg.vector_norm(residual.double())
            * torch.linalg.vector_norm(bound_query.double())
        )
        total = 0.0
        for fact, term in zip(residual_facts, residual_bound, strict=True):
            total += integer_dot(term, bound_query) / denominator
            residual_terms.append(
                {
                    "query_id": query["id"],
                    "fact_id": fact["fact_id"],
                    "fact_role": fact["role"],
                    "fact_text": fact["text"],
                    "dot": integer_dot(term, bound_query),
                    "denominator": denominator,
                    "contribution": integer_dot(term, bound_query) / denominator,
                }
            )
        assert abs(total - cosine(residual, bound_query)) <= 1e-12

    a, b = torch.tensor([1, 1, 1, 1]), torch.tensor([1, -1, -1, -1])
    c, d = torch.tensor([1, 1, -1, -1]), torch.tensor([1, -1, 1, 1])
    assert torch.equal(a + b, c + d)
    ambiguity = cfg.RUN_DIR / "ambiguity.json"
    write_json(
        ambiguity,
        {
            "a": a.tolist(),
            "b": b.tolist(),
            "c": c.tolist(),
            "d": d.tolist(),
            "shared_sum": (a + b).tolist(),
            "inventories_distinct": True,
        },
    )
    outputs = [
        write_table("removal_checks", checks),
        write_table("removal_scores", scores),
        write_table("removal_contributions", residual_terms),
        ambiguity,
    ]
    finish_stage("subtract", outputs)
    print("Stored jazz removed and re-added exactly. Semantic-query edits reported separately.")


if __name__ == "__main__":
    main()
