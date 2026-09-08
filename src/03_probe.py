"""Show exact role cancellation, surviving semantic evidence, and crosstalk."""

import math

import torch

from experiment import settings as cfg
from experiment.embeddings import cached_vectors
from experiment.encoding import query_input, required_inputs
from experiment.projection import project
from experiment.readout import bind, bundle, cosine, integer_dot
from experiment.storage import (
    begin_stage,
    finish_stage,
    fixtures,
    load_basis,
    load_records,
    read_table,
    write_table,
)


def main():
    begin_stage("probe", ("encode",))
    people, queries = fixtures()
    embeddings = cached_vectors(
        cfg.RUN_DIR / "embeddings.parquet", required_inputs(people, queries)
    )
    basis, records = load_basis(), load_records()
    facts = read_table("facts").to_dicts()
    scores, contributions, checks, coordinates = [], [], [], []

    for fact in facts:
        role = basis["roles"][fact["role"]]
        value = torch.tensor(fact["value"], dtype=torch.int32)
        recovered = bind(role, bind(role, value))
        assert torch.equal(recovered, value)
        checks.append(
            {
                "fact_id": fact["fact_id"],
                "isolated_inversion_exact": True,
                "bound_vs_value": cosine(bind(role, value), value),
                "bound_vs_role": cosine(bind(role, value), role),
            }
        )

    for query in queries:
        _, query_value = project(embeddings[query_input(query)], basis)
        role = basis["roles"][query["role"]]
        for person_id, record in records.items():
            person_terms = [f for f in facts if f["person_id"] == person_id]
            probe = bind(role, record)
            expanded = [
                bind(role, torch.tensor(f["bound"], dtype=torch.int32)) for f in person_terms
            ]
            same = [term for f, term in zip(person_terms, expanded) if f["role"] == query["role"]]
            cross = [term for f, term in zip(person_terms, expanded) if f["role"] != query["role"]]
            signal, residual = bundle(same), bundle(cross)
            assert torch.equal(probe, bundle(expanded))
            assert torch.equal(probe, signal + residual)
            assert integer_dot(probe, probe) == integer_dot(record, record)
            denominator = math.sqrt(
                integer_dot(probe, probe) * integer_dot(query_value, query_value)
            )
            score = cosine(probe, query_value)
            contribution_sum = 0.0
            for fact, term in zip(person_terms, expanded, strict=True):
                dot = integer_dot(term, query_value)
                contribution_sum += dot / denominator
                contributions.append(
                    {
                        "query_id": query["id"],
                        "person_id": person_id,
                        "fact_id": fact["fact_id"],
                        "fact_text": fact["text"],
                        "fact_role": fact["role"],
                        "same_role": fact["role"] == query["role"],
                        "dot": dot,
                        "denominator": denominator,
                        "contribution": dot / denominator,
                    }
                )
            assert abs(contribution_sum - score) <= 1e-12
            bound_score = cosine(record, bind(role, query_value))
            assert abs(bound_score - score) <= 1e-12
            scores.append(
                {
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "role": query["role"],
                    "person_id": person_id,
                    "score": score,
                    "bound_query_score": bound_score,
                    "same_role_score": integer_dot(signal, query_value) / denominator,
                    "cross_role_score": integer_dot(residual, query_value) / denominator,
                    "probe_norm": math.sqrt(integer_dot(probe, probe)),
                    "signal_norm": math.sqrt(integer_dot(signal, signal)),
                    "residual_norm": math.sqrt(integer_dot(residual, residual)),
                    "contribution_error": abs(contribution_sum - score),
                }
            )
            if query["id"] in ("brown_eyes", "lounge_music"):
                for index in range(16):
                    coordinates.append(
                        {
                            "query_id": query["id"],
                            "person_id": person_id,
                            "coordinate": index,
                            "probe": int(probe[index]),
                            "same_role_sum": int(signal[index]),
                            "cross_role_sum": int(residual[index]),
                            "query": int(query_value[index]),
                        }
                    )
    outputs = [
        write_table("probe_scores", scores),
        write_table("contributions", contributions),
        write_table("inversion_checks", checks),
        write_table("probe_excerpt", coordinates),
    ]
    finish_stage("probe", outputs)
    print(f"Verified {len(checks)} isolated inversions and {len(scores)} full-record probes.")


if __name__ == "__main__":
    main()
