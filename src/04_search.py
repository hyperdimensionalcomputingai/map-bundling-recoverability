"""Rank people with partial semantic cues, then compose a negative query."""

import math

import polars as pl
import torch

from experiment import settings as cfg
from experiment.embeddings import cached_vectors
from experiment.encoding import (
    person_facts,
    prefixed,
    query_input,
    required_inputs,
)
from experiment.projection import project, random_role
from experiment.readout import (
    bind,
    block_score,
    cosine,
    integer_dot,
    rank_records,
)
from experiment.storage import (
    begin_stage,
    finish_stage,
    fixtures,
    lance_scores,
    load_basis,
    load_records,
    read_table,
    write_table,
)


def main():
    begin_stage("search", ("encode", "probe"))
    people, queries = fixtures()
    basis, records = load_basis(), load_records()
    embeddings = cached_vectors(
        cfg.RUN_DIR / "embeddings.parquet", required_inputs(people, queries)
    )
    values = {text: project(vector, basis)[1] for text, vector in embeddings.items()}
    controls = [
        {
            "id": "lounge_wrong_role",
            "role": "eye_color",
            "text": "lounge music",
            "scenario": "wrong_role",
        },
        {
            "id": "lounge_absent_role",
            "role": "absent",
            "text": "lounge music",
            "scenario": "absent_role",
        },
    ]
    roles = {
        **basis["roles"],
        "absent": random_role(cfg.DIMENSIONS, cfg.SEED, "absent"),
    }
    evaluation_queries = [{**q, "scenario": "semantic"} for q in queries] + controls
    results = []
    for query in evaluation_queries:
        # Controls reuse the same prefixed phrase; only the supplied role changes.
        text = prefixed(query["text"], "query")
        value = values[text]
        bound_query = bind(roles[query["role"]], value)
        database_scores = lance_scores(bound_query)
        for ranked in rank_records(records, bound_query):
            person_id = ranked["person_id"]
            person = next(p for p in people if p["id"] == person_id)
            facts = person_facts(person)
            slots = cfg.ROLES + ("absent",)
            native = block_score(facts, embeddings, query["role"], embeddings[text], slots)
            projected = block_score(facts, values, query["role"], value, slots)
            probe_score = cosine(bind(roles[query["role"]], records[person_id]), value)
            assert abs(probe_score - ranked["score"]) <= 1e-12
            error = abs(database_scores[person_id] - ranked["score"])
            assert error <= 1e-5, f"LanceDB score mismatch for {person_id}: {error}"
            results.append(
                {
                    **ranked,
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "role": query["role"],
                    "scenario": query["scenario"],
                    "query_input": text,
                    "native_block_score": native,
                    "bipolar_block_score": projected,
                    "lance_score": database_scores[person_id],
                    "lance_error": error,
                }
            )

    query_by_id = {q["id"]: values[query_input(q)] for q in queries}
    tea = bind(roles["interest"], query_by_id["tea"])
    brown = bind(roles["eye_color"], query_by_id["brown_eyes"])
    negative_rows, negative_terms = [], []
    fact_rows = read_table("facts").to_dicts()
    for name, query in (("tea", tea), ("tea_minus_brown", tea - brown)):
        database_scores = lance_scores(query)
        for ranked in rank_records(records, query):
            person_id = ranked["person_id"]
            record = records[person_id]
            denominator = math.sqrt(integer_dot(record, record) * integer_dot(query, query))
            total = 0.0
            for fact in (f for f in fact_rows if f["person_id"] == person_id):
                term = torch.tensor(fact["bound"], dtype=torch.int32)
                contribution = integer_dot(term, query) / denominator
                total += contribution
                negative_terms.append(
                    {
                        "query_id": name,
                        "person_id": person_id,
                        "fact_id": fact["fact_id"],
                        "fact_text": fact["text"],
                        "dot": integer_dot(term, query),
                        "denominator": denominator,
                        "contribution": contribution,
                    }
                )
            assert abs(total - ranked["score"]) <= 1e-12
            assert abs(database_scores[person_id] - ranked["score"]) <= 1e-5
            negative_rows.append(
                {
                    **ranked,
                    "query_id": name,
                    "raw_dot": integer_dot(record, query),
                    "record_norm": math.sqrt(integer_dot(record, record)),
                    "query_norm": math.sqrt(integer_dot(query, query)),
                    "lance_score": database_scores[person_id],
                }
            )

    # This is deliberately a separate literal predicate on source metadata.
    source_filter = pl.DataFrame(people).select(
        pl.col("id").alias("person_id"),
        (pl.col("interests").list.contains("tea") & (pl.col("eye_color") != "brown")).alias(
            "included"
        ),
    )
    outputs = [
        write_table("search_scores", results),
        write_table("negative_scores", negative_rows),
        write_table("negative_contributions", negative_terms),
        write_table("source_filter", source_filter.to_dicts()),
    ]
    finish_stage("search", outputs)
    print(
        read_table("search_scores")
        .filter(pl.col("query_id").is_in(["music", "lounge_music", "r_and_b"]))
        .select("query_text", "person_id", "score", "rank")
    )


if __name__ == "__main__":
    main()
