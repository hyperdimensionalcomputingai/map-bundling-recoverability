"""Make the shared projection, encode five facts per person, and inspect geometry."""

import torch

from experiment import settings as cfg
from experiment.embeddings import cached_vectors
from experiment.encoding import (
    encode_person,
    person_facts,
    query_input,
    required_inputs,
)
from experiment.projection import make_projection, project
from experiment.readout import cosine, unit
from experiment.storage import (
    begin_stage,
    facts_schema,
    finish_stage,
    fixtures,
    records_schema,
    store_lance_records,
    tensor_hash,
    write_table,
)


def main():
    manifest = begin_stage("encode", ("embed",))
    people, queries = fixtures()
    embeddings = cached_vectors(
        cfg.RUN_DIR / "embeddings.parquet", required_inputs(people, queries)
    )
    basis = make_projection(cfg.EMBEDDING_DIM, cfg.DIMENSIONS, cfg.SEED, cfg.ROLES)
    torch.save(basis, cfg.RUN_DIR / "projection.pt")
    linear, values = {}, {}
    for text, embedding in embeddings.items():
        linear[text], values[text] = project(embedding, basis)

    records, fact_rows, coordinates = [], [], []
    for person in people:
        raw_sum, facts, bound = encode_person(person, values, basis["roles"])
        records.append(
            {
                "person_id": person["id"],
                "encoder_id": manifest["encoder_id"],
                "fact_count": len(facts),
                "raw_norm": float(torch.linalg.vector_norm(raw_sum.double())),
                "raw_sum": raw_sum.tolist(),
                "vector": unit(raw_sum).float().tolist(),
            }
        )
        for fact, vector in zip(facts, bound, strict=True):
            fact_rows.append(
                {
                    **fact,
                    "value": values[fact["input"]].tolist(),
                    "bound": vector.tolist(),
                }
            )
        # First 16 coordinates of the actual 4096-D run, not a separate experiment.
        for index in range(16):
            for fact, vector in zip(facts, bound, strict=True):
                coordinates.append(
                    {
                        "person_id": person["id"],
                        "coordinate": index,
                        "fact_id": fact["fact_id"],
                        "role": fact["role"],
                        "role_coordinate": int(basis["roles"][fact["role"]][index]),
                        "value_coordinate": int(values[fact["input"]][index]),
                        "bound_coordinate": int(vector[index]),
                        "raw_sum": int(raw_sum[index]),
                    }
                )

    geometry = []
    distinct = {(fact["role"], fact["input"]) for p in people for fact in person_facts(p)}
    for query in queries:
        text = query_input(query)
        for role, document in sorted(distinct):
            if role != query["role"]:
                continue
            geometry.append(
                {
                    "query_id": query["id"],
                    "query_text": query["text"],
                    "role": role,
                    "document_input": document,
                    "query_input": text,
                    "native_cosine": cosine(embeddings[text], embeddings[document]),
                    "linear_cosine": cosine(linear[text], linear[document]),
                    "bipolar_cosine": cosine(values[text], values[document]),
                    "query_native_norm": float(torch.linalg.vector_norm(embeddings[text])),
                    "document_native_norm": float(torch.linalg.vector_norm(embeddings[document])),
                    "query_linear_norm": float(torch.linalg.vector_norm(linear[text])),
                    "document_linear_norm": float(torch.linalg.vector_norm(linear[document])),
                    "query_sign_zeros": int((linear[text] == 0).sum()),
                    "document_sign_zeros": int((linear[document] == 0).sum()),
                }
            )
    outputs = [
        cfg.RUN_DIR / "projection.pt",
        write_table("records", records, schema=records_schema(cfg.DIMENSIONS)),
        write_table("facts", fact_rows, schema=facts_schema(cfg.DIMENSIONS)),
        write_table("geometry", geometry),
        write_table("coordinate_excerpt", coordinates),
    ]
    store_lance_records()
    hashes = {name: tensor_hash(value) for name, value in basis.items() if name != "roles"}
    hashes.update({f"role:{name}": tensor_hash(value) for name, value in basis["roles"].items()})
    finish_stage("encode", outputs, details={"projection_and_role_hashes": hashes})
    print(f"Encoded {len(records)} records; saved {len(geometry)} geometry comparisons.")


if __name__ == "__main__":
    main()
