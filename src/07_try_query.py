"""Edit the role/text below and query the existing records without rebuilding."""

ROLE = "interest"
QUERY_TEXT = "lounge music"

from experiment import settings as cfg
from experiment.embeddings import cached_vectors, read_cache, warm_cache
from experiment.encoding import query_input
from experiment.projection import project
from experiment.readout import bind, rank_records
from experiment.storage import (
    digest,
    file_hash,
    lance_scores,
    load_basis,
    load_records,
    require_stage,
    write_json,
)


def main():
    manifest = require_stage("encode")
    text = query_input({"role": ROLE, "text": QUERY_TEXT})
    cache_path = cfg.RUN_DIR / "embeddings.parquet"
    if text not in read_cache(cache_path):
        cache_path = cfg.RUN_DIR / "queries" / "embeddings.parquet"
        warm_cache(cache_path, [text])
    embedding = cached_vectors(cache_path, [text])[text]
    basis = load_basis()
    _, value = project(embedding, basis)
    query = bind(basis["roles"][ROLE], value)
    records = load_records()
    ranked = rank_records(records, query)
    database_scores = lance_scores(query)
    for row in ranked:
        assert abs(row["score"] - database_scores[row["person_id"]]) <= 1e-5
        print(f"{row['person_id']}: cosine={row['score']:.6f}, rank={row['rank']}")
    path = cfg.RUN_DIR / "queries" / f"{digest([ROLE, text])[:16]}.json"
    write_json(
        path,
        {
            "role": ROLE,
            "text": QUERY_TEXT,
            "input": text,
            "encoder_id": manifest["encoder_id"],
            "model_digest": cfg.MODEL_DIGEST,
            "projection_hash": file_hash(cfg.RUN_DIR / "projection.pt"),
            "records_hash": file_hash(cfg.RUN_DIR / "tables" / "records.parquet"),
            "results": ranked,
        },
    )
    print(f"Saved query evidence to {path}")


if __name__ == "__main__":
    main()
