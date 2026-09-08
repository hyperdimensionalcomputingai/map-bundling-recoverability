"""Cache the fixed document values and semantic queries using local Ollama."""

from experiment import settings as cfg
from experiment.embeddings import local_version, warm_cache
from experiment.encoding import required_inputs
from experiment.storage import (
    begin_stage,
    finish_stage,
    fixtures,
    write_json,
)


def main():
    manifest = begin_stage("embed", ())
    people, queries = fixtures()
    inputs = required_inputs(people, queries)
    result = warm_cache(cfg.RUN_DIR / "embeddings.parquet", inputs)
    # A complete cache can be replayed with the local service stopped.
    if not result["cache_only"]:
        manifest["ollama_version"] = local_version()
        write_json(cfg.RUN_DIR / "manifest.json", manifest)
    finish_stage("embed", [cfg.RUN_DIR / "embeddings.parquet"], details=result)
    print(f"Cached {len(inputs)} distinct prefixed inputs ({result['new_inputs']} new).")


if __name__ == "__main__":
    main()
