# MAP bundling recoverability

A small, reproducible MAP experiment using two people and locally generated semantic embeddings. It follows the research contract in [PLAN.md](PLAN.md). The scripts expose the arithmetic so you can follow a role probe, inspect crosstalk, rank records, and distinguish removing a known contribution from subtracting a related query.

## Run the example

Clone the repository and install the locked Python environment:

```sh
git clone git@github.com:hyperdimensionalcomputingai/map-bundling-recoverability.git
cd map-bundling-recoverability
uv sync --frozen
```

Use Python 3.13 or later and a local installation of [Ollama](https://ollama.com/). Start your local Ollama server with the existing `nomic-embed-text:latest` model available. The expected model digest is recorded in `src/experiment/settings.py`; the experiment refuses an unexpected model rather than silently substituting one.

For a new machine, obtain `nomic-embed-text` with `ollama pull nomic-embed-text` and start Ollama before running the embedding step. The `latest` tag is mutable: the scripts verify the pinned digest, and a different model requires a deliberately separate run. The original run used Ollama 0.33.3. Generated caches are not included in the repository, so the first run needs the matching model; offline replay becomes available after that run.

Run these scripts in order:

```sh
uv run src/01_embed.py
uv run src/02_encode.py
uv run src/03_probe.py
uv run src/04_search.py
uv run src/05_subtract.py
uv run src/06_report.py
```

Open `artifacts/demo/REPORT.md` for the measured results, figures, checks, and linked Parquet evidence. Only the first step needs Ollama. After its cache is populated, all six steps can replay offline. Rerunning a step invalidates downstream completion markers, so rerun the remaining steps before treating a report as current.

To try another phrase, edit `ROLE` and `QUERY_TEXT` near the top of `src/07_try_query.py`, then run:

```sh
uv run src/07_try_query.py
```

This embeds only an uncached query, reuses the frozen projection and records, and writes separate exploratory results under `artifacts/demo/queries/`. It does not rebuild the records or change the fixed experiment inputs. There are no command-line arguments or console entrypoints.

## What to inspect

| Script | Evidence |
| --- | --- |
| `01_embed.py` | Exact prefixed input cache and local model identity |
| `02_encode.py` | Shared Rademacher projection, raw MAP sums, native/linear/signed geometry |
| `03_probe.py` | Exact isolated binding inversion, signal plus crosstalk, per-fact score contributions |
| `04_search.py` | Record ranking, exact Torch/LanceDB agreement, role controls, negative-query contrast |
| `05_subtract.py` | Exact stored-vector removal, related-query edits, scale and sign loss |
| `06_report.py` | Technical report and three figures from the saved tables |

The embedding model supplies semantics; the projection and MAP algebra organize them. A query such as “lounge music” is not assumed to equal the stored “jazz” vector. Ranking always returns a best record, including for an unrelated query, so a winner alone is not evidence of a good match. The report preserves scores and margins, including disappointing outcomes.

Bundling has no unique general inverse. Keeping raw integer sums lets us subtract a *known exact contribution*; it does not recover an unknown inventory. Role unbinding produces a noisy probe. Similarity scores measure its alignment with a query without reconstructing all original facts.

## Design and scope

`data/` contains the fixed readable fixtures. Numbered scripts live directly in `src/`; reusable functions live in `src/experiment/`. Settings are plain constants. There is one seeded 768-to-4096 Rademacher matrix, independent bipolar role vectors, and explicit deterministic handling of zero sign votes. The linear projection divides by the square root of the output dimension; the later sign step is measured separately.

Vector arithmetic uses Torch and TorchHD. Polars handles tables, PyArrow defines vector storage types, and LanceDB provides exact cosine retrieval. Project code does not use NumPy or Pandas. Raw sums are int32, dot products are int64, analytical cosines are float64, and stored search vectors are L2-normalized float32. TorchHD MAP normalization is a sign operation and is not used as L2 normalization.

This example uses exact search over two records. It does **not** establish large-dataset scalability or semantic robustness under growing interference. A larger deployment can index role-bound queries against record vectors with approximate nearest-neighbor search, with recall and memory measured separately. The demo deliberately keeps the full-table parity audit and exact reference; these are teaching checks, not a production query path. Capacity and interference sweeps remain a future experiment.

Generated artifacts and local databases are ignored by Git. The manifest records model, fixtures, seed, package versions, source hashes, output hashes, and stage completion. A changed experiment contract or environment requires a new `RUN_DIR` in settings. Keep `uv.lock` with the code. Do not mix artifacts from different encoders.

## Verify

```sh
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

Tests use small exact vectors and mocked model responses. They check algebraic identities, projection replay, tie handling, invalid inputs, cache identity, typed storage, and stale artifact detection without asserting a desired semantic ranking. The fixed scripts also check measured contribution sums, subtraction identities, and Torch/LanceDB parity during the real run.
