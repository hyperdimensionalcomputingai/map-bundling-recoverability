# MAP bundling recoverability

This repository explores how useful information remains accessible when several facts are combined into one hypervector. It uses Multiply-Add-Permute (MAP), where binding multiplies coordinates element by element and bundling adds the resulting fact hypervectors.

Binding an isolated value with a known bipolar role is exactly reversible: applying the same role again recovers the value. A bundle behaves differently. Probing a role exposes its associated values within a mixture of contributions from the other facts. We can compare that mixture with a query, but the bundle does not provide a general inverse that lists its unknown ingredients.

## What the experiments explore

Two fictional people provide a small example that can be inspected directly:

- **Maya:** age 34, brown eyes, interests in tea, climbing, and jazz.
- **Nina:** age 33, blue eyes, interests in tea, cooking, and photography.

Each person contributes five role-value facts to one additive hypervector. Their names identify the records and are not encoded. The experiments ask:

- **What survives a role probe?** Expand the result into associated values and crosstalk, then measure each fact's contribution to a query score.
- **Why can useful information stand out?** Examine how other roles can contribute little to a similarity score even though their hypervectors remain in the mixture.
- **Can a partial semantic cue retrieve a complete record?** Compare queries such as “lounge music” with both people without first converting the phrase into a stored label such as “jazz.”
- **Can queries express preferences?** Combine a positive tea cue with a negative brown-eyes cue and inspect how the ranking changes.
- **What can be removed exactly?** Subtract the stored jazz contribution, compare with a fresh encoding without jazz, and contrast this with subtracting a related query.

The embedding model supplies semantic relationships. A shared projection maps those embeddings into MAP space; binding makes facts addressable by role, and bundling combines their evidence. A high score can reflect several related interests, so matching Maya for “lounge music” does not mean the system has decoded the specific fact “jazz.”

## Setup

You need Git, uv, Python 3.13 or later, and a local Ollama installation.

```sh
git clone git@github.com:hyperdimensionalcomputingai/map-bundling-recoverability.git
cd map-bundling-recoverability
uv sync --frozen
ollama pull nomic-embed-text
```

Start Ollama before embedding. If it is not already running, run `ollama serve` in a separate terminal. The experiment connects to `http://127.0.0.1:11434` and expects:

- Model: `nomic-embed-text:latest`
- Model digest: `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f`
- Embedding width: 768 coordinates

The original run used Ollama 0.33.3. Because `latest` is mutable, pulling the model does not guarantee the expected digest. The scripts check its identity and reject a different model; using another model requires deliberately updating the settings and choosing a separate run directory.

## Run the walkthrough

Run the scripts in order from the repository root:

```sh
uv run src/01_embed.py
uv run src/02_encode.py
uv run src/03_probe.py
uv run src/04_search.py
uv run src/05_subtract.py
uv run src/06_report.py
```

- `01_embed.py` caches the exact stored-value and query inputs, with model identity checks.
- `02_encode.py` creates the projection and person hypervectors, measuring similarity before projection, after projection, and after signing.
- `03_probe.py` checks isolated binding inversion and exposes the contributions within full-record probes.
- `04_search.py` ranks records, checks Torch against exact LanceDB search, and evaluates role controls and negative evidence.
- `05_subtract.py` compares known-fact removal with semantic-query subtraction and checks the effects of normalization and signing.
- `06_report.py` writes the measured results, takeaways, and three figures to `artifacts/demo/REPORT.md`.

Evidence tables are saved as Parquet files alongside the report. Only the embedding step needs Ollama. Caches are not shipped with the repository, so the first run needs the matching model; once the cache is populated, the entire sequence can replay offline. Rerunning a stage invalidates downstream completion markers, so rerun the remaining stages before treating the report as current.

## Try your own query

Edit `ROLE` and `QUERY_TEXT` at the top of `src/07_try_query.py`. Choose `age`, `eye_color`, or `interest` as the role, then run:

```sh
uv run src/07_try_query.py
```

An uncached phrase needs Ollama. The script reuses the saved projection and person hypervectors, prints both records' scores and ranks, and saves separate query evidence under `artifacts/demo/queries/`.

## Representation and reproducibility

The implementation keeps its choices visible:

- Stored text uses the `search_document:` prefix; query text uses `search_query:`. Identical wording in those two modes need not produce identical embeddings.
- One seeded Rademacher matrix maps 768-dimensional unit embeddings to 4,096 coordinates, with linear scaling by `1/sqrt(4096)` before signing. Roles are independently generated bipolar hypervectors; the master seed is 2026.
- Individual projected values are signed to obtain bipolar MAP hypervectors. Complete records retain their integer sums, preserving the information needed to subtract known contributions.
- Raw sums use int32, dot products use int64, analytical cosines use float64, and stored search hypervectors use L2-normalized float32 coordinates. MAP signing and L2 normalization are separate operations.
- Torch and TorchHD handle arithmetic; Polars, PyArrow, and LanceDB handle tables, typed storage, and exact retrieval. Maintained code uses neither NumPy nor Pandas.

Numbered scripts live directly in `src/`, shared functions in `src/experiment/`, and fixed inputs in `data/`. Configuration uses plain constants in `src/experiment/settings.py`; there are no command-line arguments or dispatchers.

Generated artifacts and local databases are ignored by Git. Each run records model identity, input hashes, seeds, dependency versions, source and output hashes, and stage completion. A changed encoding configuration or environment requires a new `RUN_DIR`. Keep the locked environment and each run's artifacts together rather than mixing hypervectors from different encoders.

## Interpreting the results

Similarity and ranking recover useful evidence, not an exact symbolic inventory. An argmax selects the highest-scoring record even for an unrelated query, and cosine is not a confidence probability. The report includes scores, margins, and controls so those distinctions remain visible.

Subtracting an exact stored contribution from the raw sum is an exact update. Subtracting a related phrase changes semantic alignment instead. Negative query terms express preferences; guaranteed exclusion requires checking the source fields.

This is a two-record teaching example using exhaustive search. It does not measure production latency, approximate-search recall, or the number of facts a bundle can reliably hold. Capacity under increasing interference remains a separate research question.

## Verify

```sh
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest
```

The tests use small exact hypervectors and mocked model responses to check algebra, projection replay, cache identity, typed storage, and invalid inputs. They do not require a running model or assert preferred semantic rankings. The walkthrough also checks contribution sums, exact subtraction identities, and Torch/LanceDB score agreement during execution.
