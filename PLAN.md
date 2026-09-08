# What can we recover from a bundled hypervector?

Implementation specification for MAP bundling recoverability. Revised 2026-09-07 for semantic queries with local Nomic embeddings.

## Outcome and scope

Build a small MAP demonstration that a reader can follow from person records through semantic value encoding, binding, additive bundling, role probes, partial queries, and known-fact removal. Binding provides reversible associations; bundling superimposes evidence that we can query. Their different recovery behavior serves different purposes and is not an implementation mistake.

The main question is now: **How can a new semantic query match information in a bundled record without enumerating a predefined vocabulary of possible values?** A query about `lounge music` should be able to express a relationship to a stored `jazz` interest if the embedding model captures that relationship. We must measure the example rather than promise its outcome.

This document records the implementation contract, not a blog draft. The fixed walkthrough is now implemented; see [README.md](README.md) for the runnable scripts and the generated [technical report](artifacts/demo/REPORT.md) for measured evidence. Verification on 2026-09-07 passed 32 tests and reproduced all 15 evidence tables exactly during offline replay with embedding network calls blocked. **How well bundling preserves semantic relationships under increasing interference is a future experiment.** Do not implement load, dimension, corpus-size, vocabulary-size, or capacity sweeps here. No ANN benchmark, new large dataset, HRR/BSC implementation, embedding-model comparison, or graph integration is needed.

Use Torch and TorchHD for vector generation, projection, MAP operations, scoring, and reductions. Use Polars for analysis, PyArrow for schemas and interchange, LanceDB for persisted exact retrieval, and `uv` for dependencies. Avoid direct NumPy usage; allow only a minimal documented boundary conversion if necessary. **Do not use Pandas**, including `to_pandas()` or Pandas-backed helpers. Favor readable, maintainable code over cleverness.

Use the existing package under `src/experiment/` for shared helpers, with ordinary teaching scripts as the entrypoints. At initial inspection the scaffold had Python `>=3.13`, an empty README, no declared dependencies, and no lockfile or `.venv`. During implementation, remove the placeholder console-script registration from `[project.scripts]` and its greeting function. Preserve unrelated edits.

## Sources and the change in approach

- [The preceding HRR/MAP study](https://hyperdimensionalcomputing.ai/blog/hrr-map-similarity/) measured encoder-defined similarity and neighbor retrieval in a controlled fixture. This follow-up adds semantic queries and explicit role probing; the previous study did not benchmark unbinding accuracy or establish universal algebra equivalence.
- [Achlioptas, Database-friendly random projections: Johnson-Lindenstrauss with binary coins (2003)](https://www.sciencedirect.com/science/article/pii/S0022000003000254), with an accessible [conference paper](https://users.math.msu.edu/users/iwenmark/Teaching/MTH995/Papers/JL_Database_Subgaussian.pdf), grounds the dense independent ±1 projection and output-dimension scaling. Its linear distance-preservation result does not automatically cover sign quantization.
- [Nomic's model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) documents retrieval task prefixes. [Ollama's embedding API](https://docs.ollama.com/api/embed) accepts batched text and can reject truncation. Pin the actual local model identity rather than inferring its exact upstream revision from a mutable tag.
- [TorchHD's MAP implementation](https://torchhd.readthedocs.io/en/stable/_modules/torchhd/tensors/map.html) supplies multiplication, addition, and bipolar normalization. [LanceDB vector search](https://docs.lancedb.com/search/vector-search) supplies exhaustive cosine retrieval and Arrow output through its Python API.
- [Kanerva (2009), sections 6.1-6.3 and 6.9](https://redwood.berkeley.edu/wp-content/uploads/2018/01/kanerva2009hyperdimensional.pdf) explains cleanup memory, superposition, and holistic records. A dictionary and suitable constraints can make particular sums recoverable; avoid an absolute impossibility claim.

The existing sibling tea encoder uses a shared Rademacher projection followed by signing. Reuse that conceptual convention, while implementing this project's Torch-only arithmetic, task-prefix contract, and stronger cache identity explicitly. Do not copy its NumPy path or silently inherit its preprocessing.

## 1. Fixed data and queries

Check in `data/people.json`:

```json
[
  {"id": "maya", "name": "Maya", "age": 34, "eye_color": "brown", "interests": ["tea", "climbing", "jazz"]},
  {"id": "nina", "name": "Nina", "age": 33, "eye_color": "blue", "interests": ["tea", "cooking", "photography"]}
]
```

Names/IDs are metadata; omit height. Each record has five unit-weight facts. Encode each interest separately, without averaging or normalizing an interest sub-bundle. Interest order is not represented. Canonicalize Unicode and whitespace, lowercase text, and reject duplicate canonical interests. Missing fields contribute no term; do not invent a zero-valued semantic fact.

Use these value-text templates consistently: age `34 years old`, eye color `brown eyes`, interest `jazz`. All values pass through the local embedding pipeline; only role vectors are independent random symbols. Age similarity is whatever the model encodes, not a calibrated numerical-distance representation. Preserve source values for exact output and provenance.

Check in `data/queries.json` with explicit roles and phrases:

| Role | Phrases | Purpose |
| --- | --- | --- |
| `interest` | `jazz`, `music`, `lounge music`, `R&B` | Exact wording plus related concepts supplied by the user |
| `interest` | `tea`, `rock climbing`, `preparing meals`, `taking photographs` | Further literal and semantic cues for both people |
| `interest` | `aircraft maintenance` | A plausibly unrelated control, not an assumed zero score |
| `eye_color` | `brown eyes`, `blue eyes` | Single-valued field probes and negative evidence |
| `age` | `34 years old`, `33 years old` | Complete the scalar-field walkthrough without claiming ordinal fidelity |

These are fixed evaluation prompts, not a decoder vocabulary. The query path also accepts a new phrase at runtime with a known role, embeds it, and scores existing records without re-encoding them. No matching text needs to have been stored. Do not infer an arbitrary natural-language query's role with an LLM; supply the role explicitly to keep this experiment focused.

Do not assign semantic relevance labels by inspecting scores. Report all prompts, including weak or unexpected matches. The two fictional people illustrate a realistic encoding/query pattern; they do not establish production search quality.

## 2. Local embeddings and reproducibility

A local preflight on 2026-09-07 verified:

| Item | Observed value |
| --- | --- |
| Endpoint | `http://127.0.0.1:11434` |
| Ollama version | `0.33.3` |
| Installed tag | `nomic-embed-text:latest` |
| Model digest | `0a109f422b47e3a30ba2b10eca18548e944e8a23073ee3f3e947efcf3c45e59f` |
| Reported family / quantization | `nomic-bert` / `F16` |
| Embedding width | 768, confirmed by a two-input `/api/embed` call |

This only verifies availability and shape, not semantic outcomes or latency. Revalidate the model digest at implementation time. Do not silently substitute a model or pull a newer tag. A changed digest requires a new manifest and embedding cache.

Use `ollama.Client(host=...).embed` with `truncate=False`. Batch unique texts in a stable order. Use `search_document: <value text>` for stored values and `search_query: <query phrase>` for queries. Store the complete prefixed input. The same phrase with different prefixes is a different embedding request; even the query `jazz` need not equal the stored `jazz` vector. Use the stored document-side vector for exact removal and algebraic self-recovery tests.

Convert returned lists directly to CPU `torch.float64`, reject wrong widths/nonfinite or zero vectors, and L2-normalize. Retain the full 768 dimensions; do not add whitening, centering, layer normalization, dimension truncation, or per-field preprocessing that changes the geometry. No remote embedding service is involved.

Cache original returned embeddings in Arrow/Parquet, keyed by model digest, exact prefixed input, API options, and preprocessing version. Record response order and check row counts. Hash cache contents and normalized vectors. Provide an offline replay mode that errors on a missing entry; it must never invent an embedding. Caching is required for reproducibility, not an optional performance optimization. A new runtime phrase can use the local model and extend the cache without altering existing vectors.

## 3. Rademacher projection and MAP representation

Use `d = 768`, `D = 4096`, master seed `2026`. Make one matrix `R` of shape `[d, D]` with independent entries taking ±1 with equal probability. Keep it fixed across all fields, people, document inputs, and query inputs. Derive its CPU generator seed from a versioned SHA-256 key independent of the role generators. Never use Python's process-randomized `hash()`.

For a unit embedding row `e`, define the linear output `z = eR / sqrt(D)`. This scaling preserves squared norm and inner product in expectation over R. It is `sqrt(D)`, not `sqrt(d)`, for this matrix orientation. Keep R's entries genuinely independent; do not force exact column balance, fit the matrix on the fixture, or resample for favorable scores. This is a 768-to-4096 expansion, not dimensionality reduction or new semantic information. The linear map is not generally an orthogonal matrix.

The main MAP value is `h = bipolarize(z)`, retaining ±1 coordinates. This sign step is an explicit modeling choice to obtain bipolar factors, separate from additive bundling. The projection's positive scale does not affect signs, but matters for the linear geometry audit. Signing is nonlinear: neither exact cosine preservation nor the rotationally invariant Gaussian-hyperplane angular formula is an exact guarantee for these Rademacher directions. Measure the actual stages at the fixed configuration.

```python
import math
import torch
import torchhd

# embedding_rows: validated unit float64 CPU tensors, shape [batch, 768].
# projection_generator: dedicated seeded CPU generator.
d, D = 768, 4096
R = 2 * torch.randint(0, 2, (d, D), generator=projection_generator,
                      dtype=torch.int32, device="cpu") - 1
projected = (embedding_rows @ R.to(torch.float64)) / math.sqrt(D)
# A separately seeded, fixed ±1 vector handles exact zeros identically
# for every text. Count zeros; never randomize ties independently per call.
h = torch.where(projected > 0, 1,
                torch.where(projected < 0, -1, projection_tie_vector))
h = h.to(torch.int32).as_subclass(torchhd.MAPTensor)
```

Persist R and the projection tie vector with shapes, dtypes, generator derivation, and content hashes. Repeatable inputs must yield identical signed values with the frozen environment and artifacts. Do not promise bit-identical floating-point projection across hardware or library releases. Use CPU, gradients disabled, and no autocast as the reference path. Do not create GPU or dimension-sweep machinery.

Generate independent bipolar `AGE`, `EYE`, and `INTEREST` roles with `torchhd.MAPTensor.random`, using namespaced seeds distinct from R. Reuse each role consistently. Roles remain ±1 without unit normalization before multiplication. Binding and additive bundling use `torchhd.bind` and `torchhd.multiset`, with plain-Torch identities as test references.

Promote operands to `torch.int32` before bundling; TorchHD multibundling retains the input dtype. Use `torch.int64` for exact integer dots and squared norms, and `torch.float64` for reference cosine. Store raw sums separately from float32 L2-normalized search vectors. In this five-fact fixture, integer ranges are comfortably bounded; still validate dimensions and avoid int8 reductions.

**TorchHD `normalize()` is not L2 normalization for MAP.** It applies a bipolar threshold with its own zero rule. Use an explicit norm division after checking nonzero norms for search vectors. Use our fixed tie rules for sign controls. `inverse()` assumes a bipolar operand; `negative()` negates a known vector and does not discover bundle members.

## 4. What the equations should make visible

Write `H_doc(text)` and `H_query(text)` for the projected bipolar document and query values. For example:

```text
m = AGE ⊗ H_doc("34 years old") + EYE ⊗ H_doc("brown eyes")
  + INTEREST ⊗ H_doc("tea")
  + INTEREST ⊗ H_doc("climbing")
  + INTEREST ⊗ H_doc("jazz")
```

Nina has the corresponding five source facts. The interest probe is exactly:

```text
p = INTEREST ⊗ m
  = H_doc("tea") + H_doc("climbing") + H_doc("jazz") + ξ

ξ = INTEREST ⊗ AGE ⊗ H_doc("34 years old")
  + INTEREST ⊗ EYE ⊗ H_doc("brown eyes")
```

For `h_q = H_query("lounge music")`, score the supplied query directly:

```text
cos(p, h_q) = [Σ_interest values (h_value · h_q) + ξ · h_q]
             / [||p|| ||h_q||]
```

Export each dot-product contribution using this common denominator. Adding individually normalized term cosines would be wrong. Other interests can contribute intentional semantic evidence, incidental semantic similarity, or negative alignment. They are not independent random distractors. Distinguish these same-role contributions from interference due to other roles, and separate both from projection/sign distortion.

For any bipolar role r and nonzero record s and query h, the following identity is exact:

```text
cos(r ⊗ s, h) = cos(s, r ⊗ h)
```

The role flips coordinates, preserves norms, and is self-inverse. Thus probing an existing record and searching complete records with a role-bound semantic cue implement the same score. A probe does not become a readable string, and no operation here decodes an embedding into original text.

### How argmax works in this design

A single semantic query against Maya needs one score, not an argmax over possible labels. To find the best person, compute a score for each stored record and select its ID:

```python
# record_unit_vectors: [record_count, D], stored L2-normalized float tensors.
# role and query_hv: bipolar tensors of shape [D].
query = (role * query_hv).to(torch.float64)
query = query / torch.linalg.vector_norm(query)
scores = record_unit_vectors.to(torch.float64) @ query
winner_index = torch.argmax(scores).item()
winner_person_id = person_ids[winner_index]
```

`max` returns the score; `argmax` returns its position, which maps to a stored person. Top-k returns multiple records. The record collection is the search population, but it is not an enumerated vocabulary of values such as `jazz`, `rock`, or `R&B`. There is no cleanup stage snapping the query to a label.

Show all scores for the two-person fixture. A top result exists even for a poor query; neither argmax nor cosine is a confidence probability. Do not impose a threshold derived for independent random candidates on semantic embeddings. Exact field text, if displayed, comes from source metadata and is labeled as such. A traditional dictionary cleanup example may be mentioned in the report for contrast, but is not a second implementation track.

### Why near orthogonality helps readout

Conditioned on fixed bipolar semantic values a and b, binding one with an independent random role mask makes its dot/D against the other have mean zero and variance `1/D` over the mask. The role mask makes cross-role alignment small on average; it does not erase the residual vector. For matching roles, multiplication cancels and preserves the full value-to-query similarity.

This explains the intended separation: same-role semantic relationships remain available while other roles tend to interfere less. Shared roles produce correlated residual contributions, so their variances cannot automatically be added as if every fact were independent. Related value vectors should not be near-orthogonal. Do not predict semantic scores with the earlier independent-value `1/sqrt(N)` or maximum-distractor formulas.

Demonstrate the actual residual contributions and norms for this fixture. The effect of increasing interference, correlated record populations, or error thresholds belongs to the future study, not a statistical guarantee claimed here.

## 5. Experiments and acceptance evidence

### E1. Trace semantic encoding and isolate projection from signing

Embed the fixed record values and query prompts once. For every query-value pair in the queried role, report native embedding cosine, cosine after the scaled linear projection, and bipolar cosine after signing. Also report native/projected norms and zero-sign counts. Include the user's music phrases and all controls, not just favorable pairs.

This is a baseline correctness/geometry audit, not a projection benchmark or an interference sweep. A changed score or rank must be visible. The source embedding score supplies context for what the model thought; it is not a human relevance label. No claim that signing preserves every ranking is required for implementation completion.

Use a 16-coordinate excerpt of the real D=4096 vectors to illustrate arithmetic, clearly labeled as an excerpt. Do not calculate retrieval scores on that excerpt or introduce a separate toy projection seed.

### E2. Unbind one role, then test an open semantic query

Verify exact isolated inversion `EYE * (EYE * H_doc("brown eyes")) == H_doc("brown eyes")`. Then show that probing Maya's full record also retains the other facts as a residual. Do the analogous interest probe, exposing three value vectors at once, and run all role-appropriate prompts for both people, including age.

For every probe/query pair, export all five fact contributions, their same-role/cross-role classification, their sum, norms, and cosine. Assert that the expansion equals the probe coordinatewise and that contribution scores sum to full cosine within `1e-12`. Verify that same-role multiplication preserves cosine and that probe norm equals record norm.

The output answers whether the supplied concept aligns with the record; it does not enumerate an interest list. No k=3 oracle, candidate vocabulary, missing-label simulation, or dictionary-calibrated threshold is needed.

### E3. Search records using partial semantic cues

Construct `INTEREST * H_query("lounge music")`, rank Maya and Nina, and show exact equality with E2's role-probe scores. Repeat the fixed query list with the specified roles. Explain `argmax` using the actual two-record table, and return both records for inspection rather than presenting top-2 recall as a meaningful accuracy metric.

Persist and query the same record vectors in LanceDB with exact cosine search. Compare scores with Torch. The `07_try_query.py` script must accept a phrase absent from `data/queries.json` through its editable `QUERY_TEXT` constant, without rebuilding record vectors; test this with a mocked deterministic embedding response, not an extra scored semantic benchmark.

For a role-specific control, query the same `lounge music` value under EYE and INTEREST. Show how semantic matching depends on the bound role. For an absent-role control, use a separately keyed role absent from both records. Report results without assuming every wrong-role score is exactly zero.

Use an explicit block-by-role reference for the displayed semantic queries: concatenate the sum of value vectors for each role into separate slots, with the query in its supplied role and zeros elsewhere. Compute this reference both in native embedding space and projected bipolar space. The first exposes the model's aggregate semantic evidence; the second removes cross-role interference while keeping the exact signed values. Compare with the bundled result. The block representation is an audit baseline with explicit role storage, not a replacement claimed to be worse. Do not attribute differences due to sign quantization to bundling.

### E4. Compose positive and negative semantic evidence

Use `t = INTEREST * H_query("tea")`, `b = EYE * H_query("brown eyes")`, and compare the baseline t with `q = t - b`. Report each fact's contribution, raw dot, norms, cosine, and rank for both people. Normalize the complete query once. Both terms are query-side semantic values; the brown-eye penalty need not exactly cancel a stored document-side fact.

Do not promise Nina ranks first or transplant the old idealized 0 and `1/sqrt(10)` scores. Inspect native query/document similarity, projected similarity, and cross-role terms to explain what actually happens. No weight sweep is needed.

Show the distinction from exact exclusion using the source predicate `tea in interests AND eye_color != brown`. It returns Nina for this literal fixture. That predicate is a separate exact question, not an exact reference for all semantic matches. A negative similarity contribution penalizes evidence and may affect related concepts; it is not Boolean NOT.

### E5. Remove a known stored fact without claiming semantic decomposition

Use the exact stored contribution `f_jazz = INTEREST * H_doc("jazz")`. Require `m - f_jazz` to equal a fresh raw encoding of Maya with jazz omitted, coordinate for coordinate. Re-add the fact and recover m exactly. Compare the semantic music-query scores before and after removal; record remaining same-role and cross-role contributions. A nonzero residual similarity does not mean the known subtraction failed.

Contrast this with subtracting `INTEREST * H_query("lounge music")`: that is a semantic counterfactual edit, not exact removal of the jazz fact. It need not equal the re-encoded record. Even the same text with a query prefix is not interchangeable with the cached document vector. Exact deletion requires the contribution, weight, model, and projection originally used.

Compare raw sums, L2-normalized search vectors, and majority-signed record vectors. For `u=m/L`, the correct scaled removal is `L*u - f_jazz` up to floating-point precision, or normalize `u - f_jazz/L` for the residual direction. Signing the entire sum discards margins and generally prevents this update. Use a fixed separately seeded tie vector for zero votes in the even four-fact signed residual and report ties. This sign-of-the-record control differs from signing projected individual values.

Retain an exact, clearly pedagogical ambiguity example:

```text
a = [1,  1,  1,  1]    b = [1, -1, -1, -1]
c = [1,  1, -1, -1]    d = [1, -1,  1,  1]
a + b = c + d = [2, 0, 0, 0]
```

A sum alone cannot tell these two inventories apart. This is not an empirical collision claim for Nomic vectors. Dictionaries or structural constraints can make particular sums recoverable; knowing a contribution also permits its subtraction. Neither capability supplies a universal operation that discovers unknown operands or restores list order.

## 6. Code and storage

**Do not build a CLI.** Use numbered Python teaching scripts runnable directly with `uv run`, without argparse, Click, Typer, subcommands, console-script entrypoints, or a generic runner. Each script should show its experiment steps in readable order, with a normal `main()` function and `if __name__ == "__main__":` guard. Shared helpers handle repeated math and storage; the scripts should not be opaque one-line dispatchers.

Put shared settings such as model identity, D, seeds, and run directory in a small `settings.py`. Put the exploratory query role/text at the top of `07_try_query.py` so readers can edit and rerun it. Avoid a configuration loader or command-line flags. Resolve data and artifact paths relative to the repository, not the invoking shell directory.

```text
data/people.json
data/queries.json
src/
  01_embed.py       # cache fixed values and prompts from local Ollama
  02_encode.py      # E1 projection audit and record construction
  03_probe.py       # E2 isolated inversion, role probes and contributions
  04_search.py      # E3-E4 partial, role-specific and negative queries
  05_subtract.py    # E5 known removal and representation controls
  06_report.py      # assemble saved evidence into tables and figures
  07_try_query.py   # edit ROLE and QUERY_TEXT, then run against stored records
  experiment/
    __init__.py     # package only; no command dispatcher
    settings.py    # explicit shared constants
    embeddings.py  # local Ollama requests and versioned cache
    projection.py  # shared Rademacher matrix, sign step, role generation
    encoding.py    # canonical facts and TorchHD record construction
    readout.py     # role probes, cosine scoring, record ranking
    storage.py     # Arrow schemas and LanceDB round trips
    reporting.py   # Polars summaries and Matplotlib outputs
tests/            # algebra, projection, query, cache and storage contracts
artifacts/<run_id>/
  manifest.json
  embeddings.parquet
  projection.pt
  lancedb/
  tables/*.parquet
  figures/*.png
  REPORT.md
```

Keep runnable teaching scripts directly in `src/` and reusable code in `src/experiment/`. Import helpers through the package, for example `from experiment.encoding import encode_person`; do not import numbered scripts or modify `sys.path`. Add subpackages only if a module grows enough to warrant a clear grouping. Keep fixtures in `data/`, generated outputs in `artifacts/`, and tests in `tests/`. The module list is guidance, not a reason to create empty files or unnecessary layers. Keep scoring free of source-answer metadata; the evaluation layer joins facts and labels after scoring. Canonical source records may be fetched to display exact text, but distinguish that lookup from semantic vector matching.

Add and lock `torch`, `torch-hd` (import `torchhd`), `ollama`, `polars`, `pyarrow`, and `lancedb`; use `matplotlib` for a few static evidence figures and `pytest` for development. Keep the existing Python/build settings unless a verified compatibility issue requires a documented change. Do not introduce a general embedding-provider framework.

Use explicit Arrow types and fixed-size vector lists. Persist at least:

| Artifact | Required contents |
| --- | --- |
| Embedding cache | Exact prefixed text, input hash, model digest, API/preprocessing identity, original 768-vector and norm |
| Projection artifact | Int32 R, tie vectors, roles, dimensions, seed derivation, hashes |
| Records | Person ID, fact count, int32 raw sum, float64 norm, float32 unit vector, encoder identity |
| Query evidence | Query ID, supplied role/text/prefix, person ID, score, rank, margin, ties, source/native and projected block-reference scores |
| Contributions | Query/person IDs, source fact ID, role relation, integer dot, common denominator, contribution score |
| Geometry | Query/value IDs, native and linear cosines, bipolar cosine, norms, zero counts |
| Removal checks | Fact used, representation, coordinate equality/error, before/after scores, norm and sign-tie counts |

Prefer batched CPU tensor `.tolist()` into Arrow and `.to_arrow()` from LanceDB, followed by `pl.from_arrow`. Keep arithmetic in Torch. Store one embedding/projection identity per run; reject mismatches. Never compare vectors produced with different matrices or silently changed model digests.

Use exhaustive cosine search with no ANN index, explicitly bypassing an index if one exists. Convert `_distance` to cosine with `1 - _distance`. Compare with Torch within `1e-5` for float32 storage. For near-tied ranks, report score differences and tied sets within tolerance, then use IDs only for stable display. Do not treat arbitrary tie ordering as a semantic finding.

### Coding practices: clarity before cleverness

Treat this as a small, reader-facing research codebase. A reader should be able to follow one experiment from its input facts to its reported scores without navigating a framework. Prefer the simplest implementation that preserves the mathematical and reproducibility contracts above.

- Write short, focused functions with descriptive names such as `encode_person`, `probe_role`, and `score_candidates`. Use names from the equations where they help connect math to code, and distinguish `raw_sum`, `unit_vector`, and `signed_vector` explicitly.
- Keep the experiment steps visible and sequential: generate atoms, bind facts, sum them, probe, score candidates, then evaluate. Prefer explicit intermediate tensors over dense expressions, nested comprehensions, or clever broadcasting. State expected tensor shapes and dtypes at function boundaries and explain non-obvious dimensions.
- Use ordinary functions and simple data structures by default. Add a dataclass only when a named group of fields improves readability. Avoid class hierarchies, plugin registries, generic experiment engines, custom tensor wrappers, and abstractions for algebras or backends this project does not use.
- Extract helpers for repeated mathematical operations or a clear responsibility. Do not hide a single understandable step behind several layers of delegation merely to remove a few repeated lines. The proposed module layout identifies responsibilities, not a requirement to create empty files or split every helper into its own module.
- Keep arithmetic and candidate scoring free of file I/O and global mutable state. Pass configuration and random generators explicitly; keep persistence and reporting at the edges. Avoid in-place tensor mutations unless their purpose and ownership are obvious, especially for shared role vectors and cached value vectors.
- Add useful type hints to function interfaces and brief docstrings describing inputs, outputs, shapes, and mathematical assumptions. Comments should explain a decision or a subtle invariant, such as dtype promotion or the common cosine denominator, rather than narrate obvious syntax.
- Validate inputs and configuration once at the relevant boundary, with clear errors that identify the offending field or shape. Catch exceptions only when recovery is meaningful or additional context helps; do not silently substitute defaults, skip failed trials, or repeat defensive checks throughout trusted internal code.
- Start with readable CPU Torch code and ordinary loops over experiments or batches. Vectorize straightforward tensor arithmetic, but add optional caches beyond the required embedding cache, concurrency, device-specific paths, or other optimizations only after measuring a real bottleneck. Keep any optimization equivalent to a simple reference and explain the measured reason for it.
- Keep dependencies and configuration small. Use the standard library where it suffices and the existing Torch/TorchHD, Polars, Arrow, and LanceDB APIs directly. Add an option only for an experiment choice that is actually exercised; do not build speculative extensibility.
- Test mathematical identities, decision rules, storage boundaries, and meaningful failure cases using small, inspectable fixtures. Prefer direct assertions over elaborate mocking or tests that merely duplicate implementation steps. Follow the proportional verification requirements above without adding a new tooling stack solely for these guidelines.

Before considering implementation complete, read through a baseline experiment as a newcomer: are the data flow, tensor meanings, and scoring decisions apparent? Simplify unnecessary indirection and remove unused options or helpers. Readability and maintainability take priority over brevity, cleverness, or hypothetical future reuse.


## 7. Verification and deliverables

The report must separate observed scores, mathematical identities, modeling choices, and interpretations. Deliver a concise walkthrough supported by machine-readable evidence:

1. Native embedding, linear projection, and bipolar similarity scores for every fixed query/value pair, including all music phrases and controls.
2. Exact isolated unbinding and full-bundle residual expansions, with a small coordinate excerpt and a contribution chart using the common cosine denominator.
3. Record rankings from semantic queries and equivalent role probes; explain where argmax acts and where it is unnecessary.
4. Positive/negative query scores and the separate exact source filter.
5. Known stored-fact removal, semantic-query subtraction, normalization, and majority-sign controls.

Use Matplotlib with Python lists from Torch or Polars when possible. Every displayed number must come from a result row. Do not use a two-dimensional projection as evidence of high-dimensional similarity. No capacity curves, stress datasets, statistical error guarantees, or performance claims belong in this report.

Planned runbook, available after implementation:

```bash
uv sync --frozen
uv run src/01_embed.py
uv run src/02_encode.py
uv run src/03_probe.py
uv run src/04_search.py
uv run src/05_subtract.py
uv run src/06_report.py
# Optional: edit ROLE and QUERY_TEXT at the top of this script first.
uv run src/07_try_query.py
uv run pytest -q
```

`01_embed.py` creates the embedding cache and manifest, contacting Ollama only for missing inputs after verifying model identity. Scripts 02-06 read saved artifacts and must work with Ollama stopped; missing prerequisites produce a clear message naming the script to run first. Track completion per stage and refuse to report incomplete evidence. The exploratory script writes a separate result and any new query embeddings without modifying the fixed experiment cache or tables.

For the same frozen inputs and settings, scripts may regenerate their own derived outputs in the configured run directory. If model, data, settings, or projection identity changes, require a new run directory in `settings.py` rather than silently mixing artifacts. No overwrite flags or interactive prompts are needed.

Record the source/settings hashes, Git revision and dirty state (or a null revision with source hashes for an unborn repository), dependency versions, Ollama version and model digest, embedding width, prefix and text templates, preprocessing, matrix shape/scale/hash, role and tie seeds, tensor/storage dtypes, and the script name and effective settings. Store complete embedding responses as vectors for offline replay; reproducibility must not depend solely on the mutable model tag.

Required tests and checks:

- TorchHD binding/bundling agrees with direct Torch multiplication/addition. Exact isolated inversion, distributivity, role isometry, full-probe decomposition, and integer removal/re-addition pass.
- R has the specified shape and only ±1 entries; its independent Bernoulli generation, output-dimension scale, and shared use match the documented construction. Never require an exactly balanced random matrix. A small analytic fixture catches a wrong scale even though cosine and signs alone would not detect it.
- Identical text/mode uses identical cache inputs. Different prefixes cannot collide in cache keys. Appending a query does not change R, roles, existing embeddings, or stored records. Invalid widths, zero vectors, nonfinite responses, model-identity mismatches, and offline misses fail clearly.
- Projection/sign and whole-record sign are distinct steps with explicit zero handling. Floating-point unit normalization is never replaced by TorchHD MAP normalization. Use deterministic fixtures to test the difference.
- Role-probe and role-bound-query cosine agree within `1e-12`; contribution scores sum to full scores. No source values or presumed correct answers enter the scoring functions.
- The role-block reference is computed from the same value vectors and includes within-role correlations in its norm. It must not assume every record norm is `sqrt(ND)` or replace all semantic cross terms with zero.
- Correct handling of duplicate facts, missing fields, invalid roles, dimensions, zero norms, near ties, and absent roles. The unrelated prompt is retained regardless of its rank; do not assert a semantic model output in a unit test.
- Raw removal matches re-encoding; normalized removal uses the original norm; related-query subtraction is not asserted to equal known-fact deletion. Use the exact ambiguity example and a constructive sign-loss fixture for mathematical claims.
- Arrow round trips preserve integer raw sums and declared widths. LanceDB exact search matches Torch under the stated tolerance. No Pandas calls or NumPy arithmetic enter maintained code. No command-line parser, dispatcher, or console-script entrypoint is introduced.
- Replay the completed demonstration from cached embeddings without the Ollama service, reproducing integer vectors and score tables within the declared tolerances. Validate only affected checks again after a relevant change.

Implementation is complete when the documented commands regenerate this fixed walkthrough, mathematical/storage checks pass, and every expected artifact exists with provenance. Unexpected semantic rankings are valid results: investigate whether they originate in the model, projection, signing, or cross-role interference, and keep them visible. Do not change the model, seed, prompts, or data merely to obtain the preferred example.

## Deferred research

A separate experiment will ask how well bundling preserves the embedding model's semantic relationships under increasing interference, and how dimension, role/value correlations, record population, and retrieval indexing affect that behavior. This plan supplies the transparent encoding and scoring path needed for that future study, without defining its sweep grid, capacity thresholds, calibration protocol, or success criteria now. The blog narrative will be developed in a separate session from the completed walkthrough.
