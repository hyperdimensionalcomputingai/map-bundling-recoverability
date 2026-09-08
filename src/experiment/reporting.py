"""Turn saved evidence into an inspectable report, without recalculating embeddings."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from . import settings as cfg
from .storage import load_json, read_table

MUSIC = ["jazz", "music", "lounge_music", "r_and_b"]


def markdown(table: pl.DataFrame) -> str:
    def cell(value):
        if isinstance(value, float):
            return f"{value:.3g}" if 0 < abs(value) < 1e-5 else f"{value:.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(table.columns) + " |\n"
    separator = "| " + " | ".join("---" for _ in table.columns) + " |\n"
    return (
        header
        + separator
        + "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in table.iter_rows())
    )


def figures(run_dir: Path = cfg.RUN_DIR) -> list[Path]:
    directory = run_dir / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#246c91", "#d08938", "#5b8351"]
    output = []

    geometry = read_table("geometry", run_dir).filter(
        pl.col("query_id").is_in(MUSIC) & (pl.col("document_input") == "search_document: jazz")
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    rows = geometry.to_dicts()
    for j, (column, label) in enumerate(
        (
            ("native_cosine", "Native embedding"),
            ("linear_cosine", "Linear Rademacher"),
            ("bipolar_cosine", "Signed MAP"),
        )
    ):
        ax.bar(
            [i + (j - 1) * 0.24 for i in range(len(rows))],
            [r[column] for r in rows],
            width=0.23,
            label=label,
            color=colors[j],
        )
    ax.set_xticks(list(range(len(rows))), [r["query_text"] for r in rows])
    ax.set_ylabel("Cosine with stored jazz value")
    ax.set_title("Separate embedding geometry, linear projection, and signing")
    ax.legend(fontsize=8)
    path = directory / "01_geometry.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    output.append(path)

    terms = (
        read_table("contributions", run_dir)
        .filter((pl.col("query_id") == "lounge_music") & (pl.col("person_id") == "maya"))
        .to_dicts()
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    ax.barh(
        [r["fact_text"] for r in terms],
        [r["contribution"] for r in terms],
        color=[colors[0] if r["same_role"] else colors[1] for r in terms],
    )
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Contribution to the full probe cosine (common denominator)")
    ax.set_title("Maya queried with lounge music: each stored fact still contributes")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=colors[0], label="Same role"),
            Patch(color=colors[1], label="Other role"),
        ]
    )
    path = directory / "02_contributions.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    output.append(path)

    removal = read_table("removal_scores", run_dir)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), layout="constrained")
    for j, (operation, label) in enumerate(
        (
            ("original", "Original"),
            ("raw_remove_stored_jazz", "Stored jazz removed"),
            ("remove_lounge_query", "Lounge query subtracted"),
        )
    ):
        rows = removal.filter(pl.col("operation") == operation).to_dicts()
        ax.bar(
            [i + (j - 1) * 0.24 for i in range(len(rows))],
            [r["score"] for r in rows],
            width=0.23,
            label=label,
            color=colors[j],
        )
    ax.set_xticks(list(range(len(rows))), [r["query_text"] for r in rows])
    ax.set_ylabel("Cosine with role-bound query")
    ax.set_title("Known-fact removal and related-query subtraction do different things")
    ax.legend(fontsize=8)
    path = directory / "03_removal.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    output.append(path)
    return output


def build_report(run_dir: Path = cfg.RUN_DIR) -> Path:
    manifest = load_json(run_dir / "manifest.json")
    geometry = read_table("geometry", run_dir)
    probes = read_table("probe_scores", run_dir)
    search = read_table("search_scores", run_dir)
    removal = read_table("removal_checks", run_dir)
    music = search.filter(pl.col("query_id").is_in(MUSIC)).select(
        "query_text",
        "person_id",
        "native_block_score",
        "bipolar_block_score",
        "score",
        "rank",
    )
    winners = search.filter((pl.col("scenario") == "semantic") & (pl.col("rank") == 1)).select(
        "query_text", "person_id", "score", "tied"
    )
    negative = read_table("negative_scores", run_dir).select(
        "query_id", "person_id", "raw_dot", "score", "rank"
    )
    contributions = read_table("contributions", run_dir).filter(
        (pl.col("query_id") == "lounge_music") & (pl.col("person_id") == "maya")
    )
    controls = search.filter(pl.col("scenario") != "semantic").select(
        "query_id", "person_id", "score", "rank"
    )
    jazz_geometry = geometry.filter(pl.col("document_input") == "search_document: jazz").select(
        "query_text", "native_cosine", "linear_cosine", "bipolar_cosine"
    )
    all_scores = search.filter(pl.col("scenario") == "semantic").select(
        "query_text", "role", "person_id", "score", "rank", "margin", "tied"
    )
    inversion = read_table("inversion_checks", run_dir)
    records = read_table("records", run_dir).select("person_id", "fact_count", "raw_norm")
    excerpt = read_table("probe_excerpt", run_dir).filter(
        (pl.col("query_id") == "lounge_music") & (pl.col("person_id") == "maya")
    )
    lounge_geometry = jazz_geometry.filter(pl.col("query_text") == "lounge music").row(
        0, named=True
    )
    lounge_results = search.filter(pl.col("query_id") == "lounge_music").sort("rank")
    unrelated_result = search.filter(pl.col("query_id") == "aircraft_maintenance").sort("rank")
    same_role_score = contributions.filter(pl.col("same_role"))["contribution"].sum()
    cross_role_score = contributions.filter(~pl.col("same_role"))["contribution"].sum()
    tea_winner = negative.filter(pl.col("query_id") == "tea").sort("rank")["person_id"][0]
    negative_winner = negative.filter(pl.col("query_id") == "tea_minus_brown").sort("rank")[
        "person_id"
    ][0]
    residual_lounge = read_table("removal_scores", run_dir).filter(
        (pl.col("operation") == "raw_remove_stored_jazz") & (pl.col("query_id") == "lounge_music")
    )["score"][0]
    paths = sorted((run_dir / "tables").glob("*.parquet"))
    evidence_links = "\n".join(f"- [{path.stem}](tables/{path.name})" for path in paths)
    report = f"""# Probing and querying a bundled semantic record

A fixed teaching experiment with local Nomic embeddings and bipolar MAP. Two fictional people each contribute five facts. No candidate vocabulary is used to interpret query text: we embed the supplied phrase and compare its role-bound vector with stored person vectors.

This report is generated from saved tables. It measures the specified example, not production scalability or capacity under increasing interference. The complete input and software identity is in [manifest.json](manifest.json).

## What the run returned

These are observed winners, not ground-truth relevance labels. A winner is returned even for a poor query. Ties are explicitly marked.

{markdown(winners)}

The native model, the projection/sign step, and bundling each affect the final scores. The following reference comparisons keep their effects separate:

{markdown(music)}

`native_block_score` keeps roles in separate slots in embedding space. `bipolar_block_score` uses the same explicit slots with signed values. `score` is the bundled-record cosine. Both block references retain correlations among values sharing a role. Comparing the first two exposes projection/sign changes; comparing the latter two exposes cross-role interference, including its effect on the norm. None is a human relevance annotation.

## From model embeddings to MAP values

Stored values use `search_document:` and queries use `search_query:`. Identical wording in those modes is not an identical vector. The shared Rademacher matrix maps {cfg.EMBEDDING_DIM} coordinates to {cfg.DIMENSIONS}, with scale `1/sqrt(D)` before signing. It does not add semantic information, and linear projection guarantees do not automatically survive signing.

All fixed interest queries against stored jazz:

{markdown(jazz_geometry)}

![Native, linear, and signed geometry](figures/01_geometry.png)

The complete geometry table contains {geometry.height} query/value pairs, including non-music controls. Every prompt remains in the results regardless of its score.

**E1 takeaway.** For lounge music versus stored jazz, cosine moves from {lounge_geometry["native_cosine"]:.3f} in the embedding model to {lounge_geometry["linear_cosine"]:.3f} after linear projection and {lounge_geometry["bipolar_cosine"]:.3f} after signing, so these stages already change the score before any bundling occurs. This baseline matters because a later change in retrieval cannot automatically be blamed on bundling, and the signed score should not be interpreted on the original embedding model's scale.

## Binding reverses an association; a full probe retains a mixture

All {inversion.height} isolated role/value pairs passed exact coordinatewise inversion. For the full Maya interest probe:

```text
INTEREST * Maya = H_doc(tea) + H_doc(climbing) + H_doc(jazz)
               + INTEREST * AGE * H_doc(34 years old)
               + INTEREST * EYE * H_doc(brown eyes)
```

The last two terms remain in the vector. When compared with a query, their projections can be small without their vector norms being small. This unwanted cross-role mixture is crosstalk. Same-role similarity to tea, climbing, and jazz is preserved rather than randomized away. A same-role contribution is model alignment, which is not automatically human relevance.

Maya queried with lounge music, with every term divided by the same full-probe denominator:

{markdown(contributions.select("fact_text", "same_role", "dot", "denominator", "contribution"))}

![Individual fact contributions](figures/02_contributions.png)

Across {probes.height} probes, the largest contribution-sum discrepancy was {probes["contribution_error"].max():.3g}. Role probes and role-bound queries agreed within `1e-12`. Their equality follows from the bipolar role's sign flips preserving dot products and norms; it does not depend on semantic relevance.

This excerpt shows the first 16 coordinates of the real {cfg.DIMENSIONS}-dimensional Maya / lounge music probe. Scores use all coordinates.

{markdown(excerpt.select("coordinate", "probe", "same_role_sum", "cross_role_sum", "query"))}

Observed record norms, without an assumption that all five facts are orthogonal:

{markdown(records)}

**E2 takeaway.** Maya's lounge-music score contains {same_role_score:.3f} from the three interests combined and {cross_role_score:.3f} from the other roles, so the observed match reflects a mixture of interests rather than an isolated recovery of jazz. The small cross-role contribution illustrates cancellation during scoring, while the exact isolated inversions show that the remaining mixture comes from superposition rather than a failure of binding's inverse.

## What argmax does here

Given one supplied phrase, scoring Maya needs one cosine. To search people, compute a cosine for each record, then `argmax` selects the position of the highest score; the position maps to a person ID. `max` would return the score itself. There is no intermediate lookup that converts lounge music into a jazz label, and the probe is not cleaned into original text. Source fields can be fetched separately for exact output.

Every fixed query and both records:

{markdown(all_scores)}

Exact LanceDB search agreed with the Torch reference: largest absolute discrepancy {search["lance_error"].max():.3g}. No ANN index or accuracy claim is involved. Returning both of two people is not a meaningful recall benchmark.

Changing or omitting the role gives these controls:

{markdown(controls)}

For one fixed bipolar value and query, an independent wrong-role mask makes each coordinate of their dot product a fair positive or negative vote. Over D independent coordinates the dot product has mean zero and standard deviation `sqrt(D)`. Since each bipolar vector has norm `sqrt(D)`, their cosine has standard deviation `1/sqrt(D)`, which is {1 / cfg.DIMENSIONS**0.5:.6f} here. Positive and negative contributions tend to cancel when scoring, although the masked vector itself still has full norm.

This explains why independent role masks can suppress a cross-role term's alignment with a query. It does not make correlated same-role interests disappear, and terms sharing a mask need not be independent. Their measured sum and the full record norm determine the actual score. A dictionary cleanup would use this separation to select a candidate; our semantic record search uses it to rank people. Neither operation literally erases noise coordinates.

Near orthogonality is an expectation over independent role masks, not a promise that each control scores zero. The model's correlated semantic values cannot be treated as an independent random vocabulary or assigned a random-vector confidence threshold.

**E3 takeaway.** Lounge music ranks {lounge_results["person_id"][0]} first by a cosine margin of {lounge_results["margin"][0]:.3f}, while even the unrelated aircraft-maintenance query ranks {unrelated_result["person_id"][0]} first by {unrelated_result["margin"][0]:.3f}. This demonstrates semantic record retrieval without decoding a stored label, but also why an argmax result needs its scores and context: agreement between Torch and LanceDB validates the retrieval calculation, not relevance or scalability.

## Negative evidence is a score contribution

Compare the tea query with `INTEREST * H_query(tea) - EYE * H_query(brown eyes)`:

{markdown(negative)}

The negative term penalizes semantic alignment. Query/document prefixes and related values prevent an assumption of exact cancellation. It is not Boolean NOT. Separately, the literal source predicate `tea in interests AND eye_color != brown` gives:

{markdown(read_table("source_filter", run_dir))}

That predicate answers a narrower exact question; it is not ground truth for all semantic matches.

**E4 takeaway.** Adding the negative brown-eyes term changes the leading record from {tea_winner} to {negative_winner}, showing that a composed query can adjust the preference expressed by the tea cue. This is useful for soft ranking, but exact exclusion still requires a source-data predicate because semantic subtraction neither deletes records nor guarantees Boolean NOT.

## Known removal is different from discovering unknown operands

Subtracting the exact stored jazz contribution reproduced a fresh encoding with jazz omitted, and re-adding it restored the original integer accumulator. Related-query subtraction is a different edit. Neither lounge music nor even query-prefixed jazz is assumed to equal the document-side stored contribution.

{markdown(removal.select("operation", "expected", "exact", "within_1e_12", "max_coordinate_error"))}

![Known removal and semantic edits](figures/03_removal.png)

L2-normalized records need the original scale for an exact update; whole-record majority signing has discarded margins. The four-fact residual had {removal["residual_zero_votes"][0]} zero coordinates, handled with the frozen tie vector for the sign control. Nonzero music-query similarity after known removal can come from other interests and cross-role interference; it does not contradict coordinatewise removal.

A constructive sign-loss example makes the lost information explicit: raw sums `[3, 1]` and `[1, 3]` both sign to `[1, 1]`. Removing the same known fact `[1, 1]` leaves `[2, 0]` and `[0, 2]`. With a fixed negative tie choice, their signed residuals differ. The original signed vector and known fact alone cannot distinguish these cases. The test suite checks this example.

The [ambiguity example](ambiguity.json) shows two distinct two-vector inventories sharing `[2, 0, 0, 0]`. It demonstrates the absence of a universal inverse, not a measured collision in Nomic data. External dictionaries and constraints can make particular sums recoverable. This implementation probes and compares a bundle without decoding an unknown symbolic inventory.

**E5 takeaway.** Removing the stored jazz vector is exact even though the residual still scores {residual_lounge:.3f} against lounge music, because the remaining facts also contribute to that query. This matters for updates: a raw bundle supports removing a known contribution, while discovering unknown operands, subtracting a related phrase, and updating a majority-signed bundle are different operations with different information requirements.

## Reproduce and inspect

Run the numbered scripts documented in the repository README. Scripts 02-06 use cached vectors and do not call Ollama. Edit the constants in `src/07_try_query.py` for a new query; it leaves fixed experiment tables unchanged. No CLI is required.

Model: `{cfg.MODEL}`. Digest: `{cfg.MODEL_DIGEST}`. Ollama version recorded by the embedding stage: `{manifest.get("ollama_version", "cached run; see embedding provenance")}`. Master seed: {cfg.SEED}. Projection/roles and all evidence files are hashed in the manifest.

{evidence_links}

## Limits and next question

Two people and one fixed representation demonstrate mechanics. They do not establish universal semantic correctness, an optimal dimension, a calibrated threshold, or production latency. Embedding inference and projection happen once per query; exhaustive record comparison scales with record count and vector width. How well semantic relationships survive increasing interference remains a separate future experiment.
"""
    path = run_dir / "REPORT.md"
    path.write_text(report)
    return path
