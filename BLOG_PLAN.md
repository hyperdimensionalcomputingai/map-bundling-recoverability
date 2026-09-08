# What can we recover from a bundled hypervector?

## Story and opening

- Continue from the earlier explanation of binding: a known role gives us a way to retrieve its associated value. Revisit the exact bipolar MAP identity `ROLE ⊗ (ROLE ⊗ VALUE) = VALUE`, then ask what changes when several associations occupy the same vector.
- Connect to the wider HDC family accurately: MAP has exact isolated unbinding; HRR supports unbinding, with ordinary correlation-based recovery generally approximate. Avoid extending exact MAP reversibility to every algebra. Link to the [earlier VSA explanation](https://hyperdimensionalcomputing.ai/blog/a-brief-history-of-vsa/).
- Recall that the [HRR/MAP study](https://hyperdimensionalcomputing.ai/blog/hrr-map-similarity/) showed preservation of similar encoder-defined relationships in its tested setting. Distinguish that result from a measurement of unbinding accuracy; use MAP here because its arithmetic makes the next question easier to follow.
- Establish the central question: **If bundling does not have an inverse that reconstructs its unknown ingredients, how does information remain recoverable?**
- Develop the thesis through the story: binding creates addressable associations; bundling superimposes their evidence so that roles and partial queries remain useful. Their different recovery properties serve different purposes by design.

## What the evidence needs to establish

Use the existing [report](artifacts/demo/REPORT.md) as supporting evidence within the narrative, rather than organizing the post around experiment numbers or script execution.

| Reader's question | What the experiments demonstrate |
| --- | --- |
| What survives when facts are bundled? | A role probe contains the associated value or values plus contributions from other roles. |
| What is crosstalk, and why can useful evidence stand out? | The residual stays in the vector, while independent role masks can make its positive and negative score contributions largely cancel. |
| How is a noisy vector interpreted? | Similarity measures alignment with a supplied query; argmax selects the highest-scoring record, without decoding its original fields. |
| Can an unstored phrase find relevant information? | Semantic queries such as lounge music can retrieve a record containing jazz, with measured scores and margins. |
| Can a query express wanted and unwanted evidence? | Adding and subtracting role-bound cues changes record ranking; exact exclusion remains a source-data operation. |
| What can actually be reversed? | Isolated MAP binding and known-contribution removal from a raw sum are exact; neither supplies a general inverse for an unknown bundle. |

## Narrative outline

### 1. From one reversible association to a complete record

- Introduce Maya and Nina with their existing ages, eye colors, and three interests each. Keep names as record identifiers.
- Build Maya's representation from five role-value facts; contrast the single binding identity with the resulting additive mixture.
- Explain why a fixed-width superposition has no explicit field slots to unpack, while preserving useful similarity to its constituent facts.
- Introduce semantic value encoding briefly: local Nomic embeddings, one shared Rademacher projection, then bipolar MAP values. The model supplies semantic relationships; binding and bundling organize them.

### 2. Ask one role what it can tell us

- Begin with Maya's eye color: `EYE ⊗ Maya = H_doc(brown eyes) + ξ_eye`.
- Expand enough of the expression to show where crosstalk comes from. The eye role cancels its own binding; other facts remain transformed in the probe.
- Then ask about interests: `INTEREST ⊗ Maya = H_doc(tea) + H_doc(climbing) + H_doc(jazz) + ξ_interest`.
- Make the contrast explicit: one role occurs once, the other three times. Both probes retain evidence; neither operation returns a decoded string or list.

### 3. Why the remaining mixture can still be useful

- Explain near orthogonality through positive and negative contributions to a dot product before introducing the `1/sqrt(D)` scale for a single independently masked bipolar term.
- Use the existing contribution figure to distinguish a vector's magnitude from its alignment with a query. Noise can remain substantial in the vector while contributing little to a particular score.
- Preserve the nuance that related same-role values are correlated: tea and climbing also contribute to Maya's lounge-music score. Recovery does not mean isolating jazz or erasing every other contribution.

### 4. Turn recognizable evidence into a query

- Move from probing Maya to asking which person aligns with a cue: compare a role-bound lounge-music query with both complete records.
- Explain cosine, then argmax: one measures alignment; the other returns the position of the highest score. Show both records and their margin.
- Explain the equivalence between comparing a role probe with a query value and comparing the original record with the role-bound query.
- Contrast this briefly with dictionary cleanup, which selects among supplied value candidates. Our open semantic query ranks records without first translating lounge music into a jazz label.
- Retain the unrelated-query result to show why every winner is not necessarily a meaningful match. Exact source text, when needed, comes from the retrieved record's metadata.

### 5. Refine the question using the same additive structure

- Start with tea, a shared interest, then introduce the brown-eyes penalty.
- Show how positive and negative evidence changes the preference between Maya and Nina.
- Explain why correlated semantic values and query/document differences prevent an assumption of exact cancellation. Use a literal source predicate only to distinguish guaranteed exclusion from graded preference.

### 6. Separate removing information from discovering it

- Remove the exact stored jazz contribution and compare with a fresh encoding without jazz. Explain why this works for the retained additive accumulator.
- Contrast subtraction of lounge music with removal of stored jazz. Related meaning does not make their vectors interchangeable.
- Explain why residual music similarity can survive correct removal, and briefly distinguish retained raw sums, L2 scaling, and majority signing.
- Resolve the opening question by separating exact isolated unbinding, semantic recognition in a mixture, and exact removal of a known contribution. Use the small ambiguous-sum example only if needed to explain why an unknown inventory has no universal decoder.

### 7. From useful recovery to its capacity limit

- Conclude that bundling preserves evidence we can address, compare, and compose; its usefulness does not depend on reversing every encoding step into the original object.
- Raise the next question naturally: how much can we superimpose before the correct information becomes difficult to distinguish from interference?
- Introduce a separate capacity experiment that will vary facts per bundle and dimensionality, measuring constituent readout under nominal and ordinal value encodings.
- Frame that follow-up as a study of internal bundling pressure, not database size or ANN speed. It will measure recovery curves rather than promise a universal capacity cutoff; its structured-value setup will not by itself establish a capacity limit for Nomic semantic queries.
