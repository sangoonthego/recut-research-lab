# Error Taxonomy for Advertising Consistency Classification

This taxonomy establishes systematic error categories for evaluating predictions on advertising title vs. landing page consistency.

---

## 1. Classification Error Types

In binary consistency classification (`consistent` vs. `inconsistent`):

- **False Positive (FP)**:
  - **Ground Truth**: `inconsistent`
  - **Prediction**: `consistent`
  - **Impact in Production**: Severe risk. Misleading or ungrounded ads pass the filter and reach consumers, causing regulatory non-compliance or poor searcher experience.

- **False Negative (FN)**:
  - **Ground Truth**: `consistent`
  - **Prediction**: `inconsistent`
  - **Impact in Production**: Moderate risk. Valid generated advertising headlines are unnecessarily discarded, lowering generation yield.

---

## 2. Failure Mode Categories

| Category Code | Category Name | Description | Typical Failure Mode |
|---|---|---|---|
| **E1** | **Lexical Overlap Illusion** | High n-gram overlap exists between title and LP, but the specific predicate or relation in the title is unsupported by the LP. | Classifier predicts `consistent` (FP) due to dense word matching. |
| **E2** | **Paraphrase / Synonym Gap** | Ad title uses Japanese colloquial terms, abbreviations, or creative phrasing not verbatim in the LP. | Classifier predicts `inconsistent` (FN) due to low n-gram overlap. |
| **E3** | **Promotional / Numeric Discrepancy** | Title claims specific quantitative benefits (e.g., "50% off", "0 yen initial fee"), but LP lists different terms or requires hidden conditions. | Classifier predicts `consistent` (FP) if the general product name matches. |
| **E4** | **Geographic / Scope Mismatch** | Title specifies a local region (e.g., city, prefecture) while LP is a generic nationwide directory that does not confirm local inventory. | Classifier fails on fine-grained named entity grounding. |
| **E5** | **Product Tier / Variant Drift** | Title advertises a specific model or tier, while LP describes an entirely different tier or general brand family. | Semantic drift missed by bag-of-words / n-gram models. |
| **E6** | **LP Extraction Noise / Truncation** | Landing page text contains raw navigation boilerplate, cookie disclaimers, or cut-off content. | Model classifies based on boilerplate rather than body text. |

---

## 3. Systematic Diagnostic Procedure

During error analysis (Notebook 03):
1. Filter validation predictions to misclassified instances (`y_true != y_pred`).
2. Stratify errors by prediction confidence (`|P(consistent) - 0.5|`).
3. Partition errors into FP and FN subgroups.
4. Measure correlation with text length and lexical overlap features.
5. Sample and tag representative instances according to categories E1–E6.

