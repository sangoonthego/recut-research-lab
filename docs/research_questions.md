# Research Questions and Hypotheses

**Project**: Recut Research Lab 01  
**Domain**: Applied NLP / Advertising Consistency Classification  
**Benchmark**: CyberAgent AdTEC (Ad-Consistency)  

---

## 1. Problem Statement

In search engine advertising, an advertisement consists of an ad title and an associated landing page (LP). Consistency between the title and the LP is critical: inconsistent ads mislead searchers, degrade user trust, and violate ad platform policies.

In the context of the Recut advertising-generation product, automated models generate multiple candidate ad headlines and descriptions. Before presenting these candidates to advertisers or deploying them in live campaigns, an automated consistency classification system must filter out inconsistent or unsubstantiated headlines.

This research laboratory investigates binary consistency classification (`consistent` vs. `inconsistent`) on Japanese search engine ads, establishes rigorous baselines, analyzes failure modes, and outlines transfer pathways to Recut's generative workflow.

---

## 2. Research Questions and Hypotheses

### RQ1: Lexical and Surface Representation Limits
- **Question**: To what extent can surface character n-gram statistics and lexical overlap distinguish consistent ad titles from inconsistent ones in Japanese?
- **Hypothesis H1**: Surface lexical matching (TF-IDF char n-grams) achieves strong recall on blatant topic discrepancies (e.g. ad for credit cards pointing to rental housing) but fails on subtle semantic misalignments (e.g. eligibility conditions, price promotions not found on the LP, or navigational claims).

### RQ2: Landing-Page Clustering and Split Leakage
- **Question**: Do shared landing pages across train, validation, and test splits introduce distribution leakage that artificially inflates benchmark performance?
- **Hypothesis H2**: Models memorizing specific LP domain terms will demonstrate higher performance on validation samples sharing LPs with the training set compared to unseen LPs, necessitating LP-clustered validation splits.

### RQ3: Primary Failure Modes and Error Taxonomy
- **Question**: What are the systematic error categories observed in baseline consistency classifiers?
- **Hypothesis H3**: Baseline false positives will predominantly stem from high lexical overlap where the LP mentions the keyword but does not substantiate the specific claim in the title. False negatives will occur when the ad paraphrases landing page concepts using novel phrasing.

### RQ4: Cross-Domain Transfer to Generative Recut Workflows
- **Question**: Can consistency classifiers trained on search engine ad titles generalize to generated social and display ad copy in specialized vertical markets (e.g. e-commerce, consumer goods)?
- **Hypothesis H4**: Domain transfer will encounter domain shifts in text length, stylistic tone, and domain vocabulary, requiring zero-shot evidence selection and parameter-efficient domain adaptation rather than direct inference.

---

## 3. Scope Boundaries and Ethical Guardrails

- **Ad Consistency vs. Medical Truth**: The AdTEC benchmark evaluates editorial and thematic consistency between ad titles and landing pages. It does **not** perform medical fact-checking or evaluate scientific efficacy claims.
- **Dataset Licensing**: The AdTEC dataset is published under CC BY-NC-SA 4.0 (Non-Commercial, Share-Alike). Experimental findings in this lab are strictly for research and benchmarking.
