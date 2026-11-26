"""Recut Research Lab: Research-first workspace for Advertising Consistency Classification.

This library provides tooling for data loading, auditing, baseline modeling,
evaluation, error analysis, and reproducible experimentation.
"""

from recut_lab.data import (
    ConsistencyDataset,
    load_adtec_dataset,
    load_adtec_train_valid,
    validate_schema,
)
from recut_lab.evaluation import (
    LABEL_ORDER,
    benchmark_inference_latency,
    compute_classification_metrics,
    evaluate_predictions,
)
from recut_lab.baseline import (
    BaselineModel,
    M2_DEFAULT_CONFIG,
    MajorityBaseline,
    TfidfLogRegBaseline,
    run_m2_baseline_benchmark,
)
from recut_lab.experiment import ExperimentManager
from recut_lab.diagnostics import (
    M2_0_1_DEFAULT_CONFIG,
    SeparateBlocksTfidfLogReg,
    TruncatedLpPairBaseline,
    compute_d1_score_diagnostics,
    compute_seen_unseen_confounding,
    extract_class_scores,
    run_m2_failure_diagnostics,
)
from recut_lab.generalization import (
    M2_1_DEFAULT_CONFIG,
    PROTOCOL_A,
    PROTOCOL_B,
    assign_lp_group_ids,
    audit_lp_groups,
    load_adtec_train_only,
    run_rq1_generalization_study,
)
from recut_lab.lexical_interactions import (
    LEXICAL_INTERACTION_FEATURE_NAMES,
    OVERLAP_BIN_ORDER,
    LexicalInteractionTransformer,
    assign_overlap_bins,
    compute_pairwise_lexical_features,
    extract_lexical_interaction_frame,
    fit_overlap_bin_edges,
    has_numeric_or_quantity_token,
)
from recut_lab.semantic import (
    DEFAULT_E5_LICENSE,
    DEFAULT_E5_MODEL_ID,
    DEFAULT_E5_PREFIX,
    DEFAULT_E5_REVISION,
    FrozenSentenceTransformerEncoder,
    SemanticEmbeddingBundle,
    build_semantic_pair_features,
    compute_cosine_similarity,
    extract_or_load_cached_embeddings,
    l2_normalize_embeddings,
    run_semantic_smoke_test,
)
from recut_lab.lexical_semantic import (
    M2_2_DEFAULT_CONFIG,
    MODEL_ORDER_RQ2,
    HybridLexicalSemanticLogReg,
    LexicalInteractionLogReg,
    SemanticPairLogReg,
    load_m2_1_protocol_b_folds,
    run_rq2_lexical_semantic_study,
)

__version__ = "0.1.0"

__all__ = [
    "ConsistencyDataset",
    "load_adtec_dataset",
    "load_adtec_train_valid",
    "load_adtec_train_only",
    "validate_schema",
    "LABEL_ORDER",
    "benchmark_inference_latency",
    "evaluate_predictions",
    "compute_classification_metrics",
    "BaselineModel",
    "M2_DEFAULT_CONFIG",
    "MajorityBaseline",
    "TfidfLogRegBaseline",
    "run_m2_baseline_benchmark",
    "ExperimentManager",
    "M2_0_1_DEFAULT_CONFIG",
    "SeparateBlocksTfidfLogReg",
    "TruncatedLpPairBaseline",
    "compute_d1_score_diagnostics",
    "compute_seen_unseen_confounding",
    "extract_class_scores",
    "run_m2_failure_diagnostics",
    "M2_1_DEFAULT_CONFIG",
    "PROTOCOL_A",
    "PROTOCOL_B",
    "assign_lp_group_ids",
    "audit_lp_groups",
    "run_rq1_generalization_study",
    "LEXICAL_INTERACTION_FEATURE_NAMES",
    "OVERLAP_BIN_ORDER",
    "LexicalInteractionTransformer",
    "assign_overlap_bins",
    "compute_pairwise_lexical_features",
    "extract_lexical_interaction_frame",
    "fit_overlap_bin_edges",
    "has_numeric_or_quantity_token",
    "DEFAULT_E5_LICENSE",
    "DEFAULT_E5_MODEL_ID",
    "DEFAULT_E5_PREFIX",
    "DEFAULT_E5_REVISION",
    "FrozenSentenceTransformerEncoder",
    "SemanticEmbeddingBundle",
    "build_semantic_pair_features",
    "compute_cosine_similarity",
    "extract_or_load_cached_embeddings",
    "l2_normalize_embeddings",
    "run_semantic_smoke_test",
    "M2_2_DEFAULT_CONFIG",
    "MODEL_ORDER_RQ2",
    "HybridLexicalSemanticLogReg",
    "LexicalInteractionLogReg",
    "SemanticPairLogReg",
    "load_m2_1_protocol_b_folds",
    "run_rq2_lexical_semantic_study",
]

