"""Frozen pretrained semantic embeddings and deterministic caching for RQ2.

Uses ``intfloat/multilingual-e5-small`` via ``sentence-transformers`` in strict
frozen inference mode (CPU-compatible, ``torch.no_grad()``, no fine-tuning,
deterministic batching, ``float32`` precision, revision-aware local caching).

Prefix Policy (Verified from ``intfloat/multilingual-e5-small`` README FAQ #1):
    For non-retrieval tasks including linear-probing classification and symmetric
    pair tasks, the official model card instructs prepending ``"query: "`` to each
    input text. We apply ``"query: "`` consistently to both NFKC-normalized
    ``title`` and NFKC-normalized ``lp_text``.

Important Scientific Guardrail:
    Pretrained semantic embeddings capture distributional/topical similarity, NOT
    factual contradiction, numerical verification, or advertising compliance.
"""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import json
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from recut_lab.preprocessing import normalize_japanese_text

DEFAULT_E5_MODEL_ID: str = "intfloat/multilingual-e5-small"
DEFAULT_E5_REVISION: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DEFAULT_E5_LICENSE: str = "mit"
DEFAULT_E5_PREFIX: str = "query: "
DEFAULT_E5_PREFIX_POLICY_NOTE: str = (
    "Per intfloat/multilingual-e5-small official model card FAQ #1, non-retrieval "
    "classification and symmetric pair feature extraction use the 'query: ' prefix "
    "for every input text. Applied identically to NFKC-normalized title and lp_text."
)


class TextEncoderProtocol(Protocol):
    """Minimal encoder interface enabling deterministic synthetic mocks in unit tests."""

    model_id: str
    resolved_revision: str
    license_name: str
    device: str
    max_seq_length: int
    embedding_dim: int

    def weight_sha256(self) -> str: ...

    def measure_token_lengths(self, prefixed_texts: Sequence[str]) -> Dict[str, Any]: ...

    def encode(
        self,
        prefixed_texts: Sequence[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray: ...


def hash_normalized_text(normalized_text: str) -> str:
    """Compute a deterministic SHA-256 content hash for an NFKC-normalized string."""
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def get_process_memory_mb() -> Optional[float]:
    """Measure current process resident set size (RSS) in MiB on Windows/POSIX."""
    try:
        import psutil  # type: ignore

        proc = psutil.Process(os.getpid())
        return round(float(proc.memory_info().rss) / (1024.0 * 1024.0), 2)
    except Exception:
        pass

    if os.name == "nt":
        try:
            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return round(float(counters.WorkingSetSize) / (1024.0 * 1024.0), 2)
        except Exception:
            return None
    return None


def get_peak_working_set_mb() -> Optional[float]:
    """Measure peak process working set in MiB on Windows when available."""
    if os.name == "nt":
        try:
            class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                handle, ctypes.byref(counters), counters.cb
            )
            if ok:
                return round(float(counters.PeakWorkingSetSize) / (1024.0 * 1024.0), 2)
        except Exception:
            return None
    return get_process_memory_mb()


def l2_normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize embedding rows safely to unit norm in float32."""
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D embedding array, got shape {arr.shape}.")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, np.float32(eps))
    return (arr / norms).astype(np.float32)


def compute_cosine_similarity(u_lp: np.ndarray, v_title: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute row-wise cosine similarity between LP embeddings u and title embeddings v."""
    u = np.asarray(u_lp, dtype=np.float32)
    v = np.asarray(v_title, dtype=np.float32)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError(f"Mismatched 2D embedding shapes: u={u.shape}, v={v.shape}.")
    dot = np.sum(u * v, axis=1)
    denom = np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1), np.float32(eps))
    sim = np.clip(dot / denom, -1.0, 1.0)
    return sim.astype(np.float32)


def build_semantic_pair_features(u_lp: np.ndarray, v_title: np.ndarray) -> np.ndarray:
    """Construct explicit pair representation ``[u, v, |u - v|, u * v, cos(u, v)]``.

    Args:
        u_lp: Landing-page embeddings of shape ``(N, d)``.
        v_title: Advertising title embeddings of shape ``(N, d)``.

    Returns:
        Float32 feature matrix of shape ``(N, 4 * d + 1)``.
    """
    u = np.asarray(u_lp, dtype=np.float32)
    v = np.asarray(v_title, dtype=np.float32)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError(f"Expected matching 2D arrays, got u={u.shape}, v={v.shape}.")

    abs_diff = np.abs(u - v).astype(np.float32)
    elem_prod = (u * v).astype(np.float32)
    cos_sim = compute_cosine_similarity(u, v).reshape(-1, 1)

    features = np.hstack([u, v, abs_diff, elem_prod, cos_sim]).astype(np.float32)
    if not np.all(np.isfinite(features)):
        raise ValueError("Non-finite values detected in constructed semantic pair features.")
    return features


class FrozenSentenceTransformerEncoder:
    """Strictly frozen SentenceTransformer wrapper for reproducible CPU inference."""

    def __init__(
        self,
        model_id: str = DEFAULT_E5_MODEL_ID,
        revision: Optional[str] = DEFAULT_E5_REVISION,
        license_name: str = DEFAULT_E5_LICENSE,
        device: str = "cpu",
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.requested_revision = revision
        self.license_name = license_name
        self.device = device

        # Enforce deterministic CPU execution settings where supported
        torch.manual_seed(42)

        self.model = SentenceTransformer(
            model_id,
            device=device,
            revision=revision,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Resolve actual commit SHA from Hugging Face cache/hub metadata when available
        self.resolved_revision = self._resolve_revision(revision)
        self.max_seq_length = int(self.model.max_seq_length)
        self.embedding_dim = int(self.model.get_sentence_embedding_dimension())

    def _resolve_revision(self, fallback_revision: Optional[str]) -> str:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(self.model_id, revision=fallback_revision)
            if getattr(info, "sha", None):
                return str(info.sha)
        except Exception:
            pass
        return str(fallback_revision or "unknown")

    def weight_sha256(self) -> str:
        """Compute a SHA-256 digest over all frozen encoder parameters."""
        hasher = hashlib.sha256()
        for name, param in sorted(self.model.named_parameters(), key=lambda kv: kv[0]):
            hasher.update(name.encode("utf-8"))
            hasher.update(param.detach().cpu().numpy().tobytes())
        return hasher.hexdigest()

    def measure_token_lengths(self, prefixed_texts: Sequence[str]) -> Dict[str, Any]:
        """Measure untruncated tokenized lengths and truncation rates against max_seq_length."""
        tokenizer = self.model.tokenizer
        lengths: List[int] = []
        # Tokenize in chunks to avoid huge temporary buffers and silence >max_length warnings
        chunk_size = 512
        for start in range(0, len(prefixed_texts), chunk_size):
            batch = list(prefixed_texts[start : start + chunk_size])
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                verbose=False,
            )
            lengths.extend(len(ids) for ids in encoded["input_ids"])

        arr = np.asarray(lengths, dtype=np.int64)
        truncated_mask = arr > self.max_seq_length
        truncated_count = int(truncated_mask.sum())
        n_total = int(arr.size)
        p50, p95, p99 = (
            np.percentile(arr, [50, 95, 99]).tolist() if n_total > 0 else [0.0, 0.0, 0.0]
        )

        return {
            "count": n_total,
            "max_seq_length": int(self.max_seq_length),
            "truncated_count": truncated_count,
            "truncated_pct": round(truncated_count / max(n_total, 1) * 100.0, 2),
            "min_tokens": int(arr.min()) if n_total > 0 else 0,
            "median_tokens": round(float(p50), 2),
            "mean_tokens": round(float(arr.mean()), 2) if n_total > 0 else 0.0,
            "p95_tokens": round(float(p95), 2),
            "p99_tokens": round(float(p99), 2),
            "max_tokens": int(arr.max()) if n_total > 0 else 0,
            "per_item_token_lengths": lengths,
        }

    def encode(
        self,
        prefixed_texts: Sequence[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode texts deterministically with gradients disabled in float32."""
        import torch

        self.model.eval()
        with torch.no_grad():
            embs = self.model.encode(
                list(prefixed_texts),
                batch_size=int(batch_size),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=bool(normalize_embeddings),
                device=self.device,
            )
        arr = np.asarray(embs, dtype=np.float32)
        if normalize_embeddings:
            arr = l2_normalize_embeddings(arr)
        if not np.all(np.isfinite(arr)):
            raise ValueError("Encoder produced non-finite embeddings.")
        return arr


@dataclass
class SemanticEmbeddingBundle:
    """In-memory handle to cached frozen LP & title embeddings aligned by content hash."""

    lp_embeddings: np.ndarray
    lp_hash_to_idx: Dict[str, int]
    lp_token_lengths_by_hash: Dict[str, int]
    title_embeddings: np.ndarray
    title_hash_to_idx: Dict[str, int]
    title_token_lengths_by_hash: Dict[str, int]
    manifest: Dict[str, Any]
    cache_dir: Path

    def align_for_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        lp_col: str = "lp_text",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return row-aligned ``(u_lp, v_title)`` matrices for every row in ``df``."""
        lp_indices: List[int] = []
        title_indices: List[int] = []

        for lp_raw, title_raw in zip(df[lp_col].tolist(), df[title_col].tolist()):
            lp_h = hash_normalized_text(normalize_japanese_text(lp_raw))
            t_h = hash_normalized_text(normalize_japanese_text(title_raw))
            if lp_h not in self.lp_hash_to_idx:
                raise KeyError(f"Normalized LP hash '{lp_h[:12]}' not found in embedding cache.")
            if t_h not in self.title_hash_to_idx:
                raise KeyError(f"Normalized title hash '{t_h[:12]}' not found in embedding cache.")
            lp_indices.append(self.lp_hash_to_idx[lp_h])
            title_indices.append(self.title_hash_to_idx[t_h])

        u = self.lp_embeddings[lp_indices]
        v = self.title_embeddings[title_indices]
        if u.shape[0] != len(df) or v.shape[0] != len(df):
            raise AssertionError("Aligned embedding row count does not match DataFrame length.")
        return u, v

    def build_pair_matrix(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        lp_col: str = "lp_text",
    ) -> np.ndarray:
        """Align embeddings for ``df`` and construct ``[u, v, |u-v|, u*v, cos(u,v)]``."""
        u, v = self.align_for_dataframe(df, title_col=title_col, lp_col=lp_col)
        return build_semantic_pair_features(u, v)

    def get_row_token_lengths(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        lp_col: str = "lp_text",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned tokenized lengths ``(lp_tokens, title_tokens)`` for each row."""
        lp_toks = [
            self.lp_token_lengths_by_hash[hash_normalized_text(normalize_japanese_text(x))]
            for x in df[lp_col].tolist()
        ]
        t_toks = [
            self.title_token_lengths_by_hash[hash_normalized_text(normalize_japanese_text(x))]
            for x in df[title_col].tolist()
        ]
        return np.asarray(lp_toks, dtype=np.int64), np.asarray(t_toks, dtype=np.int64)


def _compute_cache_key(
    model_id: str,
    revision: str,
    title_prefix: str,
    lp_prefix: str,
    normalize_embeddings: bool,
    dtype: str,
    lp_hashes: Sequence[str],
    title_hashes: Sequence[str],
) -> str:
    """Create a deterministic, revision-aware and content-aware cache directory name."""
    slug = model_id.replace("/", "_").replace("-", "_")
    policy_payload = json.dumps(
        {
            "model_id": model_id,
            "revision": revision,
            "title_prefix": title_prefix,
            "lp_prefix": lp_prefix,
            "normalize_embeddings": bool(normalize_embeddings),
            "dtype": dtype,
            "lp_set_sha256": hashlib.sha256("\n".join(lp_hashes).encode("utf-8")).hexdigest(),
            "title_set_sha256": hashlib.sha256("\n".join(title_hashes).encode("utf-8")).hexdigest(),
        },
        sort_keys=True,
    )
    policy_hash = hashlib.sha256(policy_payload.encode("utf-8")).hexdigest()[:12]
    rev_short = revision[:10] if revision else "unknown"
    return f"{slug}_{rev_short}_{policy_hash}"


def extract_or_load_cached_embeddings(
    df: pd.DataFrame,
    cache_base_dir: Path | str = "data/processed/embeddings_cache",
    model_id: str = DEFAULT_E5_MODEL_ID,
    revision: Optional[str] = DEFAULT_E5_REVISION,
    license_name: str = DEFAULT_E5_LICENSE,
    title_prefix: str = DEFAULT_E5_PREFIX,
    lp_prefix: str = DEFAULT_E5_PREFIX,
    prefix_policy_note: str = DEFAULT_E5_PREFIX_POLICY_NOTE,
    batch_size: int = 32,
    device: str = "cpu",
    normalize_embeddings: bool = True,
    title_col: str = "title",
    lp_col: str = "lp_text",
    encoder: Optional[TextEncoderProtocol] = None,
    force_recompute: bool = False,
) -> SemanticEmbeddingBundle:
    """Encode unique normalized LP texts and titles once (or load deterministic cache).

    Strictly label-independent: never inspects or reads the ``label`` column.
    Stores zero raw Japanese text in cache metadata files (content SHA-256 hashes only).
    """
    # Deduplicate NFKC-normalized texts deterministically (sorted by content hash)
    unique_lp_map: Dict[str, str] = {}
    for raw_lp in df[lp_col].tolist():
        norm_lp = normalize_japanese_text(raw_lp)
        unique_lp_map[hash_normalized_text(norm_lp)] = norm_lp

    unique_title_map: Dict[str, str] = {}
    for raw_t in df[title_col].tolist():
        norm_t = normalize_japanese_text(raw_t)
        unique_title_map[hash_normalized_text(norm_t)] = norm_t

    sorted_lp_hashes = sorted(unique_lp_map.keys())
    sorted_title_hashes = sorted(unique_title_map.keys())

    # Resolve revision early if encoder was already provided
    eff_revision = (
        encoder.resolved_revision if encoder is not None else str(revision or DEFAULT_E5_REVISION)
    )
    eff_model_id = encoder.model_id if encoder is not None else model_id

    cache_key = _compute_cache_key(
        model_id=eff_model_id,
        revision=eff_revision,
        title_prefix=title_prefix,
        lp_prefix=lp_prefix,
        normalize_embeddings=normalize_embeddings,
        dtype="float32",
        lp_hashes=sorted_lp_hashes,
        title_hashes=sorted_title_hashes,
    )
    cache_dir = Path(cache_base_dir) / cache_key
    lp_npy = cache_dir / "lp_embeddings.npy"
    title_npy = cache_dir / "title_embeddings.npy"
    lp_hashes_path = cache_dir / "lp_hashes.json"
    title_hashes_path = cache_dir / "title_hashes.json"
    manifest_path = cache_dir / "cache_manifest.json"

    cache_complete = (
        not force_recompute
        and lp_npy.is_file()
        and title_npy.is_file()
        and lp_hashes_path.is_file()
        and title_hashes_path.is_file()
        and manifest_path.is_file()
    )

    if cache_complete:
        cached_lp_meta = json.loads(lp_hashes_path.read_text(encoding="utf-8"))
        cached_title_meta = json.loads(title_hashes_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if (
            cached_lp_meta["hashes"] == sorted_lp_hashes
            and cached_title_meta["hashes"] == sorted_title_hashes
            and manifest.get("resolved_revision") == eff_revision
        ):
            lp_embs = np.load(lp_npy).astype(np.float32)
            title_embs = np.load(title_npy).astype(np.float32)
            manifest["loaded_from_cache"] = True
            return SemanticEmbeddingBundle(
                lp_embeddings=lp_embs,
                lp_hash_to_idx={h: i for i, h in enumerate(sorted_lp_hashes)},
                lp_token_lengths_by_hash={
                    h: int(tok)
                    for h, tok in zip(sorted_lp_hashes, cached_lp_meta["token_lengths"])
                },
                title_embeddings=title_embs,
                title_hash_to_idx={h: i for i, h in enumerate(sorted_title_hashes)},
                title_token_lengths_by_hash={
                    h: int(tok)
                    for h, tok in zip(sorted_title_hashes, cached_title_meta["token_lengths"])
                },
                manifest=manifest,
                cache_dir=cache_dir,
            )

    # Otherwise extract frozen embeddings deterministically
    active_encoder: TextEncoderProtocol = (
        encoder
        if encoder is not None
        else FrozenSentenceTransformerEncoder(
            model_id=model_id,
            revision=revision,
            license_name=license_name,
            device=device,
        )
    )

    weight_sha_before = active_encoder.weight_sha256()

    prefixed_lps = [f"{lp_prefix}{unique_lp_map[h]}" for h in sorted_lp_hashes]
    prefixed_titles = [f"{title_prefix}{unique_title_map[h]}" for h in sorted_title_hashes]

    tracemalloc.start()
    rss_before_mb = get_process_memory_mb()

    t_tok_start = time.perf_counter()
    lp_tok_report = active_encoder.measure_token_lengths(prefixed_lps)
    title_tok_report = active_encoder.measure_token_lengths(prefixed_titles)
    tokenization_sec = time.perf_counter() - t_tok_start

    t_lp_start = time.perf_counter()
    lp_embs = active_encoder.encode(
        prefixed_lps,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    lp_encode_sec = time.perf_counter() - t_lp_start

    t_title_start = time.perf_counter()
    title_embs = active_encoder.encode(
        prefixed_titles,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    title_encode_sec = time.perf_counter() - t_title_start

    _, tracemalloc_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_mb = get_process_memory_mb()
    peak_ws_mb = get_peak_working_set_mb()

    weight_sha_after = active_encoder.weight_sha256()
    if weight_sha_before != weight_sha_after:
        raise AssertionError("Frozen encoder weights changed during embedding extraction!")

    # Verify numerical sanity & unit norm when normalized
    if normalize_embeddings:
        lp_norms = np.linalg.norm(lp_embs, axis=1)
        t_norms = np.linalg.norm(title_embs, axis=1)
        if not (np.allclose(lp_norms, 1.0, atol=1e-3) and np.allclose(t_norms, 1.0, atol=1e-3)):
            raise AssertionError("Normalized embeddings deviate from unit L2 norm.")

    # Persist cache deterministically (NO raw Japanese strings written)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(lp_npy, lp_embs.astype(np.float32))
    np.save(title_npy, title_embs.astype(np.float32))

    lp_tok_lengths = lp_tok_report.pop("per_item_token_lengths")
    title_tok_lengths = title_tok_report.pop("per_item_token_lengths")

    lp_hashes_path.write_text(
        json.dumps({"hashes": sorted_lp_hashes, "token_lengths": lp_tok_lengths}),
        encoding="utf-8",
    )
    title_hashes_path.write_text(
        json.dumps({"hashes": sorted_title_hashes, "token_lengths": title_tok_lengths}),
        encoding="utf-8",
    )

    total_encode_sec = lp_encode_sec + title_encode_sec
    cache_size_mb = (lp_npy.stat().st_size + title_npy.stat().st_size) / (1024.0 * 1024.0)

    # Compute row-level truncation prevalence across the full DataFrame
    lp_tok_by_hash = {h: int(t) for h, t in zip(sorted_lp_hashes, lp_tok_lengths)}
    t_tok_by_hash = {h: int(t) for h, t in zip(sorted_title_hashes, title_tok_lengths)}
    row_lp_trunc = sum(
        1
        for raw_lp in df[lp_col].tolist()
        if lp_tok_by_hash[hash_normalized_text(normalize_japanese_text(raw_lp))]
        > active_encoder.max_seq_length
    )
    row_title_trunc = sum(
        1
        for raw_t in df[title_col].tolist()
        if t_tok_by_hash[hash_normalized_text(normalize_japanese_text(raw_t))]
        > active_encoder.max_seq_length
    )

    manifest: Dict[str, Any] = {
        "model_id": active_encoder.model_id,
        "model_source": f"https://huggingface.co/{active_encoder.model_id}",
        "model_license": active_encoder.license_name,
        "resolved_revision": active_encoder.resolved_revision,
        "device": active_encoder.device,
        "frozen_weights": True,
        "gradients_enabled": False,
        "weight_sha256_before": weight_sha_before,
        "weight_sha256_after": weight_sha_after,
        "weights_verified_unchanged": weight_sha_before == weight_sha_after,
        "label_independent_cache": True,
        "input_normalization": "unicode_nfkc_whitespace_collapse",
        "title_prefix": title_prefix,
        "lp_prefix": lp_prefix,
        "prefix_policy_note": prefix_policy_note,
        "max_seq_length": int(active_encoder.max_seq_length),
        "embedding_dim": int(active_encoder.embedding_dim),
        "pair_feature_dim": int(4 * active_encoder.embedding_dim + 1),
        "pair_feature_formula": "[u_lp, v_title, |u_lp - v_title|, u_lp * v_title, cosine_similarity(u_lp, v_title)]",
        "dtype": "float32",
        "l2_normalized": bool(normalize_embeddings),
        "batch_size": int(batch_size),
        "unique_lp_count": len(sorted_lp_hashes),
        "unique_title_count": len(sorted_title_hashes),
        "total_rows_indexed": len(df),
        "lp_tokenization_stats_unique": lp_tok_report,
        "title_tokenization_stats_unique": title_tok_report,
        "row_level_lp_truncated_count": int(row_lp_trunc),
        "row_level_lp_truncated_pct": round(row_lp_trunc / max(len(df), 1) * 100.0, 2),
        "row_level_title_truncated_count": int(row_title_trunc),
        "row_level_title_truncated_pct": round(row_title_trunc / max(len(df), 1) * 100.0, 2),
        "tokenization_audit_seconds": round(float(tokenization_sec), 4),
        "lp_extraction_seconds": round(float(lp_encode_sec), 4),
        "title_extraction_seconds": round(float(title_encode_sec), 4),
        "total_extraction_seconds": round(float(total_encode_sec), 4),
        "ms_per_unique_lp": round(lp_encode_sec / max(len(sorted_lp_hashes), 1) * 1000.0, 4),
        "ms_per_unique_title": round(
            title_encode_sec / max(len(sorted_title_hashes), 1) * 1000.0, 4
        ),
        "estimated_online_pair_encode_ms": round(
            (lp_encode_sec / max(len(sorted_lp_hashes), 1)
             + title_encode_sec / max(len(sorted_title_hashes), 1))
            * 1000.0,
            4,
        ),
        "cache_size_mb": round(float(cache_size_mb), 4),
        "cache_key": cache_key,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "peak_working_set_mb": peak_ws_mb,
        "python_tracemalloc_peak_mb": round(tracemalloc_peak_bytes / (1024.0 * 1024.0), 4),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "loaded_from_cache": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SemanticEmbeddingBundle(
        lp_embeddings=lp_embs,
        lp_hash_to_idx={h: i for i, h in enumerate(sorted_lp_hashes)},
        lp_token_lengths_by_hash=lp_tok_by_hash,
        title_embeddings=title_embs,
        title_hash_to_idx={h: i for i, h in enumerate(sorted_title_hashes)},
        title_token_lengths_by_hash=t_tok_by_hash,
        manifest=manifest,
        cache_dir=cache_dir,
    )


def run_semantic_smoke_test(
    sample_df: pd.DataFrame,
    model_id: str = DEFAULT_E5_MODEL_ID,
    revision: Optional[str] = DEFAULT_E5_REVISION,
    batch_size: int = 8,
    device: str = "cpu",
    encoder: Optional[TextEncoderProtocol] = None,
) -> Dict[str, Any]:
    """Execute a small technical smoke test verifying frozen E5 extraction.

    Explicitly labeled ``is_research_result: False`` — purely technical verification.
    """
    t0 = time.perf_counter()
    active_encoder: TextEncoderProtocol = (
        encoder
        if encoder is not None
        else FrozenSentenceTransformerEncoder(
            model_id=model_id,
            revision=revision,
            device=device,
        )
    )
    sha_before = active_encoder.weight_sha256()

    prefixed_lps = [
        f"{DEFAULT_E5_PREFIX}{normalize_japanese_text(x)}" for x in sample_df["lp_text"].tolist()
    ]
    prefixed_titles = [
        f"{DEFAULT_E5_PREFIX}{normalize_japanese_text(x)}" for x in sample_df["title"].tolist()
    ]

    lp_tok = active_encoder.measure_token_lengths(prefixed_lps)
    lp_tok.pop("per_item_token_lengths", None)
    t_tok = active_encoder.measure_token_lengths(prefixed_titles)
    t_tok.pop("per_item_token_lengths", None)

    u = active_encoder.encode(prefixed_lps, batch_size=batch_size, normalize_embeddings=True)
    v = active_encoder.encode(prefixed_titles, batch_size=batch_size, normalize_embeddings=True)
    pair_X = build_semantic_pair_features(u, v)
    cos = pair_X[:, -1]

    sha_after = active_encoder.weight_sha256()
    elapsed = time.perf_counter() - t0

    return {
        "is_research_result": False,
        "status": "PASSED",
        "sample_count": len(sample_df),
        "model_id": active_encoder.model_id,
        "resolved_revision": active_encoder.resolved_revision,
        "device": active_encoder.device,
        "max_seq_length": int(active_encoder.max_seq_length),
        "embedding_dim": int(active_encoder.embedding_dim),
        "pair_feature_shape": list(pair_X.shape),
        "dtype": str(pair_X.dtype),
        "all_finite": bool(np.all(np.isfinite(pair_X))),
        "u_norm_mean": round(float(np.linalg.norm(u, axis=1).mean()), 6),
        "v_norm_mean": round(float(np.linalg.norm(v, axis=1).mean()), 6),
        "cosine_min": round(float(cos.min()), 4),
        "cosine_median": round(float(np.median(cos)), 4),
        "cosine_max": round(float(cos.max()), 4),
        "weights_verified_unchanged": sha_before == sha_after,
        "lp_token_stats": lp_tok,
        "title_token_stats": t_tok,
        "elapsed_seconds": round(float(elapsed), 4),
        "rss_mb": get_process_memory_mb(),
    }
