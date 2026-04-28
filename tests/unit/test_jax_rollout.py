"""Tests numéricos para `rollout_per_layer`.

Compara la implementación JAX con una referencia numpy hand-rolled construida
a partir de la misma fórmula. Todo en CPU, sin descargas ni modelos reales.
"""

import math

import numpy as np

from src.models.jax.rollout import rollout_per_layer


def _numpy_rollout_reference(
    attn: np.ndarray,
    target_layers: tuple[int, ...],
    head_fusion: str = "max",
    discard_ratio: float = 0.95,
    eps: float = 1e-8,
) -> np.ndarray:
    L, H, N, _ = attn.shape
    grid = int(math.sqrt(N - 1))
    identity = np.eye(N)
    protect = np.zeros((N, N), dtype=bool)
    protect[0, :] = True
    protect[:, 0] = True
    valid = int((~protect).sum())
    k = min(int(N * N * discard_ratio), valid) if discard_ratio > 0 else 0

    result = identity.copy()
    snaps = []
    for i in range(L):
        a = attn[i]
        if head_fusion == "max":
            fused = a.max(axis=0)
        elif head_fusion == "mean":
            fused = a.mean(axis=0)
        else:
            fused = a.min(axis=0)

        if k > 0:
            flat = fused.reshape(-1).copy()
            scored = np.where(protect.reshape(-1), np.inf, flat)
            drop_idx = np.argsort(scored)[:k]
            flat[drop_idx] = 0.0
            fused = flat.reshape(N, N)

        fused = (fused + identity) * 0.5
        fused = fused / np.clip(fused.sum(axis=-1, keepdims=True), eps, None)
        result = fused @ result

        if i in target_layers:
            cls_to_patch = result[0, 1:].reshape(grid, grid)
            cls_to_patch = cls_to_patch / max(cls_to_patch.max(), eps)
            snaps.append(cls_to_patch)

    return np.stack(snaps, axis=0)


def _make_synthetic_attn(L: int = 3, H: int = 3, grid: int = 2) -> np.ndarray:
    rng = np.random.default_rng(42)
    N = grid * grid + 1
    raw = rng.uniform(0.0, 1.0, size=(L, H, N, N)).astype(np.float32)
    # Normalizar como atenciones reales (filas suman 1).
    return raw / raw.sum(axis=-1, keepdims=True)


def test_rollout_per_layer_matches_numpy_reference_max() -> None:
    attn = _make_synthetic_attn(L=4, H=3, grid=3)
    target = (0, 1, 2, 3)
    jax_out = np.asarray(
        rollout_per_layer(
            attn, target_layers=target, head_fusion="max", discard_ratio=0.5
        )
    )
    np_out = _numpy_rollout_reference(
        attn, target_layers=target, head_fusion="max", discard_ratio=0.5
    )
    np.testing.assert_allclose(jax_out, np_out, rtol=1e-5, atol=1e-6)


def test_rollout_per_layer_matches_numpy_reference_mean_no_discard() -> None:
    attn = _make_synthetic_attn(L=2, H=3, grid=2)
    target = (0, 1)
    jax_out = np.asarray(
        rollout_per_layer(
            attn, target_layers=target, head_fusion="mean", discard_ratio=0.0
        )
    )
    np_out = _numpy_rollout_reference(
        attn, target_layers=target, head_fusion="mean", discard_ratio=0.0
    )
    np.testing.assert_allclose(jax_out, np_out, rtol=1e-5, atol=1e-6)


def test_rollout_per_layer_subset_layers() -> None:
    attn = _make_synthetic_attn(L=4, H=2, grid=2)
    jax_out = np.asarray(rollout_per_layer(attn, target_layers=(1, 3)))
    assert jax_out.shape == (2, 2, 2)


def test_rollout_per_layer_output_normalized_by_max() -> None:
    attn = _make_synthetic_attn(L=2, H=2, grid=2)
    out = np.asarray(rollout_per_layer(attn, target_layers=(0, 1)))
    for layer_map in out:
        assert layer_map.max() <= 1.0 + 1e-6
        assert layer_map.min() >= 0.0
