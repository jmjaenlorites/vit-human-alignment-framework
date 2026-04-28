"""Attention rollout para ViT en JAX.

Replica la fórmula del adaptador Torch (`src/models/vit_b16.py`) pero sobre
los pesos de atención que Flax expone vía `sow_weights=True`. Snapshotea el
rollout en cada capa solicitada para mantener el contrato
`list[Array[B, grid, grid]]` de la rama Torch.
"""

import math

import jax
import jax.numpy as jnp


def rollout_per_layer(
    attn_weights: jnp.ndarray,
    target_layers: tuple[int, ...],
    head_fusion: str = "max",
    discard_ratio: float = 0.95,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Rollout para una sola imagen.

    Args:
        attn_weights: `[num_layers, num_heads, N, N]` (N = num_tokens incl. CLS).
        target_layers: índices estáticos de las capas en las que snapshotear el
            rollout acumulado.
        head_fusion: estrategia de fusión por cabeza (`"max"`, `"mean"`, `"min"`).
        discard_ratio: fracción de las atenciones más bajas que se ponen a cero
            por capa (manteniendo intactas filas/columnas del CLS).
        eps: estabilizador numérico.

    Returns:
        `[len(target_layers), grid, grid]` con el mapa CLS→patches por capa.
    """
    attn_weights = jnp.asarray(attn_weights)
    L, _, N, _ = attn_weights.shape
    grid = int(math.sqrt(N - 1))
    assert grid * grid == N - 1, f"Token count {N} no produce un grid cuadrado"

    identity = jnp.eye(N)
    protect = jnp.zeros((N, N), dtype=bool).at[0, :].set(True).at[:, 0].set(True)
    valid = int((~protect).sum())
    k = min(int(N * N * discard_ratio), valid) if discard_ratio > 0 else 0

    result = identity
    snaps: list[jnp.ndarray] = []

    for i in range(L):
        a = attn_weights[i]
        if head_fusion == "max":
            fused = a.max(axis=0)
        elif head_fusion == "mean":
            fused = a.mean(axis=0)
        elif head_fusion == "min":
            fused = a.min(axis=0)
        else:
            raise ValueError(f"Unsupported head_fusion: {head_fusion!r}")

        if k > 0:
            flat = fused.reshape(-1)
            scored = jnp.where(protect.reshape(-1), jnp.inf, flat)
            drop_idx = jnp.argsort(scored)[:k]
            flat = flat.at[drop_idx].set(0.0)
            fused = flat.reshape(N, N)

        fused = (fused + identity) * 0.5
        fused = fused / fused.sum(axis=-1, keepdims=True).clip(min=eps)
        result = fused @ result

        if i in target_layers:
            cls_to_patch = result[0, 1:].reshape(grid, grid)
            cls_to_patch = cls_to_patch / cls_to_patch.max().clip(min=eps)
            snaps.append(cls_to_patch)

    return jnp.stack(snaps, axis=0)


rollout_batch = jax.vmap(
    rollout_per_layer, in_axes=(0, None, None, None, None), out_axes=0
)
