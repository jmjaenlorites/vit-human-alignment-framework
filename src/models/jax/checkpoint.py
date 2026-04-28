"""Carga y adaptación de checkpoints `.npz` publicados por Google (AugReg)."""

import collections
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage

from ..checkpoint_cache import get_checkpoint_cache_dir

logger = logging.getLogger(__name__)


def download_npz(url: str) -> Path:
    """Descarga `url` al cache de checkpoints (idempotente). Retorna ruta local."""
    filename = os.path.basename(url)
    if not filename.endswith(".npz"):
        raise ValueError(f"Expected .npz URL, got: {url!r}")

    cache_dir = Path(get_checkpoint_cache_dir())
    dst = cache_dir / filename
    if dst.exists():
        return dst

    logger.info("Downloading checkpoint %s -> %s", url, dst)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)
    return dst


def _recover_tree(keys: list[str], values: list[np.ndarray]) -> dict[str, Any]:
    """Convierte claves planas separadas por '/' en un dict anidado."""
    tree: dict[str, Any] = {}
    sub_trees: dict[str, list[tuple[str, np.ndarray]]] = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            head, tail = k.split("/", 1)
            sub_trees[head].append((tail, v))
    for head, kv in sub_trees.items():
        sub_keys, sub_values = zip(*kv)
        tree[head] = _recover_tree(list(sub_keys), list(sub_values))
    return tree


def load_npz_params(path: Path) -> dict[str, Any]:
    """Carga un `.npz` publicado por Google y reconstruye el árbol de parámetros."""
    with open(path, "rb") as f:
        ckpt = np.load(f, allow_pickle=False)
        keys = list(ckpt.keys())
        values = [ckpt[k] for k in keys]
    return _recover_tree(keys, values)


def interpolate_posembed(
    posemb: np.ndarray,
    num_tokens: int,
    has_class_token: bool,
) -> np.ndarray:
    """Re-escala los positional embeddings al tamaño esperado por el modelo init.

    Args:
        posemb: tensor (1, N_old, dim).
        num_tokens: tokens objetivo (incluye CLS si has_class_token).
        has_class_token: si el primer token es CLS (no se interpola, se preserva).

    Returns:
        Tensor (1, num_tokens, dim).
    """
    assert posemb.shape[0] == 1, posemb.shape
    if has_class_token:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        num_tokens -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0, 0:]

    gs_old = int(np.sqrt(len(posemb_grid)))
    gs_new = int(np.sqrt(num_tokens))
    assert gs_old**2 == len(posemb_grid), (
        f"Expected square grid, got {gs_old**2} vs {len(posemb_grid)}"
    )
    assert gs_new**2 == num_tokens, (
        f"Expected square target grid, got {gs_new**2} vs {num_tokens}"
    )
    if gs_old == gs_new:
        return posemb

    logger.info("interpolate_posembed: grid %s -> %s", gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
    zoom = (gs_new / gs_old, gs_new / gs_old, 1)
    posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
    posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
    return np.concatenate([posemb_tok, posemb_grid], axis=1)


def adapt_pretrained_params(
    restored: dict[str, Any],
    init_params: dict[str, Any],
    has_class_token: bool = True,
) -> dict[str, Any]:
    """Reconcilia el árbol de un checkpoint con el `init_params` del modelo Flax.

    - Re-escala `Transformer.posembed_input.pos_embedding` si hay mismatch.
    - Conserva las claves del checkpoint cuando coinciden con el init.
    - Para claves presentes solo en init (p.ej. `head` con num_classes distinto)
      se conserva el init (la cabeza no se usa para inferencia de features).
    - Para claves presentes solo en restored, se descartan (warning).
    """
    out = _deep_merge_init_with_restored(init_params, restored)

    posembed = (
        out.get("Transformer", {})
        .get("posembed_input", {})
        .get("pos_embedding")
    )
    expected = (
        init_params.get("Transformer", {})
        .get("posembed_input", {})
        .get("pos_embedding")
    )
    if posembed is not None and expected is not None and posembed.shape != expected.shape:
        new_posembed = interpolate_posembed(
            np.asarray(posembed),
            num_tokens=expected.shape[1],
            has_class_token=has_class_token,
        )
        out["Transformer"]["posembed_input"]["pos_embedding"] = new_posembed

    return out


def _deep_merge_init_with_restored(
    init: dict[str, Any], restored: dict[str, Any]
) -> dict[str, Any]:
    """Para cada hoja en `init`, usa el valor restored si existe y la forma encaja.
    Si no existe, mantiene el init. Si la forma no encaja en una hoja, mantiene
    el init (caso típico: `head` con num_classes distinto).
    """
    merged: dict[str, Any] = {}
    for key, init_value in init.items():
        if isinstance(init_value, dict):
            sub_restored = restored.get(key, {})
            if not isinstance(sub_restored, dict):
                merged[key] = init_value
            else:
                merged[key] = _deep_merge_init_with_restored(init_value, sub_restored)
        else:
            restored_value = restored.get(key)
            if restored_value is None:
                merged[key] = init_value
            else:
                init_shape = getattr(init_value, "shape", None)
                restored_shape = getattr(restored_value, "shape", None)
                if (
                    init_shape is not None
                    and restored_shape is not None
                    and init_shape != restored_shape
                    and key != "pos_embedding"  # se reescala después
                ):
                    logger.info(
                        "Keeping init for key %s (shape mismatch %s vs %s)",
                        key,
                        init_shape,
                        restored_shape,
                    )
                    merged[key] = init_value
                else:
                    merged[key] = restored_value
    return merged
