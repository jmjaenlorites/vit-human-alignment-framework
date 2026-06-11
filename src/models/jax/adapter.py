"""Adaptador `BaseModelAdapter` para los Vision Transformers de Google (Flax/JAX).

Carga la arquitectura `VisionTransformer` con los nombres de submódulos que usa
el repo oficial, descarga el checkpoint `.npz` desde el bucket público de
AugReg y expone:

- `forward_features_jax`: lista de tensores `[B, num_tokens, hidden]` por capa
  (CLS + patches), idéntico contrato al adaptador Torch.
- `forward_saliency_jax`: lista de mapas de attention rollout por capa,
  `[B, grid, grid]`, replicando la fórmula de `vit_b16.py`.

Los loaders del framework siguen siendo Torch; este adaptador recibe
`torch.Tensor` en `forward_*_jax` y los convierte a `jnp.ndarray` (NHWC) en el
boundary. `self.transform` permanece como `torchvision.transforms.Compose`
para no romper la introspección de `SaliencyMIT1003Dataset`.
"""

import logging
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ..base import BaseModelAdapter
from ...utils.common_enums import BackendEnum
from .augreg_index import GoogleModelInfo, lookup_google_model
from .checkpoint import adapt_pretrained_params, download_npz, load_npz_params
from .models_vit import VisionTransformer
from .rollout import rollout_batch
from .transforms import make_torchvision_transform

logger = logging.getLogger(__name__)


GOOGLE_MODEL_PREFIX = "google::"


class DynamicGoogleViTAdapter(BaseModelAdapter):
    """Adaptador para cualquier ViT del index AugReg (filename pretrain o adapt)."""

    def __init__(self, filename: str, config: Optional[dict[str, Any]] = None):
        self._filename = filename
        self._info: GoogleModelInfo = lookup_google_model(filename)
        self.MODEL_NAME = f"{GOOGLE_MODEL_PREFIX}{filename}"
        self.DISPLAY_NAME = self.MODEL_NAME

        super().__init__(self.DISPLAY_NAME, BackendEnum.JAX, config or {})

        self.transform = make_torchvision_transform(self._info.image_size)
        self.head_fusion: str = "max"
        self.discard_ratio: float = 0.95
        self._EPS: float = 1e-8

    # ------------------------------------------------------------------
    # Inicialización
    # ------------------------------------------------------------------
    def _load_model(self, config: dict[str, Any]) -> dict[str, Any]:
        info = self._info
        flax_model = VisionTransformer(
            num_classes=info.num_classes,
            patches_size=(info.patch_size, info.patch_size),
            hidden_size=info.hidden_size,
            num_layers=info.num_layers,
            num_heads=info.num_heads,
            mlp_dim=info.mlp_dim,
            classifier="token",
        )
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((1, info.image_size, info.image_size, 3))
        init = flax_model.init(rng, dummy, train=False)
        init_params = init["params"]

        ckpt_path = download_npz(info.checkpoint_url)
        restored = load_npz_params(ckpt_path)

        # Los `.npz` envuelven el árbol bajo distintas raíces según versión:
        # algunos exponen `params/...`, la mayoría AugReg expone los keys al
        # nivel raíz. Nos quedamos con el subtree que contenga `Transformer`.
        if "Transformer" not in restored and "params" in restored:
            restored = restored["params"]

        params = adapt_pretrained_params(restored, init_params, has_class_token=True)

        return {
            "apply_fn": flax_model.apply,
            "params": params,
            "image_size": info.image_size,
            "num_layers": info.num_layers,
        }

    # ------------------------------------------------------------------
    # Conversiones torch <-> jax
    # ------------------------------------------------------------------
    def _to_nhwc(self, batch: Any) -> jnp.ndarray:
        if isinstance(batch, torch.Tensor):
            arr = batch.detach().cpu().numpy()
        else:
            arr = np.asarray(batch)
        if arr.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {arr.shape}")
        # NCHW -> NHWC
        arr = np.transpose(arr, (0, 2, 3, 1))
        return jnp.asarray(arr)

    def _apply_features(self, x: jnp.ndarray):
        """Forward que solo captura `block_out` (colección `features`).

        Las MHA siguen ejecutando, pero su `sow_weights=True` apunta a
        `intermediates`, que NO está en `mutable` — por tanto no se materializa
        el árbol de attention. Para B/8 batch=8 ahorra ~3.4 GB por forward.
        """
        return self.model["apply_fn"](
            {"params": self.model["params"]},
            x,
            train=False,
            mutable=["features"],
        )

    def _apply_saliency(self, x: jnp.ndarray):
        """Forward que solo captura `attention_weights` (colección `intermediates`).

        El sow de `block_out` apunta a `features`, que NO está en `mutable`.
        Solo se mantiene el árbol de attention que el rollout consume.
        """
        return self.model["apply_fn"](
            {"params": self.model["params"]},
            x,
            train=False,
            mutable=["intermediates"],
        )

    # ------------------------------------------------------------------
    # Forward features / saliency
    # ------------------------------------------------------------------
    def forward_features_jax(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> list[jnp.ndarray]:
        x = self._to_nhwc(batch)
        _, state = self._apply_features(x)
        block_outs = state["features"]["Transformer"]["block_out"]
        if not isinstance(block_outs, tuple):
            block_outs = tuple(block_outs)
        num_layers = self.model["num_layers"]
        if len(block_outs) != num_layers:
            raise RuntimeError(
                f"Expected {num_layers} block outputs, got {len(block_outs)}"
            )
        idxs = list(range(num_layers)) if layers is None else list(layers)
        return [block_outs[i] for i in idxs]

    def forward_saliency_jax(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> list[jnp.ndarray]:
        x = self._to_nhwc(batch)
        _, state = self._apply_saliency(x)

        num_layers = self.model["num_layers"]
        encoder_state = state["intermediates"]["Transformer"]
        attn_per_layer = []
        for i in range(num_layers):
            block_state = encoder_state[f"encoderblock_{i}"]
            mha_state = block_state["MultiHeadDotProductAttention_0"]
            attn = mha_state["attention_weights"]
            # `sow_weights=True` registra una tupla (mismo patrón que `block_out`).
            if isinstance(attn, tuple):
                attn = attn[0]
            attn_per_layer.append(attn)

        # [num_layers, B, H, N, N] -> [B, num_layers, H, N, N]
        attn = jnp.stack(attn_per_layer, axis=1)

        target = (
            tuple(range(num_layers)) if layers is None else tuple(layers)
        )
        rolled = rollout_batch(
            attn, target, self.head_fusion, self.discard_ratio, self._EPS
        )
        # rolled: [B, len(target), grid, grid]
        return [rolled[:, i] for i in range(rolled.shape[1])]

    def list_layers(self) -> list[int]:
        return list(range(self.model["num_layers"]))
