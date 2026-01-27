from typing import Any, Optional

import torch
import timm

from ..utils.common_enums import BackendEnum
from ..models.base import BaseModelAdapter
from ..utils.common_types import ArrayLike




class ViT_B_16(BaseModelAdapter):
    MODEL_NAME = "vit_base_patch16_clip_224.laion2b"

    def __init__(self, config: dict[str, Any] = {}):
        super().__init__("ViT-B-16", BackendEnum.TORCH, config)

        self.transform = timm.data.create_transform(
            **timm.data.resolve_data_config({}, model=self.MODEL_NAME),
            is_training=False,
        )

    def _load_model(self, config: dict[str, Any]) -> Any:
        model = timm.create_model(self.MODEL_NAME, pretrained=True)
        return model.to(self.device)

    def _forward_hidden_states_torch(self, batch: Any) -> list[torch.Tensor]:
        with torch.no_grad():
            x = batch.to(self.device)
            model = self.model

            if hasattr(model, "patch_embed"):
                x = model.patch_embed(x)
            if hasattr(model, "_pos_embed"):
                x = model._pos_embed(x)
            else:
                if hasattr(model, "cls_token"):
                    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat((cls_token, x), dim=1)
                if hasattr(model, "pos_embed"):
                    x = x + model.pos_embed
                if hasattr(model, "pos_drop"):
                    x = model.pos_drop(x)

            if hasattr(model, "norm_pre"):
                x = model.norm_pre(x)

            hidden_states = [x]
            for block in model.blocks:
                x = block(x)
                hidden_states.append(x)

            if hasattr(model, "norm"):
                x = model.norm(x)
                hidden_states[-1] = x

            return hidden_states

    def forward_features_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> ArrayLike:
        """Ejecuta el modelo y retorna las features por capa.

        Por defecto devuelve todas las capas (embeddings + bloques).
        """
        hidden_states = self._forward_hidden_states_torch(batch)
        if layers is None:
            return hidden_states
        if not layers:
            return []
        max_index = len(hidden_states) - 1
        if any(layer < 0 or layer > max_index for layer in layers):
            raise ValueError(f"layers must be within [0, {max_index}], got {layers}")
        return [hidden_states[layer] for layer in layers]

    def forward_saliency_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> torch.Tensor:
        """Ejecuta el modelo y retorna el saliency."""
        with torch.no_grad():
            results = self.model(batch[0].to(self.device))
        return results
