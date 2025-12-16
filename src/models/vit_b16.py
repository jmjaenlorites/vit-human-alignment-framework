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
            is_training=False
        )

    def _load_model(self, config: dict[str, Any]) -> Any:
        model = timm.create_model(self.MODEL_NAME, pretrained=True)
        return model

    def forward_features_torch(self, batch: Any, layers: Optional[list[int]] = None) -> torch.Tensor:
        """Ejecuta el modelo y retorna las features."""
        return self.model(batch)

    def forward_saliency_torch(self, batch: Any, layers: Optional[list[int]] = None) -> torch.Tensor:
        """Ejecuta el modelo y retorna el saliency."""
        with torch.no_grad():
            results = self.model(batch[0].to(self.device))
        return results