from typing import Any, Optional

import torch
import timm

from ..utils.common_enums import BackendEnum
from ..models.base import BaseModelAdapter
from ..utils.common_types import ArrayLike


class TimmViTAdapter(BaseModelAdapter):
    """Clase base para adaptadores de Vision Transformers usando timm.
    
    Usa get_intermediate_layers de timm para extraer features intermedias
    de forma consistente y mantenible.
    
    Subclases solo necesitan definir:
        - MODEL_NAME: nombre del modelo en timm
        - DISPLAY_NAME: nombre para mostrar/logs
    """
    MODEL_NAME: str  # Debe ser definido por subclases
    DISPLAY_NAME: str  # Debe ser definido por subclases

    def __init__(self, config: dict[str, Any] = {}):
        super().__init__(self.DISPLAY_NAME, BackendEnum.TORCH, config)
        self.transform = timm.data.create_transform(
            **timm.data.resolve_data_config({}, model=self.MODEL_NAME),
            is_training=False,
        )

    def _load_model(self, config: dict[str, Any]) -> Any:
        model = timm.create_model(self.MODEL_NAME, pretrained=True)
        model.eval()
        return model.to(self.device)

    def forward_features_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> ArrayLike:
        """Ejecuta el modelo y retorna las features por capa.

        Usa get_intermediate_layers con reshape=False y return_prefix_tokens=True
        para obtener formato [B, num_tokens, hidden_dim] (CLS + patches).
        
        Args:
            batch: Tensor de entrada [B, C, H, W]
            layers: Lista de índices de capas a extraer. None = todas.
            
        Returns:
            Lista de tensores [B, num_tokens, hidden_dim] por cada capa.
        """
        with torch.no_grad():
            x = batch.to(self.device)
            
            # Determinar qué capas extraer (por defecto todas)
            num_blocks = len(self.model.blocks)
            indices = list(range(num_blocks)) if layers is None else layers
            
            # Usar API nativa de timm
            intermediates = self.model.get_intermediate_layers(
                x,
                indices,
                reshape=False,              # Mantener formato [B, N, C]
                return_prefix_tokens=True,  # Devolver CLS token separado
                norm=False,                 # Sin normalización en capas intermedias
            )
            
            # Concatenar [CLS, patches] -> [B, num_tokens, hidden_dim]
            hidden_states = [
                torch.cat([prefix, spatial], dim=1)
                for spatial, prefix in intermediates
            ]
            
            return hidden_states

    def forward_saliency_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> torch.Tensor:
        """Ejecuta el modelo y retorna la salida final."""
        with torch.no_grad():
            return self.model(batch.to(self.device))


# =============================================================================
# Modelos específicos - Solo definen MODEL_NAME y DISPLAY_NAME
# =============================================================================

class ViT_B_16(TimmViTAdapter):
    """ViT-Base/16 entrenado con CLIP en LAION-2B."""
    MODEL_NAME = "vit_base_patch16_clip_224.laion2b"
    DISPLAY_NAME = "ViT-B/16"

class ViT_B_32(TimmViTAdapter):
    """ViT-Base/32 entrenado con CLIP en LAION-2B."""
    MODEL_NAME = "vit_base_patch32_clip_224.laion2b"
    DISPLAY_NAME = "ViT-B/32"

class ViT_L_14(TimmViTAdapter):
    """ViT-Large/14 entrenado con CLIP en LAION-2B."""
    MODEL_NAME = "vit_large_patch14_clip_224.laion2b"
    DISPLAY_NAME = "ViT-L/14"


class ViT_H_14(TimmViTAdapter):
    """ViT-Huge/14 entrenado con CLIP en LAION-2B."""
    MODEL_NAME = "vit_huge_patch14_clip_224.laion2b"
    DISPLAY_NAME = "ViT-H/14"
