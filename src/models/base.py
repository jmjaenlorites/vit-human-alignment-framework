from typing import Protocol, Any, Optional

import torch
import jax.numpy as jnp

from ..utils.common_enums import BackendEnum
from ..utils.common_types import ArrayLike
from ..utils.common_utils import get_backend_device

# TODO: Definir esquema/validación para ForwardOutputs
# Considerar usar TypedDict extensible o Pydantic para validar estructuras
# Estructuras comunes:
#   - Saliency: {"features": [...], "saliency": [...]}
#   - TID: {"features_ref": [...], "features_dist": [...]}
#   - Levels: {"features_img1": [...], "features_img2": [...], "features_img3": [...]}
#   - Nights: {"features_ref": [...], "features_left": [...], "features_right": [...]}
ForwardOutputs = dict[str, Any]


class BaseModelAdapter(Protocol):
    name: str
    backend: BackendEnum
    config: dict[str, Any]

    def __init__(self, name: str, backend: BackendEnum, config: dict[str, Any]):
        self.name = name
        self.backend = backend
        self.config = config
        self.device = get_backend_device(backend)
        self.model = self._load_model(config)

    def _load_model(self, config: dict[str, Any]) -> Any:
        """Carga el modelo."""
        raise NotImplementedError

    def preprocess(self, images: ArrayLike) -> ArrayLike:
        """Normaliza y resiza al `image_size` esperado."""

    def forward(
        self,
        batch: Any,
        layers: Optional[list[int]] = None,
        return_features: bool = True,
        return_saliency: bool = False,
    ) -> ForwardOutputs:
        """Ejecuta el modelo y retorna dict estandarizado."""
        assert return_features or return_saliency, (
            "At least one of return_features or return_saliency must be True"
        )
        features = self.forward_features(batch, layers) if return_features else None
        saliency = self.forward_saliency(batch, layers) if return_saliency else None
        return {"features": features, "saliency": saliency}

    def forward_features(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> ArrayLike:
        """Ejecuta el modelo y retorna las features.

        Si `layers` es None, los adaptadores deben devolver todas las capas
        disponibles (incluyendo embeddings iniciales y bloques) para métricas
        por capa.
        """
        match self.backend:
            case BackendEnum.TORCH:
                return self.forward_features_torch(batch, layers)
            case BackendEnum.JAX:
                return self.forward_features_jax(batch, layers)
            case _:
                raise ValueError(f"Backend {self.backend} not supported")

    def forward_saliency(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> ArrayLike:
        """Ejecuta el modelo y retorna el saliency.
        
        Returns:
            Para modelos con attention rollout (como ViT): lista de mapas de saliency,
            uno por cada capa solicitada. Cada mapa tiene shape [B, H, W].
            Para otros modelos: puede ser un único tensor o estructura apropiada.
        """
        match self.backend:
            case BackendEnum.TORCH:
                return self.forward_saliency_torch(batch, layers)
            case BackendEnum.JAX:
                return self.forward_saliency_jax(batch, layers)
            case _:
                raise ValueError(f"Backend {self.backend} not supported")

    def forward_features_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> ArrayLike:
        """Ejecuta el modelo y retorna las features."""
        raise NotImplementedError

    def forward_saliency_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> torch.Tensor | list[torch.Tensor]:
        """Ejecuta el modelo y retorna el saliency.
        
        Returns:
            Puede ser un único tensor o una lista de tensors (para rollout por capa).
        """
        raise NotImplementedError

    def forward_features_jax(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> jnp.ndarray:
        """Ejecuta el modelo y retorna las features."""
        raise NotImplementedError

    def forward_saliency_jax(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> jnp.ndarray:
        """Ejecuta el modelo y retorna el saliency."""
        raise NotImplementedError

    def list_layers(self) -> list[int]:
        """Lista de índices de capas disponibles para métricas por capa."""
