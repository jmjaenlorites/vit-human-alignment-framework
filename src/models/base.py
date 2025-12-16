from typing import Protocol, Any, Optional, TypedDict

import torch
import jax.numpy as jnp

from ..utils.common_enums import BackendEnum
from ..utils.common_types import ArrayLike
from ..utils.common_utils import get_backend_device

class ForwardOutputs(TypedDict):
    features: Optional[ArrayLike]
    saliency: Optional[ArrayLike]

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

    def forward(self, batch: Any, layers: Optional[list[int]] = None, return_features: bool = True, return_saliency: bool = False) -> ForwardOutputs:
        """Ejecuta el modelo y retorna dict estandarizado."""
        assert return_features or return_saliency, "At least one of return_features or return_saliency must be True"
        features = self.forward_features(batch, layers) if return_features else None
        saliency = self.forward_saliency(batch, layers) if return_saliency else None
        return ForwardOutputs(features=features, saliency=saliency)

    def forward_features(self, batch: Any, layers: Optional[list[int]] = None) -> ArrayLike:
        """Ejecuta el modelo y retorna las features."""
        match self.backend:
            case BackendEnum.TORCH:
                return self.forward_features_torch(batch, layers)
            case BackendEnum.JAX:
                return self.forward_features_jax(batch, layers)
            case _:
                raise ValueError(f"Backend {self.backend} not supported")

    def forward_saliency(self, batch: Any, layers: Optional[list[int]] = None) -> ArrayLike:
        """Ejecuta el modelo y retorna el saliency."""
        match self.backend:
            case BackendEnum.TORCH:
                return self.forward_saliency_torch(batch, layers)
            case BackendEnum.JAX:
                return self.forward_saliency_jax(batch, layers)
            case _:
                raise ValueError(f"Backend {self.backend} not supported")

    def forward_features_torch(self, batch: Any, layers: Optional[list[int]] = None) -> torch.Tensor:
        """Ejecuta el modelo y retorna las features."""
        raise NotImplementedError

    def forward_saliency_torch(self, batch: Any, layers: Optional[list[int]] = None) -> torch.Tensor:
        """Ejecuta el modelo y retorna el saliency."""
        raise NotImplementedError

    def forward_features_jax(self, batch: Any, layers: Optional[list[int]] = None) -> jnp.ndarray:
        """Ejecuta el modelo y retorna las features."""
        raise NotImplementedError

    def forward_saliency_jax(self, batch: Any, layers: Optional[list[int]] = None) -> jnp.ndarray:
        """Ejecuta el modelo y retorna el saliency."""
        raise NotImplementedError

    def list_layers(self) -> list[int]:
        """Lista de índices de capas disponibles para métricas por capa."""