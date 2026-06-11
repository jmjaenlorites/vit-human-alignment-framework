from typing import Any, Optional

import jax.numpy as jnp
import torch
import torchvision

from ..utils.common_enums import BackendEnum
from ..utils.common_types import ArrayLike
from .base import BaseModelAdapter, ForwardOutputs

SALIENCY_BASELINE_MODEL_PREFIX = "saliency-baseline::"
SALIENCY_BASELINE_UNIFORM = "uniform"
SALIENCY_BASELINE_CENTER_GAUSSIAN = "center-gaussian"
SALIENCY_BASELINE_MODEL_NAMES = (
    SALIENCY_BASELINE_UNIFORM,
    SALIENCY_BASELINE_CENTER_GAUSSIAN,
)
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 16
DEFAULT_NUM_LAYERS = 12
DEFAULT_CENTER_GAUSSIAN_SIGMA = 0.35
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def list_saliency_baseline_model_options() -> list[str]:
    return [
        f"{SALIENCY_BASELINE_MODEL_PREFIX}{baseline_name}"
        for baseline_name in SALIENCY_BASELINE_MODEL_NAMES
    ]


class SaliencyBaselineModelAdapter(BaseModelAdapter):
    """Adaptador determinista para baselines saliency-only."""

    def __init__(
        self,
        baseline_name: str,
        config: Optional[dict[str, Any]] = None,
    ):
        if baseline_name not in SALIENCY_BASELINE_MODEL_NAMES:
            raise ValueError(
                f"Saliency baseline {baseline_name} not supported. Valid options: "
                f"{', '.join(SALIENCY_BASELINE_MODEL_NAMES)}"
            )

        config = config or {}
        self.baseline_name = baseline_name
        self.image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))
        self.patch_size = int(config.get("patch_size", DEFAULT_PATCH_SIZE))
        self.num_layers = int(config.get("num_layers", DEFAULT_NUM_LAYERS))
        self.center_gaussian_sigma = float(
            config.get("center_gaussian_sigma", DEFAULT_CENTER_GAUSSIAN_SIGMA)
        )
        self._validate_config()

        super().__init__(
            f"{SALIENCY_BASELINE_MODEL_PREFIX}{baseline_name}",
            BackendEnum.TORCH,
            config,
        )
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (self.image_size, self.image_size),
                    antialias=True,
                ),
                torchvision.transforms.Normalize(
                    mean=IMAGENET_DEFAULT_MEAN,
                    std=IMAGENET_DEFAULT_STD,
                ),
            ]
        )

    def _validate_config(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.center_gaussian_sigma <= 0:
            raise ValueError("center_gaussian_sigma must be positive")

    def _load_model(self, config: dict[str, Any]) -> None:
        return None

    def forward(
        self,
        batch: Any,
        layers: Optional[list[int]] = None,
        return_features: bool = True,
        return_saliency: bool = False,
    ) -> ForwardOutputs:
        assert return_features or return_saliency, (
            "At least one of return_features or return_saliency must be True"
        )
        if return_features:
            raise NotImplementedError(
                f"{self.name} is saliency-only and does not support feature extraction"
            )
        saliency = self.forward_saliency(batch, layers) if return_saliency else None
        return {
            "features": None,
            "saliency": saliency,
        }

    def forward_features_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> ArrayLike:
        raise NotImplementedError(
            f"{self.name} is saliency-only and does not support feature extraction"
        )

    def forward_saliency_torch(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> list[torch.Tensor]:
        if not isinstance(batch, torch.Tensor):
            raise ValueError(
                f"{self.name} expects a torch.Tensor batch with shape [B, C, H, W]"
            )
        if batch.dim() != 4:
            raise ValueError(
                f"{self.name} expects a 4D batch with shape [B, C, H, W], "
                f"got shape {tuple(batch.shape)}"
            )

        x = batch.to(self.device)
        batch_size, _, height, width = x.shape
        grid_height, grid_width = self._grid_shape(height, width)
        target_layers = self._resolve_layers(layers)
        dtype = x.dtype if torch.is_floating_point(x) else torch.float32
        baseline_map = self._baseline_map(
            grid_height,
            grid_width,
            device=x.device,
            dtype=dtype,
        )
        batch_map = baseline_map.unsqueeze(0).expand(batch_size, -1, -1)
        return [batch_map.clone() for _ in target_layers]

    def forward_features_jax(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> jnp.ndarray:
        raise NotImplementedError(f"{self.name} only supports the torch backend")

    def forward_saliency_jax(
        self, batch: Any, layers: Optional[list[int]] = None
    ) -> jnp.ndarray:
        raise NotImplementedError(f"{self.name} only supports the torch backend")

    def list_layers(self) -> list[int]:
        return list(range(self.num_layers))

    def _grid_shape(self, height: int, width: int) -> tuple[int, int]:
        return (
            max(1, height // self.patch_size),
            max(1, width // self.patch_size),
        )

    def _resolve_layers(self, layers: Optional[list[int]]) -> list[int]:
        available_layers = self.list_layers()
        if layers is None:
            return available_layers

        invalid_layers = sorted(set(layers) - set(available_layers))
        if invalid_layers:
            raise ValueError(
                f"Layers {invalid_layers} not supported for {self.name}. "
                f"Valid options: {available_layers}"
            )
        return layers

    def _baseline_map(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        match self.baseline_name:
            case "uniform":
                return torch.ones(height, width, device=device, dtype=dtype)
            case "center-gaussian":
                return self._center_gaussian_map(height, width, device, dtype)
            case _:
                raise ValueError(
                    f"Unsupported saliency baseline: {self.baseline_name}"
                )

    def _center_gaussian_map(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        squared_distance = xx.square() + yy.square()
        gaussian = torch.exp(
            -squared_distance / (2 * self.center_gaussian_sigma**2)
        )
        return gaussian / gaussian.amax().clamp_min(torch.finfo(dtype).eps)
