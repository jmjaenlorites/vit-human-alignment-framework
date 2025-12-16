from typing import Protocol, Literal, Callable, Optional
from typing import Any

import torch
import jax.numpy as jnp
import torchvision
import numpy as np

from ..models.base import BaseModelAdapter, ForwardOutputs
from ..utils.common_enums import BackendEnum
from ..dataset_loaders.base import BaseDatasetLoader

class BaseMetric(Protocol):
    name: str
    type: Literal["saliency", "perceptual"]
    _backend: BackendEnum
    def __init__(self, backend: BackendEnum):
        self._backend = backend
    def calculate(self, batch: Any, batch_results: ForwardOutputs) -> list[float]:
        raise NotImplementedError

class BaseSaliencyMetric(BaseMetric):
    type: Literal["saliency"] = "saliency"
    def __init__(self, backend: BackendEnum):
        super().__init__(backend)

    def calculate(self, batch: Any, batch_results: ForwardOutputs) -> list[float]:
        saliency_map_size = batch_results["saliency"][0].shape

        # TODO: Implementar el rollout attention en lugar de este producto de matrices
        predicted_saliency_maps = batch_results["saliency"][..., None].transpose(-2, -1) @ batch_results["saliency"][..., None]


        ground_truth_saliency_maps = torchvision.transforms.Resize(saliency_map_size)(batch[1])
        ground_truth_fixation_maps = torchvision.transforms.Resize(saliency_map_size)(batch[2])

        batch_metrics = []
        match self._backend:
            case BackendEnum.TORCH:
                for predicted_saliency_map, ground_truth_saliency_map, ground_truth_fixation_map in zip(predicted_saliency_maps, ground_truth_saliency_maps, ground_truth_fixation_maps):
                    batch_metrics.append(self._calculate_torch(predicted_saliency_map, ground_truth_saliency_map, ground_truth_fixation_map))
            case BackendEnum.JAX:
                for predicted_saliency_map, ground_truth_saliency_map, ground_truth_fixation_map in zip(predicted_saliency_maps, ground_truth_saliency_maps, ground_truth_fixation_maps):
                    batch_metrics.append(self._calculate_jax(predicted_saliency_map, ground_truth_saliency_map, ground_truth_fixation_map))
        return batch_metrics

    def _calculate_torch(self, pred_map: torch.Tensor, ground_truth_saliency_map: torch.Tensor, ground_truth_fixation_map: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _calculate_jax(self, pred_map: jnp.ndarray, ground_truth_saliency_map: jnp.ndarray, ground_truth_fixation_map: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class BaseMetricCalculator(Protocol):
    name: str
    _metrics: list[BaseMetric]
    _backend: BackendEnum
    _return_saliency: bool = False
    _return_features: bool = False

    def get_dataset_loader(self, transform: Optional[Callable[[Any], Any]] = None) -> BaseDatasetLoader:
        """Retorna un dataset loader para el dataset interno."""

    def run(self, model: BaseModelAdapter) -> dict[str, Any]:
        """Ejecuta evaluación completa sobre el dataset interno y retorna un diccionario con los resultados de cada métrica."""
        assert model.backend == self._backend, f"Model backend {model.backend} must match calculator backend {self._backend}"
        assert len(self._metrics) > 0, "At least one metric is required"
        assert all(metric._backend == self._backend for metric in self._metrics), f"All metrics must have backend {self._backend}"

        dataset_loader = self.get_dataset_loader(transform=model.transform)
        metric_results: dict[str, list[Any]] = {}

        for batch in dataset_loader.get_iterator():
            # batch will be a tuple of unknown length depending on the dataset loader, that the model will handle
            # and return a tuple of unknown length depending on the model
            batch_results = model.forward(batch, return_features=self._return_features, return_saliency=self._return_saliency)

            for metric in self._metrics:
                if metric.name not in metric_results:
                    metric_results[metric.name] = []
                metric_results[metric.name].extend(metric.calculate(batch, batch_results))

        return {k: np.mean(v) for k, v in metric_results.items()}