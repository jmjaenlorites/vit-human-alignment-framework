from typing import Protocol, Literal, Callable, Optional
from typing import Any

import torch
import jax.numpy as jnp
import torchvision
import numpy as np

from ..models.base import BaseModelAdapter, ForwardOutputs
from ..utils.common_enums import BackendEnum
from ..dataset_loaders.base import BaseDatasetLoader

# TODO: Definir esquema/validación para los outputs de process_batch()
# Los calculators devuelven dicts con diferentes estructuras según el tipo de métrica:
#   - Saliency: {"features": [...], "saliency": [...]}
#   - TID: {"features_ref": [...], "features_dist": [...]}
#   - Levels: {"features_img1": [...], "features_img2": [...], "features_img3": [...]}
#   - Nights: {"features_ref": [...], "features_left": [...], "features_right": [...]}
# Considerar usar Pydantic, dataclasses con validación, o Protocol para cada tipo


class BaseMetric(Protocol):
    name: str
    type: Literal["saliency", "perceptual"]
    _backend: BackendEnum

    def __init__(self, backend: BackendEnum):
        self._backend = backend

    def calculate(self, batch: Any, batch_results: ForwardOutputs) -> None:
        """
        Procesa un batch y acumula estado interno para calcular métricas finales.
        No retorna nada; los resultados se obtienen llamando a finalize().

        Args:
            batch: Batch de datos del dataset loader
            batch_results: Dict con resultados del modelo (estructura varía según calculator)
        """
        raise NotImplementedError

    def finalize(self) -> dict[str, Any]:
        """
        Calcula métricas finales después de procesar todo el dataset.
        Retorna un dict con {metric_name: value}, donde value puede ser escalar o array serializado.
        Este método es obligatorio y debe ser implementado por todas las métricas.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reinicia el estado interno de la métrica."""
        raise NotImplementedError


class BaseSaliencyMetric(BaseMetric):
    type: Literal["saliency"] = "saliency"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)
        self.accumulated_values: list[float] = []

    def calculate(self, batch: Any, batch_results: ForwardOutputs) -> None:
        """
        Acumula valores de métricas de saliency por batch.

        Args:
            batch: Tupla (stimulus, saliency_gt, fixation_gt)
            batch_results: Dict con {"features": [...], "saliency": [...]}
        """
        saliency_map_size = batch_results["saliency"][0].shape

        # TODO: Implementar el rollout attention en lugar de este producto de matrices
        predicted_saliency_maps = (
            batch_results["saliency"][..., None].transpose(-2, -1)
            @ batch_results["saliency"][..., None]
        )

        ground_truth_saliency_maps = torchvision.transforms.Resize(saliency_map_size)(
            batch[1]
        )
        ground_truth_fixation_maps = torchvision.transforms.Resize(saliency_map_size)(
            batch[2]
        )

        match self._backend:
            case BackendEnum.TORCH:
                for (
                    predicted_saliency_map,
                    ground_truth_saliency_map,
                    ground_truth_fixation_map,
                ) in zip(
                    predicted_saliency_maps,
                    ground_truth_saliency_maps,
                    ground_truth_fixation_maps,
                ):
                    device = predicted_saliency_map.device
                    ground_truth_saliency_map = ground_truth_saliency_map.to(device)
                    ground_truth_fixation_map = ground_truth_fixation_map.to(device)
                    value = self._calculate_torch(
                        predicted_saliency_map,
                        ground_truth_saliency_map,
                        ground_truth_fixation_map,
                    )
                    self.accumulated_values.append(value.item())
            case BackendEnum.JAX:
                for (
                    predicted_saliency_map,
                    ground_truth_saliency_map,
                    ground_truth_fixation_map,
                ) in zip(
                    predicted_saliency_maps,
                    ground_truth_saliency_maps,
                    ground_truth_fixation_maps,
                ):
                    value = self._calculate_jax(
                        predicted_saliency_map,
                        ground_truth_saliency_map,
                        ground_truth_fixation_map,
                    )
                    self.accumulated_values.append(float(value))

    def finalize(self) -> dict[str, Any]:
        """Calcula la media de los valores acumulados."""
        if len(self.accumulated_values) > 0:
            return {self.name: float(np.mean(self.accumulated_values))}
        else:
            return {self.name: float("nan")}

    def reset(self) -> None:
        """Reinicia los valores acumulados."""
        self.accumulated_values = []

    def _calculate_torch(
        self,
        pred_map: torch.Tensor,
        ground_truth_saliency_map: torch.Tensor,
        ground_truth_fixation_map: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _calculate_jax(
        self,
        pred_map: jnp.ndarray,
        ground_truth_saliency_map: jnp.ndarray,
        ground_truth_fixation_map: jnp.ndarray,
    ) -> jnp.ndarray:
        raise NotImplementedError


class BaseMetricCalculator(Protocol):
    name: str
    _metrics: list[BaseMetric]
    _backend: BackendEnum
    _return_saliency: bool = False
    _return_features: bool = False

    def get_dataset_loader(
        self, transform: Optional[Callable[[Any], Any]] = None
    ) -> BaseDatasetLoader:
        """Retorna un dataset loader para el dataset interno."""

    def process_batch(self, batch: Any, model: BaseModelAdapter) -> dict[str, Any]:
        """
        Procesa un batch y retorna los resultados del modelo.
        Este método es abstracto y debe ser implementado por cada calculator específico.

        Args:
            batch: Batch de datos del dataset loader
            model: Modelo para hacer forward

        Returns:
            dict[str, Any]: Resultados del modelo customizados según el tipo de métrica.
                           La estructura del dict varía según el calculator:
                           - Saliency: {"features": [...], "saliency": [...]}
                           - TID: {"features_ref": [...], "features_dist": [...]}
                           - Levels: {"features_img1": [...], "features_img2": [...], "features_img3": [...]}
                           - Nights: {"features_ref": [...], "features_left": [...], "features_right": [...]}
        """
        raise NotImplementedError

    def run(self, model: BaseModelAdapter) -> dict[str, Any]:
        """Ejecuta evaluación completa sobre el dataset interno y retorna un diccionario con los resultados de cada métrica."""
        assert model.backend == self._backend, (
            f"Model backend {model.backend} must match calculator backend {self._backend}"
        )
        assert len(self._metrics) > 0, "At least one metric is required"
        assert all(metric._backend == self._backend for metric in self._metrics), (
            f"All metrics must have backend {self._backend}"
        )

        dataset_loader = self.get_dataset_loader(transform=model.transform)

        # Reset metrics before starting
        for metric in self._metrics:
            metric.reset()

        # Process all batches - metrics accumulate state internally
        for batch in dataset_loader.get_iterator():
            # Allow customization of batch processing
            batch_results = self.process_batch(batch, model)

            for metric in self._metrics:
                metric.calculate(batch, batch_results)

        # Finalize all metrics and collect results
        final_results = {}
        for metric in self._metrics:
            metric_result = metric.finalize()
            final_results.update(metric_result)

        return final_results
