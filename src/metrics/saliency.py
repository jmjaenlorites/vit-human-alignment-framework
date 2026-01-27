from typing import Any, Callable, Optional

import torch

from ..dataset_loaders.saliency import (
    BaseSaliencyDatasetLoader,
    SaliencyMIT1003TorchDatasetLoader,
)
from ..utils.common_enums import BackendEnum
from .base import BaseMetricCalculator, BaseSaliencyMetric


class AUC_Judd(BaseSaliencyMetric):
    name: str = "saliency_auc_judd"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)
        self._EPS: float = 1e-8

    def _calculate_torch(
        self,
        pred_map: torch.Tensor,
        ground_truth_saliency_map: torch.Tensor,
        ground_truth_fixation_map: torch.Tensor,
    ) -> torch.Tensor:
        scores = pred_map.flatten().to(torch.float)
        fixations = (ground_truth_fixation_map.flatten() > 0).to(
            device=pred_map.device,
            dtype=torch.float,
        )
        num_fixations = fixations.sum()
        num_non_fix = fixations.numel() - num_fixations
        if num_fixations <= 0 or num_non_fix <= 0:
            return torch.tensor(float("nan"), device=pred_map.device, dtype=torch.float)

        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        sorted_fix = fixations[sorted_indices]
        tp = torch.cumsum(sorted_fix, dim=0)
        fp = torch.cumsum(1.0 - sorted_fix, dim=0)
        tp = tp / (num_fixations + self._EPS)
        fp = fp / (num_non_fix + self._EPS)

        zeros = torch.zeros(1, device=pred_map.device, dtype=torch.float)
        ones = torch.ones(1, device=pred_map.device, dtype=torch.float)
        tp = torch.cat([zeros, tp, ones])
        fp = torch.cat([zeros, fp, ones])
        return torch.trapz(tp, fp)


class PearsonCorrelationCoefficient(BaseSaliencyMetric):
    name: str = "saliency_pearson_correlation_coefficient"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)
        self._EPS: float = 1e-8

    def _calculate_torch(
        self,
        pred_map: torch.Tensor,
        ground_truth_saliency_map: torch.Tensor,
        ground_truth_fixation_map: torch.Tensor,
    ) -> torch.Tensor:
        # Squeeze any extra dimensions to ensure 2D maps
        if pred_map.dim() > 2:
            pred_map = pred_map.squeeze()
        if ground_truth_saliency_map.dim() > 2:
            ground_truth_saliency_map = ground_truth_saliency_map.squeeze()

        x = pred_map.flatten().to(torch.float)
        y = ground_truth_saliency_map.flatten().to(
            device=pred_map.device,
            dtype=torch.float,
        )

        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = torch.sqrt((x_centered**2).sum()) * torch.sqrt((y_centered**2).sum())

        if denom <= self._EPS:
            return torch.tensor(float("nan"), device=pred_map.device, dtype=torch.float)
        result = (x_centered * y_centered).sum() / denom
        return result


class SaliencyMetricsCalculator(BaseMetricCalculator):
    def __init__(
        self, backend: BackendEnum, metrics: Optional[list[BaseSaliencyMetric]] = None
    ):
        super().__init__("SaliencyMetricsCalculator")
        self._metrics: list[BaseSaliencyMetric] = metrics or [
            AUC_Judd(backend),
            PearsonCorrelationCoefficient(backend),
        ]
        self._backend = backend
        self._return_saliency = True

    def get_dataset_loader(
        self, transform: Optional[Callable[[Any], Any]] = None
    ) -> BaseSaliencyDatasetLoader:
        """Carga dataset y metadatos necesarios (responsabilidad local)."""
        match self._backend:
            case BackendEnum.TORCH:
                return SaliencyMIT1003TorchDatasetLoader(
                    batch_size=32, shuffle=False, num_workers=2, transform=transform
                )
            case BackendEnum.JAX:
                raise ValueError(f"Backend {self._backend} not supported")
            case _:
                raise ValueError(f"Backend {self._backend} not supported")

    def process_batch(self, batch: Any, model: Any) -> dict[str, Any]:
        """
        Para saliency, hacemos forward solo del stimulus (batch[0]).
        El batch completo (stimulus, saliency_gt, fixation_gt) se usa luego en calculate().
        """
        stimulus = batch[0]
        batch_results = model.forward(
            stimulus,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )
        return batch_results
