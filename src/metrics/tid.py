import logging
import torch
import json
from typing import Any, Callable, Optional, Literal
from scipy.stats import spearmanr

from .base import BaseMetric, BaseMetricCalculator
from ..dataset_loaders.tid import TID2013TorchDatasetLoader
from ..models.base import BaseModelAdapter, ForwardOutputs
from ..utils.common_enums import BackendEnum

logger = logging.getLogger(__name__)


class BaseTIDMetric(BaseMetric):
    type: Literal["perceptual"] = "perceptual"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)


class SpearmanCorrelationMOS(BaseTIDMetric):
    """
    Calcula la correlación de Spearman entre las similitudes coseno
    de las representaciones y los MOS scores humanos.
    """

    name: str = "tid_spearman_mos"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)
        # Almacenar similitudes y MOS scores para calcular correlación al final
        self.similarities: list[list[float]] = []
        self.mos_scores: list[float] = []

    def calculate(self, batch: Any, batch_results: ForwardOutputs) -> None:
        """
        Para TID, acumulamos las similitudes y MOS scores.
        La correlación se calcula al final en finalize().

        batch: (reference_images, distorted_images, mos_scores)
        batch_results: debe contener 'features_ref' y 'features_dist' con las representaciones por capa
        """
        reference_images, distorted_images, mos_batch = batch

        # batch_results["features"] debe ser una lista de features por capa
        # Cada elemento es [batch_size, ...] con las features de esa capa
        features_ref = batch_results["features_ref"]
        features_dist = batch_results["features_dist"]

        num_layers = len(features_ref)
        batch_size = len(mos_batch)

        # Inicializar similitudes por capa si es la primera vez
        if len(self.similarities) == 0:
            self.similarities = [[] for _ in range(num_layers)]

        # Calcular similitud coseno entre referencia y distorsión por cada capa
        for layer_idx in range(num_layers):
            layer_features_ref = features_ref[layer_idx]
            layer_features_dist = features_dist[layer_idx]

            for i in range(batch_size):
                feat_ref = layer_features_ref[i].flatten()
                feat_dist = layer_features_dist[i].flatten()

                # Similitud coseno
                if self._backend == BackendEnum.TORCH:
                    similarity = self._cosine_similarity_torch(feat_ref, feat_dist)
                    self.similarities[layer_idx].append(similarity.item())
                else:
                    raise NotImplementedError(
                        f"Backend {self._backend} not implemented"
                    )

        # Guardar MOS scores
        self.mos_scores.extend(
            [mos.item() if isinstance(mos, torch.Tensor) else mos for mos in mos_batch]
        )

    def _cosine_similarity_torch(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Calcula similitud coseno entre dos tensores."""
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0), dim=1
        )[0]

    def finalize(self) -> dict[str, Any]:
        """
        Calcula la correlación de Spearman por capa después de procesar todo el dataset.
        Retorna un dict con el array de correlaciones serializado como JSON.
        """
        correlations = []
        for layer_idx, layer_similarities in enumerate(self.similarities):
            if len(layer_similarities) > 0:
                if len(layer_similarities) != len(self.mos_scores):
                    logger.warning(
                        "TID metric length mismatch: similarities=%s, mos_scores=%s",
                        len(layer_similarities),
                        len(self.mos_scores),
                    )
                min_len = min(len(layer_similarities), len(self.mos_scores))
                if min_len == 0:
                    correlations.append(float("nan"))
                    continue
                corr, _ = spearmanr(
                    layer_similarities[:min_len], self.mos_scores[:min_len]
                )
                correlations.append(float(corr))
            else:
                correlations.append(float("nan"))

        # Serialize as JSON string
        return {self.name: json.dumps(correlations)}

    def reset(self) -> None:
        """Reinicia el estado de la métrica."""
        self.similarities = []
        self.mos_scores = []


class TIDMetricsCalculator(BaseMetricCalculator):
    """
    Calculator para métricas de TID2013.
    Evalúa la correlación entre similitudes de representaciones y MOS scores humanos.
    """

    def __init__(
        self, backend: BackendEnum, metrics: Optional[list[BaseTIDMetric]] = None
    ):
        self.name = "TIDMetricsCalculator"
        self._metrics: list[BaseTIDMetric] = metrics or [
            SpearmanCorrelationMOS(backend)
        ]
        self._backend = backend
        self._return_saliency = False
        self._return_features = True

    def get_dataset_loader(
        self, transform: Optional[Callable[[Any], Any]] = None
    ) -> TID2013TorchDatasetLoader:
        """Carga dataset TID2013."""
        match self._backend:
            case BackendEnum.TORCH:
                return TID2013TorchDatasetLoader(
                    batch_size=32, shuffle=False, num_workers=2, transform=transform
                )
            case BackendEnum.JAX:
                raise ValueError(f"Backend {self._backend} not supported")
            case _:
                raise ValueError(f"Backend {self._backend} not supported")

    def process_batch(self, batch: Any, model: BaseModelAdapter) -> dict[str, Any]:
        """
        Para TID, necesitamos hacer forward en pares de imágenes (reference, distorted).
        """
        # batch = (reference_images, distorted_images, mos_scores)
        reference_images, distorted_images, mos_scores = batch

        # Forward para imágenes de referencia
        batch_results_ref = model.forward(
            reference_images,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )

        # Forward para imágenes distorsionadas
        batch_results_dist = model.forward(
            distorted_images,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )

        # Combinar resultados para las métricas
        batch_results = {
            "features_ref": batch_results_ref["features"],
            "features_dist": batch_results_dist["features"],
        }

        return batch_results
