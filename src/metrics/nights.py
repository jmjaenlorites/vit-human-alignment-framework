import json
from typing import Any, Callable, Literal, Optional

import torch

from ..dataset_loaders.nights import NightsTorchDatasetLoader
from ..models.base import BaseModelAdapter, ForwardOutputs
from ..utils.common_enums import BackendEnum
from .base import BaseMetric, BaseMetricCalculator
from .representation import RepresentationPooling, representation_vector


class BaseNightsMetric(BaseMetric):
    type: Literal["perceptual"] = "perceptual"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)


class PreferenceAccuracy(BaseNightsMetric):
    """
    Calcula la precisión en la tarea de preferencia perceptual.
    Dado una imagen de referencia y dos distorsiones, predice cuál es más similar
    a la referencia y compara con las preferencias humanas.
    """

    name: str = "nights_preference_accuracy"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)
        # Almacenar resultados por capa
        self.correct_per_layer: list[int] = []
        self.correct_values_per_layer: list[list[int]] = []
        self.prediction_values_per_layer: list[list[str]] = []
        self.human_values_per_layer: list[list[str]] = []
        self.total: int = 0
        self.representation_pooling: RepresentationPooling = "flatten"

    def calculate(self, batch: Any, batch_results: ForwardOutputs) -> None:
        """
        Para Nights, evaluamos si el modelo predice correctamente la preferencia.

        batch: (reference, left, right, left_vote, right_vote)
        batch_results: debe contener 'features_ref', 'features_left', 'features_right'
                      cada uno con features por capa
        """
        ref_batch, left_batch, right_batch, left_votes, right_votes = batch

        features_ref = batch_results["features_ref"]
        features_left = batch_results["features_left"]
        features_right = batch_results["features_right"]

        num_layers = len(features_ref)
        batch_size = len(left_votes)

        # Inicializar contadores si es la primera vez
        if len(self.correct_per_layer) == 0:
            self.correct_per_layer = [0] * num_layers
            self.correct_values_per_layer = [[] for _ in range(num_layers)]
            self.prediction_values_per_layer = [[] for _ in range(num_layers)]
            self.human_values_per_layer = [[] for _ in range(num_layers)]

        # Procesar cada elemento del batch
        for i in range(batch_size):
            self.total += 1

            # Obtener votos
            left_vote = (
                left_votes[i].item()
                if isinstance(left_votes[i], torch.Tensor)
                else left_votes[i]
            )
            right_vote = (
                right_votes[i].item()
                if isinstance(right_votes[i], torch.Tensor)
                else right_votes[i]
            )

            # Para cada capa, calcular similitudes y verificar predicción
            for layer_idx in range(num_layers):
                feat_ref = representation_vector(
                    features_ref[layer_idx][i],
                    self.representation_pooling,
                )
                feat_left = representation_vector(
                    features_left[layer_idx][i],
                    self.representation_pooling,
                )
                feat_right = representation_vector(
                    features_right[layer_idx][i],
                    self.representation_pooling,
                )

                # Calcular similitudes coseno entre referencia y cada distorsión
                if self._backend == BackendEnum.TORCH:
                    sim_left = self._cosine_similarity_torch(feat_ref, feat_left).item()
                    sim_right = self._cosine_similarity_torch(
                        feat_ref, feat_right
                    ).item()
                elif self._backend == BackendEnum.JAX:
                    sim_left = float(self._cosine_similarity_jax(feat_ref, feat_left))
                    sim_right = float(self._cosine_similarity_jax(feat_ref, feat_right))
                else:
                    raise NotImplementedError(
                        f"Backend {self._backend} not implemented"
                    )

                # Lógica de decisión (basada en notebook de Pablo):
                # Si sim_left < sim_right (right es más similar), entonces right_vote debe ser 1
                # Si sim_left > sim_right (left es más similar), entonces left_vote debe ser 1

                if sim_left < sim_right:
                    prediction = "right"
                elif sim_left > sim_right:
                    prediction = "left"
                else:
                    prediction = "tie"

                if right_vote == 1:
                    human_choice = "right"
                elif left_vote == 1:
                    human_choice = "left"
                else:
                    human_choice = "tie"

                is_correct = prediction == human_choice

                if is_correct:
                    self.correct_per_layer[layer_idx] += 1
                self.correct_values_per_layer[layer_idx].append(int(is_correct))
                self.prediction_values_per_layer[layer_idx].append(prediction)
                self.human_values_per_layer[layer_idx].append(human_choice)

    def _cosine_similarity_torch(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        """Calcula similitud coseno entre dos tensores."""
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(0), b.unsqueeze(0), dim=1
        )[0]

    def _cosine_similarity_jax(self, a: Any, b: Any) -> Any:
        """Similitud coseno entre dos vectores `jnp.ndarray` 1-D."""
        import jax.numpy as jnp

        denom = (jnp.linalg.norm(a) * jnp.linalg.norm(b)).clip(min=1e-8)
        return jnp.dot(a, b) / denom

    def finalize(self) -> dict[str, Any]:
        """
        Calcula la precisión por capa después de procesar todo el dataset.
        Retorna un dict con el array de precisiones serializado como JSON.
        """
        accuracies = []
        for layer_idx, correct in enumerate(self.correct_per_layer):
            if self.total > 0:
                accuracy = correct / self.total
                accuracies.append(float(accuracy))
            else:
                accuracies.append(float("nan"))

        # Serialize as JSON string
        return {self.name: json.dumps(accuracies)}

    def reset(self) -> None:
        """Reinicia el estado de la métrica."""
        self.correct_per_layer = []
        self.correct_values_per_layer = []
        self.prediction_values_per_layer = []
        self.human_values_per_layer = []
        self.total = 0

    def unit_records(self) -> list[dict[str, Any]]:
        records = []
        for layer_idx, values in enumerate(self.correct_values_per_layer):
            for unit_idx, value in enumerate(values):
                prediction = self.prediction_values_per_layer[layer_idx][unit_idx]
                human_value = self.human_values_per_layer[layer_idx][unit_idx]
                records.append(
                    {
                        "layer_index": layer_idx,
                        "unit_index": unit_idx,
                        "unit_value": value,
                        "human_value": human_value,
                        "prediction": prediction,
                        "is_correct": value,
                    }
                )
        return records


class NightsMetricsCalculator(BaseMetricCalculator):
    """
    Calculator para métricas de Nights.
    Evalúa la precisión en la tarea de preferencia perceptual.
    """

    def __init__(
        self,
        backend: BackendEnum,
        metrics: Optional[list[BaseNightsMetric]] = None,
        dataset_path: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.name = "NightsMetricsCalculator"
        self._metrics: list[BaseNightsMetric] = metrics or [PreferenceAccuracy(backend)]
        self._backend = backend
        self._return_saliency = False
        self._return_features = True
        self._dataset_path = dataset_path
        self._batch_size = batch_size

    def get_dataset_loader(
        self, transform: Optional[Callable[[Any], Any]] = None
    ) -> NightsTorchDatasetLoader:
        """Carga dataset Nights."""
        match self._backend:
            case BackendEnum.TORCH | BackendEnum.JAX:
                return NightsTorchDatasetLoader(
                    batch_size=self._batch_size,
                    shuffle=False,
                    num_workers=0,
                    transform=transform,
                    dataset_path=self._dataset_path,
                )
            case _:
                raise ValueError(f"Backend {self._backend} not supported")

    def process_batch(self, batch: Any, model: BaseModelAdapter) -> dict[str, Any]:
        """
        Para Nights, necesitamos hacer forward en triplets (reference, left, right).
        """
        # batch = (reference, left, right, left_vote, right_vote)
        reference, left, right, left_votes, right_votes = batch

        # Forward para imagen de referencia
        batch_results_ref = model.forward(
            reference,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )

        # Forward para imagen izquierda
        batch_results_left = model.forward(
            left,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )

        # Forward para imagen derecha
        batch_results_right = model.forward(
            right,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )

        # Combinar resultados para las métricas
        batch_results = {
            "features_ref": batch_results_ref["features"],
            "features_left": batch_results_left["features"],
            "features_right": batch_results_right["features"],
        }

        return batch_results
