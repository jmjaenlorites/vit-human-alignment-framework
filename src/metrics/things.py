import json
from typing import Any, Callable, Literal, Optional

import torch

from ..dataset_loaders.things import ThingsTorchDatasetLoader
from ..models.base import BaseModelAdapter, ForwardOutputs
from ..utils.common_enums import BackendEnum
from .base import BaseMetric, BaseMetricCalculator
from .representation import RepresentationPooling, representation_vector


class BaseThingsMetric(BaseMetric):
    type: Literal["perceptual"] = "perceptual"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)


class ThingsTripletAccuracy(BaseThingsMetric):
    """
    Calcula la precisión en la tarea de odd-one-out del dataset THINGS.
    Dado un triplet (img1, img2, img3) con un `choice_path` humano, predice cuál
    imagen es la outlier basándose en similitudes coseno entre representaciones
    por capa y compara con el `choice_path` real.
    """

    name: str = "things_triplet_accuracy"

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
        Para Things, evaluamos si el modelo predice correctamente el outlier
        elegido por humanos.

        batch: (img1, img2, img3, choice_relpaths, img1_relpaths, img2_relpaths, img3_relpaths)
        batch_results: debe contener 'features_img1', 'features_img2', 'features_img3'
                      cada uno con features por capa
        """
        (
            _img1_batch,
            _img2_batch,
            _img3_batch,
            choice_batch,
            img1_relpaths,
            img2_relpaths,
            img3_relpaths,
        ) = batch

        features_img1 = batch_results["features_img1"]
        features_img2 = batch_results["features_img2"]
        features_img3 = batch_results["features_img3"]

        num_layers = len(features_img1)
        batch_size = len(choice_batch)

        # Inicializar contadores si es la primera vez
        if len(self.correct_per_layer) == 0:
            self.correct_per_layer = [0] * num_layers
            self.correct_values_per_layer = [[] for _ in range(num_layers)]
            self.prediction_values_per_layer = [[] for _ in range(num_layers)]
            self.human_values_per_layer = [[] for _ in range(num_layers)]

        # Procesar cada elemento del batch
        for i in range(batch_size):
            self.total += 1

            choice_relpath = choice_batch[i]
            img1_relpath = img1_relpaths[i]
            img2_relpath = img2_relpaths[i]
            img3_relpath = img3_relpaths[i]

            # Para cada capa, calcular similitudes y verificar predicción
            for layer_idx in range(num_layers):
                feat1 = representation_vector(
                    features_img1[layer_idx][i],
                    self.representation_pooling,
                )
                feat2 = representation_vector(
                    features_img2[layer_idx][i],
                    self.representation_pooling,
                )
                feat3 = representation_vector(
                    features_img3[layer_idx][i],
                    self.representation_pooling,
                )

                if self._backend == BackendEnum.TORCH:
                    sim_12 = self._cosine_similarity_torch(feat1, feat2).item()
                    sim_13 = self._cosine_similarity_torch(feat1, feat3).item()
                    sim_23 = self._cosine_similarity_torch(feat2, feat3).item()
                elif self._backend == BackendEnum.JAX:
                    sim_12 = float(self._cosine_similarity_jax(feat1, feat2))
                    sim_13 = float(self._cosine_similarity_jax(feat1, feat3))
                    sim_23 = float(self._cosine_similarity_jax(feat2, feat3))
                else:
                    raise NotImplementedError(
                        f"Backend {self._backend} not implemented"
                    )

                # Lógica de decisión (idéntica a Levels):
                # - sim_13 mayor => img2 es outlier
                # - sim_23 mayor => img1 es outlier
                # - sim_12 mayor => img3 es outlier
                if (sim_13 > sim_12) and (sim_13 > sim_23):
                    prediction = "img2"
                elif (sim_23 > sim_12) and (sim_23 > sim_13):
                    prediction = "img1"
                elif (sim_12 > sim_13) and (sim_12 > sim_23):
                    prediction = "img3"
                else:
                    prediction = "tie"

                if choice_relpath == img1_relpath:
                    human_choice = "img1"
                elif choice_relpath == img2_relpath:
                    human_choice = "img2"
                elif choice_relpath == img3_relpath:
                    human_choice = "img3"
                else:
                    human_choice = str(choice_relpath)

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
        for _layer_idx, correct in enumerate(self.correct_per_layer):
            if self.total > 0:
                accuracy = correct / self.total
                accuracies.append(float(accuracy))
            else:
                accuracies.append(float("nan"))

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


class ThingsMetricsCalculator(BaseMetricCalculator):
    """
    Calculator para métricas de Things.
    Evalúa la precisión en la tarea de odd-one-out (triplet).
    """

    def __init__(
        self,
        backend: BackendEnum,
        metrics: Optional[list[BaseThingsMetric]] = None,
        dataset_path: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.name = "ThingsMetricsCalculator"
        self._metrics: list[BaseThingsMetric] = metrics or [
            ThingsTripletAccuracy(backend)
        ]
        self._backend = backend
        self._return_saliency = False
        self._return_features = True
        self._dataset_path = dataset_path
        self._batch_size = batch_size

    def get_dataset_loader(
        self, transform: Optional[Callable[[Any], Any]] = None
    ) -> ThingsTorchDatasetLoader:
        """Carga dataset Things.

        El loader es Torch para todos los backends; la ingesta no depende del
        backend numérico del modelo.
        """
        num_workers = 0 if self._backend == BackendEnum.JAX else 2
        match self._backend:
            case BackendEnum.TORCH | BackendEnum.JAX:
                return ThingsTorchDatasetLoader(
                    batch_size=self._batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    transform=transform,
                    dataset_path=self._dataset_path,
                )
            case _:
                raise ValueError(f"Backend {self._backend} not supported")

    def process_batch(self, batch: Any, model: BaseModelAdapter) -> dict[str, Any]:
        """
        Para Things, hacemos forward sobre los tres miembros del triplet y
        agregamos features por imagen.
        """
        img1, img2, img3, *_ = batch

        batch_results_img1 = model.forward(
            img1,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )
        batch_results_img2 = model.forward(
            img2,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )
        batch_results_img3 = model.forward(
            img3,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )

        return {
            "features_img1": batch_results_img1["features"],
            "features_img2": batch_results_img2["features"],
            "features_img3": batch_results_img3["features"],
        }
