import json
import os
from typing import Any, Callable, Literal, Optional

import torch

from ..dataset_loaders.levels import LevelsTorchDatasetLoader
from ..models.base import BaseModelAdapter, ForwardOutputs
from ..utils.common_enums import BackendEnum
from .base import BaseMetric, BaseMetricCalculator
from .representation import RepresentationPooling, representation_vector


class BaseLevelsMetric(BaseMetric):
    type: Literal["perceptual"] = "perceptual"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)


class TripletAccuracy(BaseLevelsMetric):
    """
    Calcula la precisión en la tarea de clasificación de triplets.
    Dado un triplet (img1, img2, img3) y una imagen seleccionada,
    predice cuál imagen es la outlier basándose en similitudes de representaciones.
    """

    name: str = "levels_triplet_accuracy"

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
        Para Levels, evaluamos si el modelo predice correctamente el outlier.

        batch: (img1, img2, img3, selected_image_filenames)
        batch_results: debe contener 'features_img1', 'features_img2', 'features_img3'
                      cada uno con features por capa
        """
        img1_batch, img2_batch, img3_batch, selected_batch, *extra = batch
        img1_names = extra[0] if len(extra) > 0 else None
        img2_names = extra[1] if len(extra) > 1 else None
        img3_names = extra[2] if len(extra) > 2 else None

        features_img1 = batch_results["features_img1"]
        features_img2 = batch_results["features_img2"]
        features_img3 = batch_results["features_img3"]

        num_layers = len(features_img1)
        batch_size = len(selected_batch)

        # Inicializar contadores si es la primera vez
        if len(self.correct_per_layer) == 0:
            self.correct_per_layer = [0] * num_layers
            self.correct_values_per_layer = [[] for _ in range(num_layers)]
            self.prediction_values_per_layer = [[] for _ in range(num_layers)]
            self.human_values_per_layer = [[] for _ in range(num_layers)]

        # Procesar cada elemento del batch
        for i in range(batch_size):
            self.total += 1

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

                # Calcular similitudes coseno entre pares
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

                # Obtener el nombre de archivo seleccionado
                selected_filename = selected_batch[i]

                # Lógica de decisión (basada en notebook de Pablo):
                # Si img1-img3 es más similar, entonces img2 es el outlier (debería ser seleccionado)
                # Si img2-img3 es más similar, entonces img1 es el outlier (debería ser seleccionado)
                # Si img1-img2 es más similar, entonces img3 es el outlier (debería ser seleccionado)

                # Necesitamos comparar con los nombres de archivo originales
                # selected_batch contiene nombres de archivo sin path
                img1_name = (
                    img1_names[i]
                    if img1_names is not None
                    else os.path.basename(self.get_image_path(batch, i, 0))
                )
                img2_name = (
                    img2_names[i]
                    if img2_names is not None
                    else os.path.basename(self.get_image_path(batch, i, 1))
                )
                img3_name = (
                    img3_names[i]
                    if img3_names is not None
                    else os.path.basename(self.get_image_path(batch, i, 2))
                )

                if (sim_13 > sim_12) and (sim_13 > sim_23):
                    prediction = "img2"
                elif (sim_23 > sim_12) and (sim_23 > sim_13):
                    prediction = "img1"
                elif (sim_12 > sim_13) and (sim_12 > sim_23):
                    prediction = "img3"
                else:
                    prediction = "tie"

                if selected_filename == img1_name:
                    human_choice = "img1"
                elif selected_filename == img2_name:
                    human_choice = "img2"
                elif selected_filename == img3_name:
                    human_choice = "img3"
                else:
                    human_choice = str(selected_filename)

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

    def get_image_path(self, batch, batch_idx: int, img_idx: int) -> str:
        """Helper para obtener el path de una imagen del batch."""
        # Esta es una función auxiliar que necesita acceso a los paths originales
        # Por ahora retornamos el nombre del selected que ya está disponible
        # TODO: Mejorar esto para tener acceso a los paths originales
        return ""

    def finalize(self) -> dict[str, Any]:
        """
        Calcula la precisión por capa después de procesar todo el dataset.
        Retorna un dict con el array de precisiones serializado como JSON.
        """
        # TODO: Devolver resultados para todos los splits por defecto (p.ej.,
        # serializando un JSON con cada split o usando otra representación).
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


class LevelsMetricsCalculator(BaseMetricCalculator):
    """
    Calculator para métricas de Levels.
    Evalúa la precisión en la tarea de clasificación de triplets.
    """

    def __init__(
        self,
        backend: BackendEnum,
        split: Literal[
            "between_class", "class_border", "within_class"
        ] = "between_class",
        metrics: Optional[list[BaseLevelsMetric]] = None,
        levels_path: Optional[str] = None,
        imagenet_path: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.name = f"LevelsMetricsCalculator_{split}"
        self._metrics: list[BaseLevelsMetric] = metrics or [TripletAccuracy(backend)]
        self._backend = backend
        self._return_saliency = False
        self._return_features = True
        self.split = split
        self._levels_path = levels_path
        self._imagenet_path = imagenet_path
        self._batch_size = batch_size

    def get_dataset_loader(
        self, transform: Optional[Callable[[Any], Any]] = None
    ) -> LevelsTorchDatasetLoader:
        """Carga dataset Levels con el split especificado.

        El loader es Torch para todos los backends; la ingesta no depende del
        backend numérico del modelo.
        """
        num_workers = 0 if self._backend == BackendEnum.JAX else 2
        match self._backend:
            case BackendEnum.TORCH | BackendEnum.JAX:
                return LevelsTorchDatasetLoader(
                    batch_size=self._batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    split=self.split,
                    transform=transform,
                    levels_path=self._levels_path,
                    imagenet_path=self._imagenet_path,
                )
            case _:
                raise ValueError(f"Backend {self._backend} not supported")

    def process_batch(self, batch: Any, model: BaseModelAdapter) -> dict[str, Any]:
        """
        Para Levels, necesitamos hacer forward en triplets de imágenes.
        """
        # batch = (img1, img2, img3, selected_image_filenames, img1_name, img2_name, img3_name)
        img1, img2, img3, selected, *_ = batch

        # Forward para cada imagen del triplet
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

        # Combinar resultados para las métricas
        batch_results = {
            "features_img1": batch_results_img1["features"],
            "features_img2": batch_results_img2["features"],
            "features_img3": batch_results_img3["features"],
        }

        return batch_results
