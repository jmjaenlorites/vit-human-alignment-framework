"""Base classes para métricas de visturing."""

import json
from typing import Any, Callable, Literal, Optional

import numpy as np

from ..base import BaseMetric, BaseMetricCalculator
from ...dataset_loaders.base import BaseDatasetLoader
from ...models.base import BaseModelAdapter, ForwardOutputs
from ...utils.common_enums import BackendEnum
from ...utils.common_types import ArrayLike


class BaseVisTuringMetric(BaseMetric):
    """
    Clase base para todas las métricas de visturing.

    Las métricas de visturing evalúan propiedades psicofísicas comparando
    diferencias de features del modelo con respuestas humanas (ground truth).
    """

    type: Literal["visturing"] = "visturing"

    def __init__(self, backend: BackendEnum):
        super().__init__(backend)
        # Almacenar diferencias acumuladas por capa
        self.accumulated_diffs_per_layer: list[list[float]] = []
        # Ground truth y metadata se cargan en el método de cada métrica
        self.ground_truth_data: dict[str, Any] = {}

    def calculate(self, batch: Any, batch_results: ForwardOutputs) -> None:
        """
        Acumula diferencias de features entre test y referencia.

        Args:
            batch: Tupla (test_img, ref_img) del dataset
            batch_results: Dict con {"features_test": [...], "features_ref": [...]}
                          donde cada lista contiene features por capa
        """
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        num_layers = len(features_test)

        # Inicializar listas por capa si es la primera vez
        if len(self.accumulated_diffs_per_layer) == 0:
            self.accumulated_diffs_per_layer = [[] for _ in range(num_layers)]

        # Calcular diferencias por capa
        for layer_idx in range(num_layers):
            feat_test = features_test[layer_idx]  # [batch, dim]
            feat_ref = features_ref[layer_idx]  # [batch, dim] o [1, dim]

            # Calcular diferencia según backend
            if self._backend == BackendEnum.TORCH:
                diffs = self._calculate_diffs_torch(feat_test, feat_ref)
            elif self._backend == BackendEnum.JAX:
                diffs = self._calculate_diffs_jax(feat_test, feat_ref)
            else:
                raise NotImplementedError(f"Backend {self._backend} not supported")

            # Acumular diferencias
            self.accumulated_diffs_per_layer[layer_idx].extend(diffs.tolist())

    def _calculate_diffs_torch(
        self, features_test: ArrayLike, features_ref: ArrayLike
    ) -> ArrayLike:
        """
        Calcula diferencias entre features usando PyTorch.

        Args:
            features_test: Features de test [batch, dim]
            features_ref: Features de referencia [batch, dim] o [1, dim]

        Returns:
            Array de diferencias [batch]
        """
        import torch

        # Aplanar features
        feat_test = features_test.flatten(1)  # [batch, dim]
        feat_ref = features_ref.flatten(1)  # [batch, dim] o [1, dim]

        # Distancia euclidiana
        diff = torch.sqrt(((feat_test - feat_ref) ** 2).sum(dim=-1))

        return diff.cpu().numpy()

    def _calculate_diffs_jax(
        self, features_test: ArrayLike, features_ref: ArrayLike
    ) -> ArrayLike:
        """
        Calcula diferencias entre features usando JAX.

        Args:
            features_test: Features de test [batch, dim]
            features_ref: Features de referencia [batch, dim] o [1, dim]

        Returns:
            Array de diferencias [batch]
        """
        import jax.numpy as jnp

        # Aplanar features
        feat_test = features_test.reshape(features_test.shape[0], -1)
        feat_ref = features_ref.reshape(features_ref.shape[0], -1)

        # Distancia euclidiana
        diff = jnp.sqrt(((feat_test - feat_ref) ** 2).sum(axis=-1))

        return np.array(diff)

    def finalize(self) -> dict[str, Any]:
        """
        Calcula correlaciones entre diferencias y ground truth.

        Returns:
            Dict con correlaciones por capa serializado como JSON
        """
        correlations_per_layer = []

        for layer_idx, diffs in enumerate(self.accumulated_diffs_per_layer):
            if len(diffs) > 0:
                # Calcular correlación con ground truth para esta capa
                corr = self._compute_correlation(np.array(diffs))
                correlations_per_layer.append(float(corr))
            else:
                correlations_per_layer.append(float("nan"))

        # Serializar como JSON string
        return {self.name: json.dumps(correlations_per_layer)}

    def _compute_correlation(self, diffs: np.ndarray) -> float:
        """
        Calcula correlación entre diferencias y ground truth.

        Este método debe ser implementado por cada métrica específica.

        Args:
            diffs: Array de diferencias acumuladas

        Returns:
            Valor de correlación (float)
        """
        raise NotImplementedError(
            "Subclasses must implement _compute_correlation method"
        )

    def reset(self) -> None:
        """Reinicia el estado de la métrica."""
        self.accumulated_diffs_per_layer = []
        self.ground_truth_data = {}
        if hasattr(self, "_diffs_per_layer"):
            self._diffs_per_layer = []


class VisTuringCalculator(BaseMetricCalculator):
    """
    Calculator para métricas de visturing.

    Procesa pares de imágenes (test, ref) y calcula diferencias de features.

    Note: When used with CSV-driven runner, uses default configurations for each property.
    Defaults run full sweeps (channels/frequencies/masks) where applicable to match
    visturing's reference behavior. Levels has a similar limitation (uses default split
    "between_class"). Future migration to JSON-based configuration will allow specifying
    experiment-specific settings per metric.
    """

    def __init__(
        self,
        backend: BackendEnum,
        metrics: list[BaseVisTuringMetric],
        dataset_loader: BaseDatasetLoader,
        name: str = "VisTuringCalculator",
    ):
        self.name = name
        self._metrics = metrics
        self._backend = backend
        self._return_saliency = False
        self._return_features = True
        self._dataset_loader = dataset_loader

    def get_dataset_loader(
        self, transform: Optional[Callable[[Any], Any]] = None
    ) -> BaseDatasetLoader:
        """
        Retorna el dataset loader configurado.

        Args:
            transform: Transformación del modelo (se aplica en el dataset loader)

        Returns:
            Dataset loader para el experimento
        """
        # El transform se pasa al dataset loader en la inicialización
        # Si necesitamos actualizarlo, lo hacemos aquí
        if hasattr(self._dataset_loader, "transform"):
            self._dataset_loader.transform = transform
        return self._dataset_loader

    def process_batch(self, batch: Any, model: BaseModelAdapter) -> dict[str, Any]:
        """
        Procesa un batch de pares (test, ref) y extrae features.

        Args:
            batch: Tupla (test_images, ref_images) del dataset
            model: Modelo para hacer forward

        Returns:
            Dict con {"features_test": [...], "features_ref": [...]}
        """
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Batch must contain at least (test_images, ref_images)")
        test_images, ref_images = batch[:2]

        # Forward para imágenes de test
        batch_results_test = model.forward(
            test_images,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )

        # Forward para imágenes de referencia
        batch_results_ref = model.forward(
            ref_images,
            return_features=self._return_features,
            return_saliency=self._return_saliency,
        )

        # Combinar resultados
        batch_results = {
            "features_test": batch_results_test["features"],
            "features_ref": batch_results_ref["features"],
        }

        return batch_results


def create_default_visturing_calculator(
    backend: BackendEnum,
    metrics: list[BaseVisTuringMetric],
    data_path: Optional[str] = None,
    gt_path: Optional[str] = None,
) -> VisTuringCalculator:
    """
    Crea un calculator con configuración por defecto para métricas de visturing.

    Este calculator se usa cuando las métricas se ejecutan desde CSV.
    Usa configuración por defecto para cada propiedad:
    - Prop1: Spectral sensitivity (default configuration)
    - Prop2: Weber law, achromatic channel
    - Prop3_4: CSF, achromatic channel
    - Etc.

    Note: Similar to LevelsMetricsCalculator which uses default split "between_class".
    Future JSON-based configuration will allow customizing these parameters.

    Args:
        backend: Backend (torch o jax)
        metrics: Lista de métricas a ejecutar
        data_path: Ruta a los datos (default: "./data/visturing")
        gt_path: Ruta al ground truth (default: "./data/visturing")

    Returns:
        VisTuringCalculator configurado con defaults
    """
    from ..visturing.prop1 import Prop1TorchDatasetLoader

    # Use Prop1 as default (most basic property)
    # In future JSON config, users can specify which property to evaluate
    dataset_loader = Prop1TorchDatasetLoader(
        batch_size=32,
        shuffle=False,
        num_workers=2,
        data_path=data_path or "./data/visturing",
    )

    # Load lambdas for metrics that need it
    try:
        _, _, lambdas = dataset_loader.load_data()
        for metric in metrics:
            if hasattr(metric, "lambdas") and metric.lambdas is None:
                metric.lambdas = lambdas
    except Exception:
        # Data will be loaded when calculator runs
        pass

    calculator = VisTuringCalculator(
        backend=backend,
        metrics=metrics,
        dataset_loader=dataset_loader,
        name="VisTuringCalculator_Default",
    )

    return calculator
