"""Métrica para Propiedad 1: Spectral Sensitivities."""

import json
from typing import Any, Optional

import numpy as np
from scipy.stats import pearsonr

from .base import BaseVisTuringMetric, VisTuringCalculator
from .distance_functions import pearson_correlation_jax
from .ground_truth import load_ground_truth_file
from ...dataset_loaders.visturing.prop1 import Prop1TorchDatasetLoader
from ...utils.common_enums import BackendEnum


class SpectralSensitivityPearson(BaseVisTuringMetric):
    """
    Calcula correlación de Pearson entre diferencias del modelo y
    sensibilidad espectral acromática.

    Esta métrica evalúa si el modelo replica la sensibilidad espectral
    del sistema visual humano a diferentes longitudes de onda.

    Note: When instantiated from CSV (without parameters), uses default configuration.
    Future JSON-based configuration will allow specifying custom gt_path and other parameters.
    """

    name: str = "visturing_spectral_sensitivity"

    def __init__(
        self,
        backend: BackendEnum,
        gt_path: Optional[str] = None,
        lambdas: Optional[np.ndarray] = None,
    ):
        super().__init__(backend)
        self.gt_path = gt_path or "./data/visturing"
        self.lambdas = lambdas  # Se establecerá desde el dataset loader
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            # Ground truth will be downloaded when calculator runs
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de sensibilidades espectrales."""
        data = load_ground_truth_file(self.gt_path, "spectral_sensitivities.mat")
        spectral_data = data["spectral_sensitivities"]

        self.ground_truth_data = {
            "x": spectral_data[0],  # Longitudes de onda
            "achromatic": spectral_data[1],  # Sensibilidad acromática
            "red_green": spectral_data[2],  # Sensibilidad rojo-verde
            "yellow_blue": spectral_data[3],  # Sensibilidad amarillo-azul
        }

    def _compute_correlation(self, diffs: np.ndarray) -> float:
        """
        Calcula correlación de Pearson entre diferencias y sensibilidad acromática.

        Args:
            diffs: Array de diferencias del modelo [N]

        Returns:
            Correlación de Pearson
        """
        if self.lambdas is None:
            raise ValueError("Lambdas not set. Must be provided by dataset loader.")

        # Interpolar ground truth a las longitudes de onda experimentales
        x_gt = self.ground_truth_data["x"]
        a_gt = self.ground_truth_data["achromatic"]

        a_interp = np.interp(self.lambdas, x_gt, a_gt)

        # Calcular correlación de Pearson
        corr, p_value = pearsonr(diffs, a_interp)

        return corr

    def finalize(self) -> dict[str, Any]:
        """
        Calcula correlaciones de Pearson por capa.

        Returns:
            Dict con correlaciones por capa y p-values
        """
        correlations_per_layer = []
        p_values_per_layer = []

        for layer_idx, diffs in enumerate(self.accumulated_diffs_per_layer):
            if len(diffs) > 0 and self.lambdas is not None:
                diffs_array = np.array(diffs)

                # Interpolar ground truth
                x_gt = self.ground_truth_data["x"]
                a_gt = self.ground_truth_data["achromatic"]
                a_interp = np.interp(self.lambdas, x_gt, a_gt)

                # Calcular correlación
                if self._backend == BackendEnum.JAX:
                    corr = float(pearson_correlation_jax(diffs_array, a_interp))
                    p_value = float("nan")
                else:
                    corr, p_value = pearsonr(diffs_array, a_interp)

                correlations_per_layer.append(float(corr))
                p_values_per_layer.append(float(p_value))
            else:
                correlations_per_layer.append(float("nan"))
                p_values_per_layer.append(float("nan"))

        # Serializar como JSON
        return {
            self.name: json.dumps(correlations_per_layer),
            f"{self.name}_pvalues": json.dumps(p_values_per_layer),
        }


def create_prop1_calculator(
    backend: BackendEnum,
    data_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    batch_size: int = 32,
) -> VisTuringCalculator:
    """
    Factory para crear un calculator de Prop1.

    Args:
        backend: Backend (torch o jax)
        data_path: Ruta a los datos
        gt_path: Ruta al ground truth
        batch_size: Tamaño de batch

    Returns:
        VisTuringCalculator configurado para Prop1
    """
    # Crear dataset loader
    dataset_loader = Prop1TorchDatasetLoader(
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        data_path=data_path,
    )

    # Cargar datos para obtener lambdas
    _, _, lambdas = dataset_loader.load_data()

    # Crear métrica
    metric = SpectralSensitivityPearson(
        backend=backend, gt_path=gt_path, lambdas=lambdas
    )

    # Crear calculator
    calculator = VisTuringCalculator(
        backend=backend,
        metrics=[metric],
        dataset_loader=dataset_loader,
        name="Prop1Calculator",
    )

    return calculator
