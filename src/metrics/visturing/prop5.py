"""Métrica para Propiedad 5: Campbell-Blakemore (frequency masking)."""

import json
from typing import Any, Optional

import numpy as np

from .base import BaseVisTuringMetric, VisTuringCalculator
from .distance_functions import (
    calculate_correlations_with_ground_truth,
    calculate_pearson_stack,
    prepare_data,
)
from .ground_truth import load_ground_truth_file
from ...dataset_loaders.visturing.prop5 import Prop5TorchDatasetLoader
from ...utils.common_enums import BackendEnum


class CampbellBlakemorePearson(BaseVisTuringMetric):
    """
    Calcula correlación de Pearson para Campbell-Blakemore.

    Evalúa el enmascaramiento por frecuencia.

    Note: When instantiated from CSV, uses default mask_freq='3'.
    Future JSON configuration will allow specifying mask frequency (3/6/12 cpd).
    """

    name: str = "visturing_campbell_blakemore_pearson"

    def __init__(
        self,
        backend: BackendEnum,
        mask_freq: str = "all",  # '3', '6', '12', 'all'
        gt_path: Optional[str] = None,
        freqs: Optional[np.ndarray] = None,
    ):
        super().__init__(backend)
        self.mask_freq = mask_freq
        self.gt_path = gt_path or "./data/visturing"
        self.freqs = freqs
        self._diffs_per_layer: list[dict[str, list[list[float]]]] = []
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de Campbell-Blakemore."""
        data = load_ground_truth_file(self.gt_path, "Campbell_Blakemore.mat")
        cb_data = data["Campbell_Blakemore"]

        # x, y1 (3cpd), y2 (6cpd), y3 (12cpd)
        self.ground_truth_data = {
            "x": cb_data[0],
            "3": cb_data[1],
            "6": cb_data[2],
            "12": cb_data[3],
        }

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 4:
            raise ValueError("Prop5 requires batch metadata (noise_key, freq_idx)")
        noise_keys = batch[2]
        freq_idxs = batch[3]

        if hasattr(noise_keys, "tolist"):
            noise_keys = noise_keys.tolist()
        if hasattr(freq_idxs, "tolist"):
            freq_idxs = freq_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            num_freqs = len(self.freqs) if self.freqs is not None else 0
            for _ in range(num_layers):
                self._diffs_per_layer.append({})

        for layer_idx in range(num_layers):
            feat_test = features_test[layer_idx]
            feat_ref = features_ref[layer_idx]

            if self._backend == BackendEnum.TORCH:
                diffs = self._calculate_diffs_torch(feat_test, feat_ref)
            elif self._backend == BackendEnum.JAX:
                diffs = self._calculate_diffs_jax(feat_test, feat_ref)
            else:
                raise NotImplementedError(f"Backend {self._backend} not supported")

            for idx, (noise_key, freq_idx) in enumerate(zip(noise_keys, freq_idxs)):
                mask_id = str(noise_key).split("_")[-1]
                if mask_id not in self._diffs_per_layer[layer_idx]:
                    num_freqs = len(self.freqs) if self.freqs is not None else 0
                    self._diffs_per_layer[layer_idx][mask_id] = [
                        [] for _ in range(num_freqs)
                    ]
                self._diffs_per_layer[layer_idx][mask_id][int(freq_idx)].append(
                    float(diffs[idx])
                )

    def finalize(self) -> dict[str, Any]:
        correlations_per_layer = []

        for layer_idx, layer_diffs in enumerate(self._diffs_per_layer):
            if not layer_diffs or self.freqs is None:
                correlations_per_layer.append(float("nan"))
                continue

            diffs = {}
            for mask_id, diffs_lists in layer_diffs.items():
                diffs_per_freq = np.array([np.mean(v) for v in diffs_lists])
                diffs[mask_id] = diffs_per_freq

            if "a" not in diffs:
                correlations_per_layer.append(float("nan"))
                continue

            diffs_a = diffs.pop("a")
            diffs_inv = {k: (diffs_a + 1e-6) / v for k, v in diffs.items()}
            diffs_inv = {k: (v - 1.0) for k, v in diffs_inv.items()}
            diffs_inv = {
                k: np.clip(v, a_min=1e-6, a_max=np.inf) for k, v in diffs_inv.items()
            }
            diffs_inv = {k: v / v.max() for k, v in diffs_inv.items()}

            if not diffs_inv:
                correlations_per_layer.append(float("nan"))
                continue

            gt_x = self.ground_truth_data["x"]
            y1_gt = self.ground_truth_data["3"]
            y2_gt = self.ground_truth_data["6"]
            y3_gt = self.ground_truth_data["12"]

            diffs_stack = np.stack(
                [
                    diffs_inv["3"][1:],
                    diffs_inv["6"][1:],
                    diffs_inv["12"][1:],
                ]
            )
            _, _, _, d1 = prepare_data(self.freqs[1:], diffs_inv["3"][1:], gt_x, y1_gt)
            _, _, _, d2 = prepare_data(self.freqs[1:], diffs_inv["6"][1:], gt_x, y2_gt)
            _, _, _, d3 = prepare_data(self.freqs[1:], diffs_inv["12"][1:], gt_x, y3_gt)
            ds = np.stack([d1, d2, d3])

            pearson_corr = float(calculate_pearson_stack(diffs_stack, ds)[0])
            correlations_per_layer.append(pearson_corr)

        return {self.name: json.dumps(correlations_per_layer)}


class CampbellBlakemoreKendall(BaseVisTuringMetric):
    """
    Calcula correlación de Kendall para Campbell-Blakemore.
    """

    name: str = "visturing_campbell_blakemore_kendall"

    def __init__(
        self,
        backend: BackendEnum,
        mask_freq: str = "all",
        gt_path: Optional[str] = None,
        freqs: Optional[np.ndarray] = None,
    ):
        super().__init__(backend)
        self.mask_freq = mask_freq
        self.gt_path = gt_path or "./data/visturing"
        self.freqs = freqs
        self._diffs_per_layer: list[dict[str, list[list[float]]]] = []
        self._load_ground_truth()

    def _load_ground_truth(self):
        """Carga datos de ground truth de Campbell-Blakemore."""
        data = load_ground_truth_file(self.gt_path, "Campbell_Blakemore.mat")
        cb_data = data["Campbell_Blakemore"]

        self.ground_truth_data = {
            "x": cb_data[0],
            "3": cb_data[1],
            "6": cb_data[2],
            "12": cb_data[3],
        }

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 4:
            raise ValueError("Prop5 requires batch metadata (noise_key, freq_idx)")
        noise_keys = batch[2]
        freq_idxs = batch[3]

        if hasattr(noise_keys, "tolist"):
            noise_keys = noise_keys.tolist()
        if hasattr(freq_idxs, "tolist"):
            freq_idxs = freq_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            for _ in range(num_layers):
                self._diffs_per_layer.append({})

        for layer_idx in range(num_layers):
            feat_test = features_test[layer_idx]
            feat_ref = features_ref[layer_idx]

            if self._backend == BackendEnum.TORCH:
                diffs = self._calculate_diffs_torch(feat_test, feat_ref)
            elif self._backend == BackendEnum.JAX:
                diffs = self._calculate_diffs_jax(feat_test, feat_ref)
            else:
                raise NotImplementedError(f"Backend {self._backend} not supported")

            for idx, (noise_key, freq_idx) in enumerate(zip(noise_keys, freq_idxs)):
                mask_id = str(noise_key).split("_")[-1]
                if mask_id not in self._diffs_per_layer[layer_idx]:
                    num_freqs = len(self.freqs) if self.freqs is not None else 0
                    self._diffs_per_layer[layer_idx][mask_id] = [
                        [] for _ in range(num_freqs)
                    ]
                self._diffs_per_layer[layer_idx][mask_id][int(freq_idx)].append(
                    float(diffs[idx])
                )

    def finalize(self) -> dict[str, Any]:
        results_per_layer = []

        for layer_idx, layer_diffs in enumerate(self._diffs_per_layer):
            if not layer_diffs or self.freqs is None:
                results_per_layer.append({})
                continue

            diffs = {}
            for mask_id, diffs_lists in layer_diffs.items():
                diffs_per_freq = np.array([np.mean(v) for v in diffs_lists])
                diffs[mask_id] = diffs_per_freq

            if "a" not in diffs:
                results_per_layer.append({})
                continue

            diffs_a = diffs.pop("a")
            diffs_inv = {k: (diffs_a + 1e-6) / v for k, v in diffs.items()}
            diffs_inv = {k: (v - 1.0) for k, v in diffs_inv.items()}
            diffs_inv = {
                k: np.clip(v, a_min=1e-6, a_max=np.inf) for k, v in diffs_inv.items()
            }
            diffs_inv = {k: v / v.max() for k, v in diffs_inv.items()}

            if not diffs_inv:
                results_per_layer.append({})
                continue

            gt_x = self.ground_truth_data["x"]
            y1_gt = self.ground_truth_data["3"]
            y2_gt = self.ground_truth_data["6"]
            y3_gt = self.ground_truth_data["12"]

            diffs_stack = np.stack(
                [
                    diffs_inv["3"][1:],
                    diffs_inv["6"][1:],
                    diffs_inv["12"][1:],
                ]
            )
            _, _, _, d1 = prepare_data(self.freqs[1:], diffs_inv["3"][1:], gt_x, y1_gt)
            _, _, _, d2 = prepare_data(self.freqs[1:], diffs_inv["6"][1:], gt_x, y2_gt)
            _, _, _, d3 = prepare_data(self.freqs[1:], diffs_inv["12"][1:], gt_x, y3_gt)
            ds = np.stack([d1, d2, d3])

            correlations = calculate_correlations_with_ground_truth(diffs_stack, ds)
            results_per_layer.append(
                {
                    "kendall": float(correlations["kendall"]),
                    "spearman": float(correlations["spearman"]),
                }
            )

        return {self.name: json.dumps(results_per_layer)}


def create_prop5_calculator(
    backend: BackendEnum,
    mask_freq: str = "all",  # '3', '6', '12', 'all'
    data_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    batch_size: int = 32,
    include_kendall: bool = True,
) -> VisTuringCalculator:
    """
    Factory para crear un calculator de Prop5.

    Args:
        backend: Backend (torch o jax)
        mask_freq: Frecuencia de máscara ('3', '6', '12')
        data_path: Ruta a los datos
        gt_path: Ruta al ground truth
        batch_size: Tamaño de batch
        include_kendall: Si incluir métrica de Kendall

    Returns:
        VisTuringCalculator configurado para Prop5
    """
    # Crear dataset loader
    noise_key = "all" if mask_freq == "all" else mask_freq
    dataset_loader = Prop5TorchDatasetLoader(
        noise_key=noise_key,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        data_path=data_path,
    )

    # Cargar datos para obtener freqs
    noises, background, freqs = dataset_loader.load_data()

    # Crear métricas
    metrics: list[BaseVisTuringMetric] = []
    metrics.append(
        CampbellBlakemorePearson(
            backend=backend, mask_freq=mask_freq, gt_path=gt_path, freqs=freqs
        )
    )
    if include_kendall:
        metrics.append(
            CampbellBlakemoreKendall(
                backend=backend, mask_freq=mask_freq, gt_path=gt_path, freqs=freqs
            )
        )

    # Crear calculator
    calculator = VisTuringCalculator(
        backend=backend,
        metrics=metrics,
        dataset_loader=dataset_loader,
        name=f"Prop5Calculator_Fmask_{mask_freq}",
    )

    return calculator
