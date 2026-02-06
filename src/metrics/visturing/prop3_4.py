"""Métrica para Propiedad 3-4: Contrast Sensitivity Function (CSF)."""

import json
from typing import Any, Optional

import numpy as np
from scipy.stats import pearsonr

from .base import BaseVisTuringMetric, VisTuringCalculator
from .distance_functions import (
    calculate_correlations_with_ground_truth,
    prepare_data,
)
from .ground_truth import load_ground_truth_file
from ...dataset_loaders.visturing.prop3_4 import Prop3_4TorchDatasetLoader
from ...utils.common_enums import BackendEnum


class CSFPearson(BaseVisTuringMetric):
    """
    Calcula correlación de Pearson para CSF.

    Evalúa la sensibilidad al contraste en función de la frecuencia espacial.

    Note: When instantiated from CSV, uses default channel='achrom'.
    Future JSON configuration will allow specifying channel (achrom/rg/yb).
    """

    name: str = "visturing_csf_pearson"

    def __init__(
        self,
        backend: BackendEnum,
        channel: str = "all",  # 'achrom', 'rg', 'yb', 'all'
        gt_path: Optional[str] = None,
        freqs: Optional[np.ndarray] = None,
    ):
        super().__init__(backend)
        self.channel = channel
        self.gt_path = gt_path or "./data/visturing"
        self.freqs = freqs
        self.channels = ["achrom", "rg", "yb"]
        self._diffs_per_layer: list[dict[str, list[list[float]]]] = []
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de CSF."""
        channel_map = {
            "achrom": ("responses_CSF_achrom.mat", "CSF_achrom"),
            "rg": ("responses_CSF_RG.mat", "CSF_RG"),
            "yb": ("responses_CSF_YB.mat", "CSF_YB"),
        }

        self.ground_truth_data = {}
        for ch, (filename, varname) in channel_map.items():
            data = load_ground_truth_file(self.gt_path, filename)
            csf_data = data[varname]
            self.ground_truth_data[ch] = {
                "x": csf_data[0],
                "y": csf_data[1],
            }

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 4:
            raise ValueError("Prop3_4 requires batch metadata (channel, freq_idx)")
        channels = batch[2]
        freq_idxs = batch[3]

        if hasattr(channels, "tolist"):
            channels = channels.tolist()
        if hasattr(freq_idxs, "tolist"):
            freq_idxs = freq_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            for _ in range(num_layers):
                layer_map: dict[str, list[list[float]]] = {}
                num_freqs = len(self.freqs) if self.freqs is not None else 0
                for ch in self.channels:
                    layer_map[ch] = [[] for _ in range(num_freqs)]
                self._diffs_per_layer.append(layer_map)

        for layer_idx in range(num_layers):
            feat_test = features_test[layer_idx]
            feat_ref = features_ref[layer_idx]

            if self._backend == BackendEnum.TORCH:
                diffs = self._calculate_diffs_torch(feat_test, feat_ref)
            elif self._backend == BackendEnum.JAX:
                diffs = self._calculate_diffs_jax(feat_test, feat_ref)
            else:
                raise NotImplementedError(f"Backend {self._backend} not supported")

            for idx, (ch, freq_idx) in enumerate(zip(channels, freq_idxs)):
                self._diffs_per_layer[layer_idx][ch][int(freq_idx)].append(
                    float(diffs[idx])
                )

    def finalize(self) -> dict[str, Any]:
        """
        Calcula correlaciones de Pearson por capa.

        Returns:
            Dict con correlaciones por capa
        """
        correlations_per_layer = []

        for layer_idx, layer_diffs in enumerate(self._diffs_per_layer):
            if not layer_diffs or self.freqs is None:
                correlations_per_layer.append(float("nan"))
                continue

            bs = []
            ds = []
            for ch in self.channels:
                diffs_lists = layer_diffs[ch]
                diffs_per_freq = np.array([np.mean(v) for v in diffs_lists])
                gt = self.ground_truth_data[ch]
                _, b, _, d = prepare_data(self.freqs, diffs_per_freq, gt["x"], gt["y"])
                bs.append(b)
                ds.append(d)

            b = np.array(bs)
            d = np.array(ds)
            corr = float(pearsonr(b.ravel(), d.ravel())[0])
            correlations_per_layer.append(corr)

        return {self.name: json.dumps(correlations_per_layer)}


class CSFKendall(BaseVisTuringMetric):
    """
    Calcula correlación de Kendall para CSF.

    Evalúa el ordenamiento de sensibilidades.

    Note: When instantiated from CSV, uses default channel='achrom'.
    """

    name: str = "visturing_csf_kendall"

    def __init__(
        self,
        backend: BackendEnum,
        channel: str = "all",
        gt_path: Optional[str] = None,
        freqs: Optional[np.ndarray] = None,
    ):
        super().__init__(backend)
        self.channel = channel
        self.gt_path = gt_path or "./data/visturing"
        self.freqs = freqs
        self.channels = ["achrom", "rg", "yb"]
        self._diffs_per_layer: list[dict[str, list[list[float]]]] = []
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de CSF."""
        channel_map = {
            "achrom": ("responses_CSF_achrom.mat", "CSF_achrom"),
            "rg": ("responses_CSF_RG.mat", "CSF_RG"),
            "yb": ("responses_CSF_YB.mat", "CSF_YB"),
        }

        self.ground_truth_data = {}
        for ch, (filename, varname) in channel_map.items():
            data = load_ground_truth_file(self.gt_path, filename)
            csf_data = data[varname]
            self.ground_truth_data[ch] = {
                "x": csf_data[0],
                "y": csf_data[1],
            }

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 4:
            raise ValueError("Prop3_4 requires batch metadata (channel, freq_idx)")
        channels = batch[2]
        freq_idxs = batch[3]

        if hasattr(channels, "tolist"):
            channels = channels.tolist()
        if hasattr(freq_idxs, "tolist"):
            freq_idxs = freq_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            for _ in range(num_layers):
                layer_map: dict[str, list[list[float]]] = {}
                num_freqs = len(self.freqs) if self.freqs is not None else 0
                for ch in self.channels:
                    layer_map[ch] = [[] for _ in range(num_freqs)]
                self._diffs_per_layer.append(layer_map)

        for layer_idx in range(num_layers):
            feat_test = features_test[layer_idx]
            feat_ref = features_ref[layer_idx]

            if self._backend == BackendEnum.TORCH:
                diffs = self._calculate_diffs_torch(feat_test, feat_ref)
            elif self._backend == BackendEnum.JAX:
                diffs = self._calculate_diffs_jax(feat_test, feat_ref)
            else:
                raise NotImplementedError(f"Backend {self._backend} not supported")

            for idx, (ch, freq_idx) in enumerate(zip(channels, freq_idxs)):
                self._diffs_per_layer[layer_idx][ch][int(freq_idx)].append(
                    float(diffs[idx])
                )

    def finalize(self) -> dict[str, Any]:
        """
        Calcula correlaciones de Kendall por capa.

        Returns:
            Dict con correlaciones por capa
        """
        results_per_layer = []

        for layer_idx, layer_diffs in enumerate(self._diffs_per_layer):
            if not layer_diffs or self.freqs is None:
                results_per_layer.append({})
                continue

            bs = []
            ds = []
            for ch in self.channels:
                diffs_lists = layer_diffs[ch]
                diffs_per_freq = np.array([np.mean(v) for v in diffs_lists])
                gt = self.ground_truth_data[ch]
                _, b, _, d = prepare_data(self.freqs, diffs_per_freq, gt["x"], gt["y"])
                bs.append(b)
                ds.append(d)

            b = np.array(bs)
            d = np.array(ds)
            correlations = calculate_correlations_with_ground_truth(b, d)
            results_per_layer.append(
                {
                    "kendall": float(correlations["kendall"]),
                    "spearman": float(correlations["spearman"]),
                    "pearson": float(correlations["pearson"]),
                }
            )

        return {self.name: json.dumps(results_per_layer)}


def create_prop3_4_calculator(
    backend: BackendEnum,
    channel: str = "all",
    data_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    batch_size: int = 32,
    include_kendall: bool = True,
) -> VisTuringCalculator:
    """
    Factory para crear un calculator de Prop3_4.

    Args:
        backend: Backend (torch o jax)
        channel: Canal a evaluar ('achrom', 'rg', 'yb')
        data_path: Ruta a los datos
        gt_path: Ruta al ground truth
        batch_size: Tamaño de batch
        include_kendall: Si incluir métrica de Kendall

    Returns:
        VisTuringCalculator configurado para Prop3_4
    """
    # Crear dataset loader
    dataset_loader = Prop3_4TorchDatasetLoader(
        channel=channel,
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
        CSFPearson(backend=backend, channel=channel, gt_path=gt_path, freqs=freqs)
    )
    if include_kendall:
        metrics.append(
            CSFKendall(backend=backend, channel=channel, gt_path=gt_path, freqs=freqs)
        )

    # Crear calculator
    calculator = VisTuringCalculator(
        backend=backend,
        metrics=metrics,
        dataset_loader=dataset_loader,
        name=f"Prop3_4Calculator_{channel}",
    )

    return calculator
