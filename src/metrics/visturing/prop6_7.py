"""Métrica para Propiedad 6-7: Contrast curves without mask."""

import json
from typing import Any, Optional

import numpy as np
from scipy.stats import pearsonr

from .base import BaseVisTuringMetric, VisTuringCalculator
from .distance_functions import calculate_correlations_with_ground_truth, prepare_data
from .ground_truth import load_ground_truth_file
from ...dataset_loaders.visturing.prop6_7 import Prop6_7TorchDatasetLoader
from ...utils.common_enums import BackendEnum


class ContrastCurvesPearson(BaseVisTuringMetric):
    """
    Calcula correlación de Pearson para curvas de contraste sin máscara.

    Evalúa visibilidad en función del contraste para diferentes frecuencias.

    Note: When instantiated from CSV, uses default channel='a', freq='1p5'.
    Future JSON configuration will allow specifying channel and frequency.
    """

    name: str = "visturing_contrast_curves_pearson"

    def __init__(
        self,
        backend: BackendEnum,
        channel: str = "all",  # 'a', 'rg', 'yb', 'all'
        freq: str = "all",  # '1p5', '3', '6', '12', '24', 'all'
        gt_path: Optional[str] = None,
        contrasts_map: Optional[dict[str, np.ndarray]] = None,
    ):
        super().__init__(backend)
        self.channel = channel
        self.freq = freq
        self.gt_path = gt_path or "./data/visturing"
        self.contrasts_map = contrasts_map or {}
        self.channels = ["a", "rg", "yb"]
        self.freq_order = ["1p5", "3", "6", "12", "24"]
        self._diffs_per_layer: list[dict[str, dict[str, list[list[float]]]]] = []
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de contrast curves."""
        channel_map = {
            "a": ("responses_no_mask_achrom_1p5_3_6_12_24.mat", "resp_no_mask_achrom"),
            "rg": ("responses_no_mask_RG_1p5_3_6_12_24.mat", "resp_no_mask_RG"),
            "yb": ("responses_no_mask_YB_1p5_3_6_12_24.mat", "resp_no_mask_YB"),
        }

        freq_idx_map = {"1p5": 1, "3": 2, "6": 3, "12": 4, "24": 5}

        self.ground_truth_data = {}
        for ch, (filename, varname) in channel_map.items():
            data = load_ground_truth_file(self.gt_path, filename)
            resp_data = data[varname]
            y_map = {f: resp_data[idx] for f, idx in freq_idx_map.items()}
            self.ground_truth_data[ch] = {"x": resp_data[0], "y": y_map}

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 5:
            raise ValueError(
                "Prop6_7 requires batch metadata (channel, freq, contrast_idx)"
            )
        channels = batch[2]
        freqs = batch[3]
        contrast_idxs = batch[4]

        if hasattr(channels, "tolist"):
            channels = channels.tolist()
        if hasattr(freqs, "tolist"):
            freqs = freqs.tolist()
        if hasattr(contrast_idxs, "tolist"):
            contrast_idxs = contrast_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            for _ in range(num_layers):
                layer_map: dict[str, dict[str, list[list[float]]]] = {}
                for ch in self.channels:
                    contrasts = self.contrasts_map.get(ch, np.array([]))
                    layer_map[ch] = {}
                    for f in self.freq_order:
                        layer_map[ch][f] = [[] for _ in range(len(contrasts))]
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

            for idx, (ch, f, contrast_idx) in enumerate(
                zip(channels, freqs, contrast_idxs)
            ):
                self._diffs_per_layer[layer_idx][ch][f][int(contrast_idx)].append(
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
            if not layer_diffs:
                correlations_per_layer.append(float("nan"))
                continue

            b_all = []
            d_all = []
            for ch in self.channels:
                contrasts = self.contrasts_map.get(ch)
                if contrasts is None:
                    continue
                gt = self.ground_truth_data[ch]
                for f in self.freq_order:
                    diffs_lists = layer_diffs[ch][f]
                    diffs_per_contrast = np.array([np.mean(v) for v in diffs_lists])
                    _, b, _, d = prepare_data(
                        contrasts, diffs_per_contrast, gt["x"], gt["y"][f]
                    )
                    b_all.append(b.ravel())
                    d_all.append(d.ravel())

            if not b_all:
                correlations_per_layer.append(float("nan"))
                continue

            b_cat = np.concatenate(b_all)
            d_cat = np.concatenate(d_all)
            nan_mask = np.isnan(b_cat)
            pearson = float(pearsonr(b_cat[~nan_mask], d_cat[~nan_mask])[0])
            correlations_per_layer.append(pearson)

        return {self.name: json.dumps(correlations_per_layer)}


class ContrastCurvesKendall(BaseVisTuringMetric):
    """
    Calcula correlación de Kendall para curvas de contraste sin máscara.

    Note: When instantiated from CSV, uses default channel='a', freq='1p5'.
    """

    name: str = "visturing_contrast_curves_kendall"

    def __init__(
        self,
        backend: BackendEnum,
        channel: str = "all",
        freq: str = "all",
        gt_path: Optional[str] = None,
        contrasts_map: Optional[dict[str, np.ndarray]] = None,
    ):
        super().__init__(backend)
        self.channel = channel
        self.freq = freq
        self.gt_path = gt_path or "./data/visturing"
        self.contrasts_map = contrasts_map or {}
        self.channels = ["a", "rg", "yb"]
        self.freq_order = ["1p5", "3", "6", "12", "24"]
        self._diffs_per_layer: list[dict[str, dict[str, list[list[float]]]]] = []
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de contrast curves."""
        channel_map = {
            "a": ("responses_no_mask_achrom_1p5_3_6_12_24.mat", "resp_no_mask_achrom"),
            "rg": ("responses_no_mask_RG_1p5_3_6_12_24.mat", "resp_no_mask_RG"),
            "yb": ("responses_no_mask_YB_1p5_3_6_12_24.mat", "resp_no_mask_YB"),
        }

        freq_idx_map = {"1p5": 1, "3": 2, "6": 3, "12": 4, "24": 5}

        self.ground_truth_data = {}
        for ch, (filename, varname) in channel_map.items():
            data = load_ground_truth_file(self.gt_path, filename)
            resp_data = data[varname]
            y_map = {f: resp_data[idx] for f, idx in freq_idx_map.items()}
            self.ground_truth_data[ch] = {"x": resp_data[0], "y": y_map}

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 5:
            raise ValueError(
                "Prop6_7 requires batch metadata (channel, freq, contrast_idx)"
            )
        channels = batch[2]
        freqs = batch[3]
        contrast_idxs = batch[4]

        if hasattr(channels, "tolist"):
            channels = channels.tolist()
        if hasattr(freqs, "tolist"):
            freqs = freqs.tolist()
        if hasattr(contrast_idxs, "tolist"):
            contrast_idxs = contrast_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            for _ in range(num_layers):
                layer_map: dict[str, dict[str, list[list[float]]]] = {}
                for ch in self.channels:
                    contrasts = self.contrasts_map.get(ch, np.array([]))
                    layer_map[ch] = {}
                    for f in self.freq_order:
                        layer_map[ch][f] = [[] for _ in range(len(contrasts))]
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

            for idx, (ch, f, contrast_idx) in enumerate(
                zip(channels, freqs, contrast_idxs)
            ):
                self._diffs_per_layer[layer_idx][ch][f][int(contrast_idx)].append(
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
            if not layer_diffs:
                results_per_layer.append({})
                continue

            order_corr = {}
            for ch in self.channels:
                contrasts = self.contrasts_map.get(ch)
                if contrasts is None:
                    continue
                gt = self.ground_truth_data[ch]
                bs = []
                ds = []
                for f in self.freq_order:
                    diffs_lists = layer_diffs[ch][f]
                    diffs_per_contrast = np.array([np.mean(v) for v in diffs_lists])
                    _, b, _, d = prepare_data(
                        contrasts, diffs_per_contrast, gt["x"], gt["y"][f]
                    )
                    bs.append(b)
                    ds.append(d)
                b_channel = np.array(bs)
                d_channel = np.array(ds)
                order_corr[ch] = calculate_correlations_with_ground_truth(
                    b_channel, d_channel
                )

            results_per_layer.append(order_corr)

        return {self.name: json.dumps(results_per_layer)}


def create_prop6_7_calculator(
    backend: BackendEnum,
    channel: str = "all",  # 'a', 'rg', 'yb', 'all'
    freq: str = "all",  # '1p5', '3', '6', '12', '24', 'all'
    data_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    batch_size: int = 32,
    include_kendall: bool = True,
) -> VisTuringCalculator:
    """
    Factory para crear un calculator de Prop6_7.

    Args:
        backend: Backend (torch o jax)
        channel: Canal a evaluar ('a', 'rg', 'yb')
        freq: Frecuencia ('1p5', '3', '6', '12', '24')
        data_path: Ruta a los datos
        gt_path: Ruta al ground truth
        batch_size: Tamaño de batch
        include_kendall: Si incluir métrica de Kendall

    Returns:
        VisTuringCalculator configurado para Prop6_7
    """
    gabor_key = "all" if channel == "all" or freq == "all" else f"{channel}_{freq}"
    dataset_loader = Prop6_7TorchDatasetLoader(
        gabor_key=gabor_key,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        data_path=data_path,
    )

    # Cargar datos para obtener contrasts
    gabors, bgs, c_a, c_rg, c_yb = dataset_loader.load_data()
    contrasts_map = {"a": c_a, "rg": c_rg, "yb": c_yb}

    # Crear métricas
    metrics: list[BaseVisTuringMetric] = []
    metrics.append(
        ContrastCurvesPearson(
            backend=backend,
            channel=channel,
            freq=freq,
            gt_path=gt_path,
            contrasts_map=contrasts_map,
        )
    )
    if include_kendall:
        metrics.append(
            ContrastCurvesKendall(
                backend=backend,
                channel=channel,
                freq=freq,
                gt_path=gt_path,
                contrasts_map=contrasts_map,
            )
        )

    # Crear calculator
    calculator = VisTuringCalculator(
        backend=backend,
        metrics=metrics,
        dataset_loader=dataset_loader,
        name=f"Prop6_7Calculator_{gabor_key}",
    )

    return calculator
