"""Métrica para Propiedad 2: Weber Law."""

import json
from typing import Any, Optional

import numpy as np
from scipy.stats import pearsonr

from .base import BaseVisTuringMetric, VisTuringCalculator
from .distance_functions import (
    calculate_spearman,
    calculate_spearman_jax,
    pearson_correlation_jax,
    prepare_data,
)
from .ground_truth import load_ground_truth_file
from ...dataset_loaders.visturing.prop2 import Prop2TorchDatasetLoader
from ...utils.common_enums import BackendEnum


class WeberLawPearson(BaseVisTuringMetric):
    """
    Calcula correlación de Pearson para la ley de Weber.

    Evalúa respuestas acromáticas y cromáticas a cambios de luminancia.

    Note: When instantiated from CSV, uses default channel='achrom'.
    Future JSON configuration will allow specifying channel (achrom/red_green/yellow_blue).
    """

    name: str = "visturing_weber_law_pearson"

    def __init__(
        self,
        backend: BackendEnum,
        channel: str = "all",  # 'achrom', 'red_green', 'yellow_blue', 'all'
        gt_path: Optional[str] = None,
        x_values_map: Optional[dict[str, np.ndarray]] = None,
        levels_map: Optional[dict[str, int]] = None,
    ):
        super().__init__(backend)
        self.channel = channel
        self.gt_path = gt_path or "./data/visturing"
        self.x_values_map = x_values_map or {}
        self.channels = ["achrom", "red_green", "yellow_blue"]
        self.levels_map = levels_map or {}
        self._diffs_per_layer: list[dict[str, list[list[float]]]] = []
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            # Ground truth will be downloaded when calculator runs
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de Weber law."""
        # Weber para acromático
        weber_data = load_ground_truth_file(self.gt_path, "weber.mat")
        weber = weber_data["weber"]

        # Respuestas cromáticas
        resp_rg_data = load_ground_truth_file(self.gt_path, "resp_RG.mat")
        resp_rg = resp_rg_data["resp_RG"]

        resp_yb_data = load_ground_truth_file(self.gt_path, "resp_YB.mat")
        resp_yb = resp_yb_data["resp_YB"]

        self.ground_truth_data = {
            "achrom": {"x": weber[0], "y": weber[1]},
            "red_green": {"x": resp_rg[0], "y": resp_rg[1]},
            "yellow_blue": {"x": resp_yb[0], "y": resp_yb[1]},
        }

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 4:
            raise ValueError("Prop2 requires batch metadata (channel, level_idx)")
        channels = batch[2]
        level_idxs = batch[3]
        sample_idxs = batch[4] if len(batch) > 4 else None
        bg_idxs = batch[5] if len(batch) > 5 else None

        if hasattr(channels, "tolist"):
            channels = channels.tolist()
        if hasattr(level_idxs, "tolist"):
            level_idxs = level_idxs.tolist()
        if sample_idxs is not None and hasattr(sample_idxs, "tolist"):
            sample_idxs = sample_idxs.tolist()
        if bg_idxs is not None and hasattr(bg_idxs, "tolist"):
            bg_idxs = bg_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            for _ in range(num_layers):
                layer_map: dict[str, list[list[float]]] = {}
                for ch in self.channels:
                    num_levels = self.levels_map.get(
                        ch, len(self.x_values_map.get(ch, []))
                    )
                    layer_map[ch] = [[] for _ in range(num_levels)]
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

            for idx, (ch, level_idx) in enumerate(zip(channels, level_idxs)):
                diff_value = float(diffs[idx])
                if (
                    self._backend == BackendEnum.TORCH
                    and ch != "achrom"
                    and sample_idxs is not None
                    and bg_idxs is not None
                ):
                    sample_idx = int(sample_idxs[idx])
                    bg_idx = int(bg_idxs[idx])
                    diff_value *= -1.0 if sample_idx < bg_idx else 1.0
                self._diffs_per_layer[layer_idx][ch][int(level_idx)].append(diff_value)

    def finalize(self) -> dict[str, Any]:
        results_per_layer = []

        for layer_idx, layer_diffs in enumerate(self._diffs_per_layer):
            if not layer_diffs:
                results_per_layer.append(
                    {"pearson_achrom": float("nan"), "pearson_chrom": float("nan")}
                )
                continue

            x_a = self.x_values_map.get("achrom")
            x_rg = self.x_values_map.get("red_green")
            x_yb = self.x_values_map.get("yellow_blue")

            if x_a is None or x_rg is None or x_yb is None:
                results_per_layer.append(
                    {"pearson_achrom": float("nan"), "pearson_chrom": float("nan")}
                )
                continue

            def _stack_channel(values: list[list[float]]) -> np.ndarray:
                lengths = [len(v) for v in values if len(v) > 0]
                if not lengths:
                    return np.array([])
                min_len = min(lengths)
                return np.stack([np.array(v[:min_len]) for v in values])

            diffs_a = _stack_channel(layer_diffs["achrom"])
            diffs_rg = _stack_channel(layer_diffs["red_green"])
            diffs_yb = _stack_channel(layer_diffs["yellow_blue"])

            if diffs_a.size == 0 or diffs_rg.size == 0 or diffs_yb.size == 0:
                results_per_layer.append(
                    {"pearson_achrom": float("nan"), "pearson_chrom": float("nan")}
                )
                continue

            gt_a = self.ground_truth_data["achrom"]
            gt_rg = self.ground_truth_data["red_green"]
            gt_yb = self.ground_truth_data["yellow_blue"]

            _, b_a, _, d_a = prepare_data(x_a, diffs_a, gt_a["x"], gt_a["y"])
            _, b_rg, _, d_rg = prepare_data(x_rg, diffs_rg, gt_rg["x"], gt_rg["y"])
            _, b_yb, _, d_yb = prepare_data(x_yb, diffs_yb, gt_yb["x"], gt_yb["y"])

            achrom_x = np.concatenate([b_a[0].ravel()])
            achrom_y = np.concatenate([d_a.ravel()])
            chrom_x = np.concatenate([b_rg[2].ravel(), b_yb[2].ravel()])
            chrom_y = np.concatenate([d_rg.ravel(), d_yb.ravel()])

            if self._backend == BackendEnum.JAX:
                corr_achrom = float(pearson_correlation_jax(achrom_x, achrom_y))
                corr_chrom = float(pearson_correlation_jax(chrom_x, chrom_y))
            else:
                corr_achrom = float(pearsonr(achrom_x, achrom_y)[0])
                corr_chrom = float(pearsonr(chrom_x, chrom_y)[0])

            results_per_layer.append(
                {
                    "pearson_achrom": corr_achrom,
                    "pearson_chrom": corr_chrom,
                }
            )

        return {self.name: json.dumps(results_per_layer)}


class WeberLawKendall(BaseVisTuringMetric):
    """
    Calcula correlación de Kendall/Spearman para la ley de Weber.

    Evalúa el ordenamiento de respuestas.

    Note: When instantiated from CSV, uses default channel='achrom'.
    Future JSON configuration will allow specifying channel.
    """

    name: str = "visturing_weber_law_kendall"

    def __init__(
        self,
        backend: BackendEnum,
        channel: str = "all",
        gt_path: Optional[str] = None,
        x_values_map: Optional[dict[str, np.ndarray]] = None,
        levels_map: Optional[dict[str, int]] = None,
    ):
        super().__init__(backend)
        self.channel = channel
        self.gt_path = gt_path or "./data/visturing"
        self.x_values_map = x_values_map or {}
        self.channels = ["achrom", "red_green", "yellow_blue"]
        self.levels_map = levels_map or {}
        self._diffs_per_layer: list[dict[str, list[list[float]]]] = []
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de Weber law."""
        # Weber para acromático
        weber_data = load_ground_truth_file(self.gt_path, "weber.mat")
        weber = weber_data["weber"]

        # Respuestas cromáticas
        resp_rg_data = load_ground_truth_file(self.gt_path, "resp_RG.mat")
        resp_rg = resp_rg_data["resp_RG"]

        resp_yb_data = load_ground_truth_file(self.gt_path, "resp_YB.mat")
        resp_yb = resp_yb_data["resp_YB"]

        self.ground_truth_data = {
            "achrom": {"x": weber[0], "y": weber[1]},
            "red_green": {"x": resp_rg[0], "y": resp_rg[1]},
            "yellow_blue": {"x": resp_yb[0], "y": resp_yb[1]},
        }

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 4:
            raise ValueError("Prop2 requires batch metadata (channel, level_idx)")
        channels = batch[2]
        level_idxs = batch[3]
        sample_idxs = batch[4] if len(batch) > 4 else None
        bg_idxs = batch[5] if len(batch) > 5 else None

        if hasattr(channels, "tolist"):
            channels = channels.tolist()
        if hasattr(level_idxs, "tolist"):
            level_idxs = level_idxs.tolist()
        if sample_idxs is not None and hasattr(sample_idxs, "tolist"):
            sample_idxs = sample_idxs.tolist()
        if bg_idxs is not None and hasattr(bg_idxs, "tolist"):
            bg_idxs = bg_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            for _ in range(num_layers):
                layer_map: dict[str, list[list[float]]] = {}
                for ch in self.channels:
                    num_levels = self.levels_map.get(
                        ch, len(self.x_values_map.get(ch, []))
                    )
                    layer_map[ch] = [[] for _ in range(num_levels)]
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

            for idx, (ch, level_idx) in enumerate(zip(channels, level_idxs)):
                diff_value = float(diffs[idx])
                if (
                    self._backend == BackendEnum.TORCH
                    and ch != "achrom"
                    and sample_idxs is not None
                    and bg_idxs is not None
                ):
                    sample_idx = int(sample_idxs[idx])
                    bg_idx = int(bg_idxs[idx])
                    diff_value *= -1.0 if sample_idx < bg_idx else 1.0
                self._diffs_per_layer[layer_idx][ch][int(level_idx)].append(diff_value)

    def finalize(self) -> dict[str, Any]:
        results_per_layer = []

        for layer_idx, layer_diffs in enumerate(self._diffs_per_layer):
            if not layer_diffs:
                results_per_layer.append({})
                continue

            layer_result: dict[str, dict[str, float]] = {}
            for ch in self.channels:
                x_vals = self.x_values_map.get(ch)
                if x_vals is None:
                    continue
                diffs_channel = np.stack([np.array(v) for v in layer_diffs[ch]])
                gt = self.ground_truth_data[ch]
                _, b, _, _ = prepare_data(x_vals, diffs_channel, gt["x"], gt["y"])
                if self._backend == BackendEnum.JAX:
                    correlations = calculate_spearman_jax(
                        b, ideal_ordering=[0, 1, 2, 3, 4]
                    )
                    layer_result[ch] = {
                        "kendall": float(correlations["kendall"]),
                    }
                else:
                    correlations = calculate_spearman(b, ideal_ordering=[0, 1, 2, 3, 4])
                    layer_result[ch] = {
                        "spearman": float(correlations["spearman"]),
                        "kendall": float(correlations["kendall"]),
                        "pearson": float(correlations["pearson"]),
                    }
            results_per_layer.append(layer_result)

        return {self.name: json.dumps(results_per_layer)}


def create_prop2_calculator(
    backend: BackendEnum,
    channel: str = "all",
    data_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    batch_size: int = 32,
    include_kendall: bool = True,
) -> VisTuringCalculator:
    """
    Factory para crear un calculator de Prop2.

    Args:
        backend: Backend (torch o jax)
        channel: Canal a evaluar ('achrom', 'red_green', 'yellow_blue')
        data_path: Ruta a los datos
        gt_path: Ruta al ground truth
        batch_size: Tamaño de batch
        include_kendall: Si incluir métrica de Kendall/Spearman

    Returns:
        VisTuringCalculator configurado para Prop2
    """
    # Crear dataset loader
    dataset_loader = Prop2TorchDatasetLoader(
        channel=channel,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        data_path=data_path,
        use_torch_upstream_semantics=backend == BackendEnum.TORCH,
    )

    # Cargar datos para obtener x_values
    data, bgs, x_a, x_rg, x_yb = dataset_loader.load_data()
    x_values_map = {"achrom": x_a, "red_green": x_rg, "yellow_blue": x_yb}
    levels_map = {
        "achrom": data["achrom"].shape[0],
        "red_green": data["red_green"].shape[0],
        "yellow_blue": data["yellow_blue"].shape[0],
    }

    # Crear métricas
    metrics: list[BaseVisTuringMetric] = []
    metrics.append(
        WeberLawPearson(
            backend=backend,
            channel=channel,
            gt_path=gt_path,
            x_values_map=x_values_map,
            levels_map=levels_map,
        )
    )
    if include_kendall:
        metrics.append(
            WeberLawKendall(
                backend=backend,
                channel=channel,
                gt_path=gt_path,
                x_values_map=x_values_map,
                levels_map=levels_map,
            )
        )

    # Crear calculator
    calculator = VisTuringCalculator(
        backend=backend,
        metrics=metrics,
        dataset_loader=dataset_loader,
        name=f"Prop2Calculator_{channel}",
    )

    return calculator
