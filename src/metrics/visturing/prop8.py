"""Métrica para Propiedad 8: Contrast masking."""

import json
from typing import Any, Optional

import numpy as np
from scipy.stats import pearsonr

from .base import BaseVisTuringMetric, VisTuringCalculator
from .distance_functions import calculate_spearman, prepare_data
from .ground_truth import load_ground_truth_file
from ...dataset_loaders.visturing.prop8 import Prop8TorchDatasetLoader
from ...utils.common_enums import BackendEnum


class ContrastMaskingKendall(BaseVisTuringMetric):
    """
    Calcula correlación de Kendall para contrast masking.

    Evalúa el efecto de máscaras de contraste en la visibilidad.

    Note: When instantiated from CSV, uses default freq='low' (3 cpd).
    Future JSON configuration will allow specifying frequency and mask contrast.
    """

    name: str = "visturing_contrast_masking_kendall"

    def __init__(
        self,
        backend: BackendEnum,
        freq: str = "all",  # 'low', 'high', 'all'
        gt_path: Optional[str] = None,
        contrasts: Optional[np.ndarray] = None,
    ):
        super().__init__(backend)
        self.freq = freq
        self.gt_path = gt_path or "./data/visturing"
        self.contrasts = contrasts
        self._diffs_per_layer: list[dict[str, dict[str, list[list[float]]]]] = []
        try:
            self._load_ground_truth()
        except FileNotFoundError:
            pass

    def _load_ground_truth(self):
        """Carga datos de ground truth de contrast masking."""
        # Usa responses_no_mask_achrom_1p5_3_6_12_24.mat
        data = load_ground_truth_file(
            self.gt_path, "responses_no_mask_achrom_1p5_3_6_12_24.mat"
        )
        resp_data = data["resp_no_mask_achrom"]

        # resp_data[0] = x (contrasts)
        # resp_data[2] = 3 cpd (low)
        # resp_data[4] = 12 cpd (high)
        freq_idx = {"low": 2, "high": 4}

        self.ground_truth_data = {
            "x": resp_data[0],
            "low": resp_data[freq_idx["low"]],
            "high": resp_data[freq_idx["high"]],
        }

    def calculate(self, batch: Any, batch_results: Any) -> None:
        features_test = batch_results["features_test"]
        features_ref = batch_results["features_ref"]

        if len(batch) < 5:
            raise ValueError(
                "Prop8 requires batch metadata (c_group, mask, contrast_idx)"
            )
        c_groups = batch[2]
        masks = batch[3]
        contrast_idxs = batch[4]

        if hasattr(c_groups, "tolist"):
            c_groups = c_groups.tolist()
        if hasattr(masks, "tolist"):
            masks = masks.tolist()
        if hasattr(contrast_idxs, "tolist"):
            contrast_idxs = contrast_idxs.tolist()

        num_layers = len(features_test)
        if not self._diffs_per_layer:
            self._diffs_per_layer = []
            num_contrasts = len(self.contrasts) if self.contrasts is not None else 0
            for _ in range(num_layers):
                self._diffs_per_layer.append({"C1": {}, "C2": {}})
                for group in ["C1", "C2"]:
                    self._diffs_per_layer[-1][group] = {}
                    for mask in []:
                        self._diffs_per_layer[-1][group][mask] = [
                            [] for _ in range(num_contrasts)
                        ]

        for layer_idx in range(num_layers):
            feat_test = features_test[layer_idx]
            feat_ref = features_ref[layer_idx]

            if self._backend == BackendEnum.TORCH:
                diffs = self._calculate_diffs_torch(feat_test, feat_ref)
            elif self._backend == BackendEnum.JAX:
                diffs = self._calculate_diffs_jax(feat_test, feat_ref)
            else:
                raise NotImplementedError(f"Backend {self._backend} not supported")

            for idx, (group, mask, contrast_idx) in enumerate(
                zip(c_groups, masks, contrast_idxs)
            ):
                group = str(group)
                mask = str(mask)
                if mask not in self._diffs_per_layer[layer_idx][group]:
                    num_contrasts = (
                        len(self.contrasts) if self.contrasts is not None else 0
                    )
                    self._diffs_per_layer[layer_idx][group][mask] = [
                        [] for _ in range(num_contrasts)
                    ]
                self._diffs_per_layer[layer_idx][group][mask][int(contrast_idx)].append(
                    float(diffs[idx])
                )

    def _ordered_masks(self, masks: list[str]) -> list[str]:
        order = ["nomask", "0075", "0150", "0225", "0300"]
        normalized = {m: m.lower().replace("no_mask", "nomask") for m in masks}
        ranked = []
        for target in order:
            for orig, norm in normalized.items():
                if norm == target or norm.endswith(target):
                    ranked.append(orig)
        for m in masks:
            if m not in ranked:
                ranked.append(m)
        return ranked

    def finalize(self) -> dict[str, Any]:
        """
        Calcula correlaciones de Kendall por capa.

        Returns:
            Dict con correlaciones por capa
        """
        results_per_layer = []

        for layer_idx, layer_diffs in enumerate(self._diffs_per_layer):
            if not layer_diffs or self.contrasts is None:
                results_per_layer.append({})
                continue

            order_corr = {}
            b_low = None
            b_high = None

            for group, label in [("C1", "low"), ("C2", "high")]:
                masks = list(layer_diffs[group].keys())
                if not masks:
                    continue
                mask_order = self._ordered_masks(masks)
                diffs_per_mask = []
                bs = []
                for mask in mask_order:
                    diffs_lists = layer_diffs[group][mask]
                    diffs_per_contrast = np.array([np.mean(v) for v in diffs_lists])
                    diffs_per_mask.append(diffs_per_contrast)
                    _, b, _, _ = prepare_data(
                        self.contrasts,
                        diffs_per_contrast,
                        self.ground_truth_data["x"],
                        self.ground_truth_data[label],
                    )
                    bs.append(b)

                b_stack = np.array(bs)
                order_corr[label] = calculate_spearman(
                    b_stack, ideal_ordering=[0, 1, 2, 3, 4]
                )
                if label == "low":
                    b_low = b_stack
                else:
                    b_high = b_stack

            pearson_val = float("nan")
            if b_low is not None and b_high is not None:
                pearson_val = float(
                    pearsonr(
                        np.concatenate([b_low[0], b_high[0]]),
                        np.concatenate(
                            [
                                self.ground_truth_data["low"],
                                self.ground_truth_data["high"],
                            ]
                        ),
                    )[0]
                )

            results_per_layer.append(
                {
                    "kendall": {
                        "low": {
                            "kendall": float(
                                order_corr.get("low", {}).get("kendall", float("nan"))
                            ),
                            "spearman": float(
                                order_corr.get("low", {}).get("spearman", float("nan"))
                            ),
                            "pearson": float(
                                order_corr.get("low", {}).get("pearson", float("nan"))
                            ),
                        },
                        "high": {
                            "kendall": float(
                                order_corr.get("high", {}).get("kendall", float("nan"))
                            ),
                            "spearman": float(
                                order_corr.get("high", {}).get("spearman", float("nan"))
                            ),
                            "pearson": float(
                                order_corr.get("high", {}).get("pearson", float("nan"))
                            ),
                        },
                    },
                    "pearson": float(pearson_val),
                }
            )

        return {self.name: json.dumps(results_per_layer)}


def create_prop8_calculator(
    backend: BackendEnum,
    freq: str = "all",  # 'low', 'high', 'all'
    mask_contrast: str = "0075",  # '0075', '0150', '0225', '0300'
    data_path: Optional[str] = None,
    gt_path: Optional[str] = None,
    batch_size: int = 32,
) -> VisTuringCalculator:
    """
    Factory para crear un calculator de Prop8.

    Args:
        backend: Backend (torch o jax)
        freq: Frecuencia ('low' para 3cpd, 'high' para 12cpd)
        mask_contrast: Contraste de máscara ('0075', '0150', '0225', '0300')
        data_path: Ruta a los datos
        gt_path: Ruta al ground truth
        batch_size: Tamaño de batch

    Returns:
        VisTuringCalculator configurado para Prop8
    """
    # Mapear freq a C1/C2
    freq_map = {"low": "C1", "high": "C2"}
    if freq == "all":
        gabor_key = "all"
    else:
        gabor_key = f"a_{freq_map[freq]}_mask_{mask_contrast}"

    dataset_loader = Prop8TorchDatasetLoader(
        gabor_key=gabor_key,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        data_path=data_path,
    )

    # Cargar datos para obtener contrasts
    gabors, bgs, contrasts = dataset_loader.load_data()

    # Crear métrica
    metric = ContrastMaskingKendall(
        backend=backend, freq=freq, gt_path=gt_path, contrasts=contrasts
    )

    # Crear calculator
    calculator = VisTuringCalculator(
        backend=backend,
        metrics=[metric],
        dataset_loader=dataset_loader,
        name=f"Prop8Calculator_{freq}_mask_{mask_contrast}",
    )

    return calculator
