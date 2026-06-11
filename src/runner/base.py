import json
import logging
from typing import Any, List, Optional

import pandas as pd

from ..metrics import load_metric
from ..metrics.base import BaseMetric
from ..metrics.levels import LevelsMetricsCalculator
from ..metrics.nights import NightsMetricsCalculator
from ..metrics.saliency import SaliencyMetricsCalculator
from ..metrics.things import ThingsMetricsCalculator
from ..metrics.tid import TIDMetricsCalculator
from ..metrics.visturing import (
    create_prop1_calculator,
    create_prop2_calculator,
    create_prop3_4_calculator,
    create_prop5_calculator,
    create_prop6_7_calculator,
    create_prop8_calculator,
    create_prop9_calculator,
    create_prop10_calculator,
)
from ..models import resolve_model
from ..utils.common_enums import BackendEnum
from ..utils.common_types import ExperimentSpec
from ..utils.common_utils import METRIC_PREFIX
from .config_schema import get_metric_spec
from .json_source import JSONExperimentSource
from .results_store import ResultsStore
from .types import ResolvedExperiment

logger = logging.getLogger(__name__)


class Runner:
    def __init__(
        self,
        csv_path: Optional[str] = None,
        json_path: Optional[str] = None,
        output_path: Optional[str] = None,
        results_path: Optional[str] = None,
        backend: BackendEnum = BackendEnum.TORCH,
    ):
        # NOTE: `backend` se conserva por retrocompatibilidad de la firma. Los
        # calculators y métricas se construyen con `model.backend` (inferido
        # por modelo) — permite mezclar modelos torch y JAX en el mismo JSON.
        self.csv_path = csv_path
        self.json_path = json_path
        self.output_path = output_path
        self.results_path = results_path
        self.backend = backend

    def load_experiment_specs(self) -> tuple[List[ExperimentSpec], pd.DataFrame]:
        """
        Carga especificaciones de experimentos desde CSV.

        TODO: Migrate to JSON format to support metric-specific configuration.
        Current CSV format only supports metric names in columns (e.g., metric_tid_spearman_mos).
        This limits configuration options for metrics like:
        - Levels: Can't specify split (uses default "between_class")
        - Visturing: Can't specify frequency/mask parameters per metric in CSV

        Future JSON format will allow:
        {
            "model_name": "vit_b16",
            "metrics": [
                {"name": "levels_triplet_accuracy", "config": {"split": "class_border"}},
                {"name": "visturing_frequency_masking_kendall", "config": {"mask_freq": "3"}},
                ...
            ]
        }
        """
        df = self._load_csv()
        experiment_specs = []
        for index, row in df.iterrows():
            metric_columns = [
                col for col in df.columns if col.startswith(METRIC_PREFIX)
            ]
            pending_metrics = self.pending_metrics(row, metric_columns)
            extra_config = row.to_dict()
            experiment_specs.append(
                ExperimentSpec(
                    model_name=row["model_name"],
                    metric_columns=metric_columns,
                    pending_metrics=pending_metrics,
                    extra_config=extra_config,
                    row_index=index,
                )
            )
        return experiment_specs, df

    def pending_metrics(
        self, row: pd.Series, metric_columns: List[str]
    ) -> List[BaseMetric]:
        """Devuelve columnas de métricas vacías (pendientes)."""
        return [
            load_metric(metric_col)
            for metric_col in metric_columns
            if self._is_cell_empty(row[metric_col])
        ]

    def execute(self) -> None:
        if self.json_path:
            self._execute_json()
            return

        if not self.csv_path:
            raise ValueError("csv_path is required when json_path is not provided")

        experiment_specs, df = self.load_experiment_specs()
        for experiment_spec in experiment_specs:
            logger.info("Experiment spec: %s", experiment_spec)

            if not experiment_spec.pending_metrics:
                continue

            model = resolve_model(experiment_spec.model_name)
            # Create the metric instances with the backend that match the model backend
            pending_metrics = [
                metric(model.backend) for metric in experiment_spec.pending_metrics
            ]

            saliency_metrics = [
                metric for metric in pending_metrics if metric.type == "saliency"
            ]
            perceptual_metrics = [
                metric for metric in pending_metrics if metric.type == "perceptual"
            ]
            visturing_metrics = [
                metric for metric in pending_metrics if metric.type == "visturing"
            ]
            tid_metrics = [
                metric
                for metric in perceptual_metrics
                if metric.name.startswith("tid_")
            ]
            levels_metrics = [
                metric
                for metric in perceptual_metrics
                if metric.name.startswith("levels_")
            ]
            nights_metrics = [
                metric
                for metric in perceptual_metrics
                if metric.name.startswith("nights_")
            ]
            things_metrics = [
                metric
                for metric in perceptual_metrics
                if metric.name.startswith("things_")
            ]

            # Visturing metrics: run the corresponding property calculators with defaults
            if visturing_metrics:
                visturing_names = {metric.name for metric in visturing_metrics}
                visturing_results = {}

                if "visturing_spectral_sensitivity" in visturing_names:
                    calc = create_prop1_calculator(model.backend)
                    visturing_results.update(calc.run(model))

                if (
                    "visturing_weber_law_pearson" in visturing_names
                    or "visturing_weber_law_kendall" in visturing_names
                ):
                    calc = create_prop2_calculator(
                        model.backend,
                        include_kendall="visturing_weber_law_kendall"
                        in visturing_names,
                    )
                    visturing_results.update(calc.run(model))

                if (
                    "visturing_csf_pearson" in visturing_names
                    or "visturing_csf_kendall" in visturing_names
                ):
                    calc = create_prop3_4_calculator(
                        model.backend,
                        include_kendall="visturing_csf_kendall" in visturing_names,
                    )
                    visturing_results.update(calc.run(model))

                if (
                    "visturing_campbell_blakemore_pearson" in visturing_names
                    or "visturing_campbell_blakemore_kendall" in visturing_names
                ):
                    calc = create_prop5_calculator(
                        model.backend,
                        include_kendall="visturing_campbell_blakemore_kendall"
                        in visturing_names,
                    )
                    visturing_results.update(calc.run(model))

                if (
                    "visturing_contrast_curves_pearson" in visturing_names
                    or "visturing_contrast_curves_kendall" in visturing_names
                ):
                    calc = create_prop6_7_calculator(
                        model.backend,
                        include_kendall="visturing_contrast_curves_kendall"
                        in visturing_names,
                    )
                    visturing_results.update(calc.run(model))

                if "visturing_contrast_masking_kendall" in visturing_names:
                    calc = create_prop8_calculator(model.backend)
                    visturing_results.update(calc.run(model))

                if "visturing_frequency_masking_kendall" in visturing_names:
                    calc = create_prop9_calculator(model.backend)
                    visturing_results.update(calc.run(model))

                if "visturing_orientation_masking_kendall" in visturing_names:
                    calc = create_prop10_calculator(model.backend)
                    visturing_results.update(calc.run(model))

                self._update_cells(
                    df,
                    experiment_spec.row_index,
                    [metric.name for metric in visturing_metrics],
                    [
                        visturing_results.get(metric.name)
                        for metric in visturing_metrics
                    ],
                )

            if saliency_metrics:
                saliency_metrics_calculator = SaliencyMetricsCalculator(
                    model.backend, saliency_metrics
                )
                saliency_results = saliency_metrics_calculator.run(model)
                self._update_cells(
                    df,
                    experiment_spec.row_index,
                    [metric.name for metric in saliency_metrics],
                    [saliency_results[metric.name] for metric in saliency_metrics],
                )

            if tid_metrics:
                logger.info(
                    "Running TID metrics: %s",
                    [metric.name for metric in tid_metrics],
                )
                tid_metrics_calculator = TIDMetricsCalculator(
                    model.backend, metrics=tid_metrics
                )
                tid_results = tid_metrics_calculator.run(model)
                self._update_cells(
                    df,
                    experiment_spec.row_index,
                    [metric.name for metric in tid_metrics],
                    [tid_results[metric.name] for metric in tid_metrics],
                )

            # Levels metrics: Currently uses default split "between_class"
            # TODO: Migrate from CSV to JSON configuration to allow specifying split
            #   (between_class/class_border/within_class) and dataset paths
            if levels_metrics:
                logger.info(
                    "Running Levels metrics with default split 'between_class': %s",
                    [metric.name for metric in levels_metrics],
                )
                levels_metrics_calculator = LevelsMetricsCalculator(
                    model.backend, metrics=levels_metrics
                )
                levels_results = levels_metrics_calculator.run(model)
                self._update_cells(
                    df,
                    experiment_spec.row_index,
                    [metric.name for metric in levels_metrics],
                    [levels_results[metric.name] for metric in levels_metrics],
                )

            if nights_metrics:
                logger.info(
                    "Running Nights metrics: %s",
                    [metric.name for metric in nights_metrics],
                )
                nights_metrics_calculator = NightsMetricsCalculator(
                    model.backend, metrics=nights_metrics
                )
                nights_results = nights_metrics_calculator.run(model)
                self._update_cells(
                    df,
                    experiment_spec.row_index,
                    [metric.name for metric in nights_metrics],
                    [nights_results[metric.name] for metric in nights_metrics],
                )

            if things_metrics:
                logger.info(
                    "Running Things metrics: %s",
                    [metric.name for metric in things_metrics],
                )
                things_metrics_calculator = ThingsMetricsCalculator(
                    model.backend, metrics=things_metrics
                )
                things_results = things_metrics_calculator.run(model)
                self._update_cells(
                    df,
                    experiment_spec.row_index,
                    [metric.name for metric in things_metrics],
                    [things_results[metric.name] for metric in things_metrics],
                )

    def _execute_json(self) -> None:
        if not self.json_path:
            raise ValueError("json_path is required for JSON execution")
        if not self.results_path:
            raise ValueError("results_path is required for JSON execution")

        results_store = ResultsStore(self.results_path)
        experiments = JSONExperimentSource(self.json_path).load()

        # Cache de un solo slot: cargamos un modelo, gastamos todas sus
        # métricas y lo liberamos antes de pasar al siguiente. La iteración en
        # `JSONExperimentSource.load` viene model-major, así que cada modelo
        # se carga UNA vez. Liberamos GPU explícitamente al cambiar de modelo
        # para que torch y jax no compitan por la VRAM.
        current_model_name: Optional[str] = None
        current_model: Any = None

        for experiment in experiments:
            logger.info("Resolved experiment: %s", experiment)
            if not results_store.should_run(experiment):
                continue

            if experiment.model_name != current_model_name:
                if current_model is not None:
                    self._release_model(current_model)
                    current_model = None
                    current_model_name = None
                current_model = resolve_model(experiment.model_name)
                current_model_name = experiment.model_name

            results_store.mark_running(experiment)
            try:
                result = self._run_metric_experiment(experiment, current_model)
            except Exception as exc:
                results_store.mark_error(experiment, str(exc))
                if _is_out_of_memory_error(exc):
                    # OOM en una métrica concreta no debe tumbar el run entero:
                    # marcamos la fila como error y seguimos. El usuario puede
                    # re-ejecutar esa métrica con menos batch_size, otra GPU,
                    # etc. sin perder el resto de experimentos.
                    logger.warning(
                        "Out-of-memory while running %s on %s — marked as "
                        "error and continuing with the next experiment. "
                        "(error: %s)",
                        experiment.experiment_id,
                        experiment.model_name,
                        exc,
                    )
                    continue
                raise

            results_store.mark_done(experiment, result)

        if current_model is not None:
            self._release_model(current_model)

    def _release_model(self, model: Any) -> None:
        """Libera la VRAM ocupada por un modelo antes de cargar el siguiente."""
        backend = getattr(model, "backend", None)
        try:
            if backend == BackendEnum.TORCH:
                self._release_torch_model(model)
            elif backend == BackendEnum.JAX:
                self._release_jax_model(model)
        except Exception:
            logger.exception(
                "Failed to release model %r cleanly; continuing.",
                getattr(model, "name", "<unknown>"),
            )

    def _release_torch_model(self, model: Any) -> None:
        import torch

        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "to"):
            try:
                inner.to("cpu")
            except Exception:
                pass
        del inner
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _release_jax_model(self, model: Any) -> None:
        import jax

        # Borrar params explícitamente (los `jax.Array` en GPU se liberan al
        # caer la última referencia).
        inner = getattr(model, "model", None)
        if isinstance(inner, dict):
            inner.pop("params", None)
        del inner
        del model
        try:
            jax.clear_caches()
        except Exception:
            pass

    def _run_metric_experiment(
        self,
        experiment: ResolvedExperiment,
        model: Any,
    ) -> Any:
        metric_spec = get_metric_spec(experiment.metric_name)

        if metric_spec.family == "visturing":
            return self._run_visturing_metric(experiment, model)

        metric = load_metric(experiment.metric_name)(model.backend)

        if metric_spec.family == "saliency":
            calculator = SaliencyMetricsCalculator(
                model.backend,
                [metric],
                dataset_path=experiment.config.get("dataset_path"),
                batch_size=experiment.config.get("batch_size", 32),
            )
            return calculator.run(model)[metric.name]

        if metric_spec.family == "tid":
            calculator = TIDMetricsCalculator(
                model.backend,
                metrics=[metric],
                dataset_path=experiment.config.get("dataset_path"),
            )
            return calculator.run(model)[metric.name]

        if metric_spec.family == "levels":
            calculator = LevelsMetricsCalculator(
                model.backend,
                split=experiment.config.get("split", "between_class"),
                metrics=[metric],
                levels_path=experiment.config.get("levels_path"),
                imagenet_path=experiment.config.get("imagenet_path"),
            )
            return calculator.run(model)[metric.name]

        if metric_spec.family == "nights":
            calculator = NightsMetricsCalculator(
                model.backend,
                metrics=[metric],
                dataset_path=experiment.config.get("dataset_path"),
            )
            return calculator.run(model)[metric.name]

        if metric_spec.family == "things":
            calculator = ThingsMetricsCalculator(
                model.backend,
                metrics=[metric],
                dataset_path=experiment.config.get("dataset_path"),
                batch_size=experiment.config.get("batch_size", 32),
            )
            return calculator.run(model)[metric.name]

        raise ValueError(f"Metric {experiment.metric_name} not supported")

    def _run_visturing_metric(self, experiment: ResolvedExperiment, model: Any) -> Any:
        config = experiment.config
        metric_name = experiment.metric_name

        if metric_name == "visturing_spectral_sensitivity":
            calculator = create_prop1_calculator(
                model.backend,
                data_path=config.get("data_path"),
                gt_path=config.get("gt_path"),
                batch_size=config.get("batch_size", 32),
            )
        elif metric_name in {
            "visturing_weber_law_pearson",
            "visturing_weber_law_kendall",
        }:
            calculator = create_prop2_calculator(
                model.backend,
                channel=config.get("channel", "all"),
                data_path=config.get("data_path"),
                gt_path=config.get("gt_path"),
                batch_size=config.get("batch_size", 32),
                include_kendall=metric_name == "visturing_weber_law_kendall",
            )
        elif metric_name in {"visturing_csf_pearson", "visturing_csf_kendall"}:
            calculator = create_prop3_4_calculator(
                model.backend,
                channel=config.get("channel", "all"),
                data_path=config.get("data_path"),
                gt_path=config.get("gt_path"),
                batch_size=config.get("batch_size", 32),
                include_kendall=metric_name == "visturing_csf_kendall",
            )
        elif metric_name in {
            "visturing_campbell_blakemore_pearson",
            "visturing_campbell_blakemore_kendall",
        }:
            calculator = create_prop5_calculator(
                model.backend,
                mask_freq=config.get("mask_freq", "all"),
                data_path=config.get("data_path"),
                gt_path=config.get("gt_path"),
                batch_size=config.get("batch_size", 32),
                include_kendall=metric_name == "visturing_campbell_blakemore_kendall",
            )
        elif metric_name in {
            "visturing_contrast_curves_pearson",
            "visturing_contrast_curves_kendall",
        }:
            calculator = create_prop6_7_calculator(
                model.backend,
                channel=config.get("channel", "all"),
                freq=config.get("freq", "all"),
                data_path=config.get("data_path"),
                gt_path=config.get("gt_path"),
                batch_size=config.get("batch_size", 32),
                include_kendall=metric_name == "visturing_contrast_curves_kendall",
            )
        elif metric_name == "visturing_contrast_masking_kendall":
            calculator = create_prop8_calculator(
                model.backend,
                freq=config.get("freq", "all"),
                mask_contrast=config.get("mask_contrast", "0075"),
                data_path=config.get("data_path"),
                gt_path=config.get("gt_path"),
                batch_size=config.get("batch_size", 32),
            )
        elif metric_name == "visturing_frequency_masking_kendall":
            calculator = create_prop9_calculator(
                model.backend,
                freq=config.get("freq", "all"),
                mask_freq=config.get("mask_freq", "1p5"),
                data_path=config.get("data_path"),
                gt_path=config.get("gt_path"),
                batch_size=config.get("batch_size", 32),
            )
        elif metric_name == "visturing_orientation_masking_kendall":
            calculator = create_prop10_calculator(
                model.backend,
                freq=config.get("freq", "all"),
                mask_orientation=config.get("mask_orientation", "0"),
                data_path=config.get("data_path"),
                gt_path=config.get("gt_path"),
                batch_size=config.get("batch_size", 32),
            )
        else:
            raise ValueError(f"Metric {metric_name} not supported")

        return calculator.run(model)[metric_name]

    def _load_csv(self) -> pd.DataFrame:
        if not self.csv_path:
            raise ValueError("csv_path is required for CSV execution")
        return pd.read_csv(self.csv_path)

    def _write_csv(self, df: pd.DataFrame) -> None:
        if self.output_path:
            df.to_csv(self.output_path, index=False)
        else:
            df.to_csv(self.csv_path, index=False)

    def _update_cells(
        self,
        df: pd.DataFrame,
        row_index: int,
        metric_names: List[str],
        results: List[Any],
        save_csv: bool = True,
    ) -> None:
        for metric_name, result in zip(metric_names, results):
            if not result:
                continue
            df.loc[row_index, METRIC_PREFIX + metric_name] = result
            logger.info("Updated metric %s", metric_name)
        if save_csv:
            self._write_csv(df)

    def _is_cell_empty(self, cell: Any) -> bool:
        if cell is None:
            return True
        if isinstance(cell, str):
            cell_lower = cell.lower()
            if (
                cell_lower == "none"
                or cell_lower == "nan"
                or cell_lower == "null"
                or cell_lower == ""
            ):
                return True
            if cell_lower.startswith("[") and "nan" in cell_lower:
                try:
                    normalized = cell_lower.replace("nan", "null")
                    parsed = json.loads(normalized)
                    if isinstance(parsed, list) and all(v is None for v in parsed):
                        return True
                except json.JSONDecodeError:
                    pass
        if isinstance(cell, float):
            return pd.isna(cell)
        return False


# Marcadores de OOM observados en la práctica con torch CUDA y JAX (XLA).
# La detección es por nombre de clase + sustring del mensaje para no acoplar
# al runner imports condicionales de torch/jax que pueden no estar disponibles.
_OOM_EXCEPTION_NAMES: frozenset[str] = frozenset(
    {
        "OutOfMemoryError",            # torch.cuda.OutOfMemoryError
        "JaxRuntimeError",             # jax.errors.JaxRuntimeError
        "XlaRuntimeError",             # jaxlib.xla_extension.XlaRuntimeError
        "RuntimeError",                # caída común con mensaje "out of memory"
    }
)
_OOM_MESSAGE_MARKERS: tuple[str, ...] = (
    "out of memory",
    "RESOURCE_EXHAUSTED",
    "CUDA out of memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
)


def _is_out_of_memory_error(exc: BaseException) -> bool:
    """Heurística para detectar OOM en torch/JAX sin importar nada.

    Revisa el nombre de la clase de la excepción y el mensaje. Esto cubre:
    - `torch.cuda.OutOfMemoryError` (torch >= 1.13)
    - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory ...`
    - `RuntimeError: CUDA out of memory ...` (torch antiguos)
    """
    if exc.__class__.__name__ not in _OOM_EXCEPTION_NAMES:
        # Recorremos también la cadena de causas (`__cause__` / `__context__`)
        # por si el OOM viene envuelto.
        cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        if cause is not None and cause is not exc:
            return _is_out_of_memory_error(cause)
        return False
    message = str(exc)
    return any(marker.lower() in message.lower() for marker in _OOM_MESSAGE_MARKERS)
