from typing import Optional, List, Any
import pandas as pd
from ..metrics.saliency import SaliencyMetricsCalculator

from ..utils.common_types import ExperimentSpec
from ..metrics.base import BaseMetric
from ..metrics import load_metric
from ..utils.common_enums import BackendEnum
from ..utils.common_utils import METRIC_PREFIX
from ..models import load_model

class Runner:
    def __init__(self, csv_path: str, output_path: Optional[str] = None, backend: BackendEnum = BackendEnum.TORCH):
        self.csv_path = csv_path
        self.output_path = output_path
        self.backend = backend

    def load_experiment_specs(self) -> tuple[List[ExperimentSpec], pd.DataFrame]:
        df = self._load_csv()
        experiment_specs = []
        for index, row in df.iterrows():
            metric_columns = [col for col in df.columns if col.startswith(METRIC_PREFIX)]
            pending_metrics = self.pending_metrics(row, metric_columns)
            extra_config = row.to_dict()
            experiment_specs.append(ExperimentSpec(model_name=row["model_name"], metric_columns=metric_columns, pending_metrics=pending_metrics, extra_config=extra_config, row_index=index))
        return experiment_specs, df

    def pending_metrics(self, row: pd.Series, metric_columns: List[str]) -> List[BaseMetric]:
        """Devuelve columnas de métricas vacías (pendientes)."""
        return [load_metric(metric_col) for metric_col in metric_columns if self._is_cell_empty(row[metric_col])]

    def execute(self) -> None:
        experiment_specs, df = self.load_experiment_specs()
        for experiment_spec in experiment_specs:
            print(experiment_spec)

            if not experiment_spec.pending_metrics:
                continue

            model = load_model(experiment_spec.model_name)
            # Create the metric instances with the backend that match the model backend
            pending_metrics = [metric(model.backend) for metric in experiment_spec.pending_metrics]

            saliency_metrics = [metric for metric in pending_metrics if metric.type == "saliency"]
            perceptual_metrics = [metric for metric in pending_metrics if metric.type == "perceptual"]

            if saliency_metrics:
                saliency_metrics_calculator = SaliencyMetricsCalculator(self.backend, saliency_metrics)
                saliency_results = saliency_metrics_calculator.run(model)
                self._update_cells(df, experiment_spec.row_index, [metric.name for metric in saliency_metrics], [saliency_results[metric.name] for metric in saliency_metrics])

            # if perceptual_metrics:
            #     perceptual_metrics_calculator = PerceptualMetricsCalculator(self.backend, perceptual_metrics)
            #     perceptual_results = perceptual_metrics_calculator.run(model)
            #     for metric in perceptual_metrics:
            #         self._update_cell(df, experiment_spec.row_index, metric.name, perceptual_results[metric.name])


    def _load_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def _write_csv(self, df: pd.DataFrame) -> None:
        if self.output_path:
            df.to_csv(self.output_path, index=False)
        else:
            df.to_csv(self.csv_path, index=False)

    def _update_cells(self, df: pd.DataFrame, row_index: int, metric_names: List[str], results: List[Any], save_csv: bool = True) -> None:
        for metric_name, result in zip(metric_names, results):
            if not result:
                continue
            df.loc[row_index, METRIC_PREFIX +metric_name] = result
        if save_csv:
            self._write_csv(df)

    def _is_cell_empty(self, cell: Any) -> bool:
        if cell is None:
            return True
        if isinstance(cell, str):
            cell_lower = cell.lower()
            if cell_lower == "none" or cell_lower == "nan" or cell_lower == "null" or cell_lower == "":
                return True
        if isinstance(cell, float):
            return pd.isna(cell)
        return False
