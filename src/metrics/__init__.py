from typing import Callable

from ..utils.common_enums import BackendEnum
from ..utils.common_utils import METRIC_PREFIX
from .base import BaseMetric
from .levels import LevelsMetricsCalculator, TripletAccuracy
from .nights import NightsMetricsCalculator, PreferenceAccuracy
from .saliency import AUC_Judd, PearsonCorrelationCoefficient, SaliencyMetricsCalculator
from .tid import SpearmanCorrelationMOS, TIDMetricsCalculator


def load_metric(metric_name: str) -> Callable[[BackendEnum], BaseMetric]:
    if metric_name.startswith(METRIC_PREFIX):
        metric_name = metric_name[len(METRIC_PREFIX) :]
    match metric_name:
        case AUC_Judd.name:
            return lambda backend: AUC_Judd(backend)
        case PearsonCorrelationCoefficient.name:
            return lambda backend: PearsonCorrelationCoefficient(backend)
        case SpearmanCorrelationMOS.name:
            return lambda backend: SpearmanCorrelationMOS(backend)
        case TripletAccuracy.name:
            return lambda backend: TripletAccuracy(backend)
        case PreferenceAccuracy.name:
            return lambda backend: PreferenceAccuracy(backend)
        case _:
            raise ValueError(f"Metric {metric_name} not supported")
