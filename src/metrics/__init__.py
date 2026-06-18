from typing import Callable

from ..utils.common_enums import BackendEnum
from ..utils.common_utils import METRIC_PREFIX
from .base import BaseMetric
from .levels import LevelsMetricsCalculator, TripletAccuracy
from .nights import NightsMetricsCalculator, PreferenceAccuracy
from .saliency import AUC_Judd, PearsonCorrelationCoefficient, SaliencyMetricsCalculator
from .tid import SpearmanCorrelationMOS, TIDMetricsCalculator
from .visturing.base import BaseVisTuringMetric, VisTuringCalculator
from .visturing.prop1 import SpectralSensitivityPearson
from .visturing.prop10 import OrientationMaskingKendall
from .visturing.prop2 import WeberLawKendall, WeberLawPearson
from .visturing.prop3_4 import CSFKendall, CSFPearson
from .visturing.prop5 import CampbellBlakemoreKendall, CampbellBlakemorePearson
from .visturing.prop6_7 import ContrastCurvesKendall, ContrastCurvesPearson
from .visturing.prop8 import ContrastMaskingKendall
from .visturing.prop9 import FrequencyMaskingKendall

METRIC_LOADERS: dict[str, Callable[[BackendEnum], BaseMetric]] = {
    AUC_Judd.name: lambda backend: AUC_Judd(backend),
    PearsonCorrelationCoefficient.name: lambda backend: PearsonCorrelationCoefficient(
        backend
    ),
    SpearmanCorrelationMOS.name: lambda backend: SpearmanCorrelationMOS(backend),
    TripletAccuracy.name: lambda backend: TripletAccuracy(backend),
    PreferenceAccuracy.name: lambda backend: PreferenceAccuracy(backend),
    SpectralSensitivityPearson.name: lambda backend: SpectralSensitivityPearson(
        backend
    ),
    WeberLawPearson.name: lambda backend: WeberLawPearson(backend),
    WeberLawKendall.name: lambda backend: WeberLawKendall(backend),
    CSFPearson.name: lambda backend: CSFPearson(backend),
    CSFKendall.name: lambda backend: CSFKendall(backend),
    CampbellBlakemorePearson.name: lambda backend: CampbellBlakemorePearson(backend),
    CampbellBlakemoreKendall.name: lambda backend: CampbellBlakemoreKendall(backend),
    ContrastCurvesPearson.name: lambda backend: ContrastCurvesPearson(backend),
    ContrastCurvesKendall.name: lambda backend: ContrastCurvesKendall(backend),
    ContrastMaskingKendall.name: lambda backend: ContrastMaskingKendall(backend),
    FrequencyMaskingKendall.name: lambda backend: FrequencyMaskingKendall(backend),
    OrientationMaskingKendall.name: lambda backend: OrientationMaskingKendall(backend),
}


def list_supported_metric_names() -> list[str]:
    return sorted(METRIC_LOADERS)


def load_metric(metric_name: str) -> Callable[[BackendEnum], BaseMetric]:
    if metric_name.startswith(METRIC_PREFIX):
        metric_name = metric_name[len(METRIC_PREFIX) :]
    try:
        return METRIC_LOADERS[metric_name]
    except KeyError as exc:
        raise ValueError(f"Metric {metric_name} not supported") from exc
