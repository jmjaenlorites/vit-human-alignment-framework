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
from .visturing.prop2 import WeberLawPearson, WeberLawKendall
from .visturing.prop3_4 import CSFPearson, CSFKendall
from .visturing.prop5 import CampbellBlakemorePearson, CampbellBlakemoreKendall
from .visturing.prop6_7 import ContrastCurvesPearson, ContrastCurvesKendall
from .visturing.prop8 import ContrastMaskingKendall
from .visturing.prop9 import FrequencyMaskingKendall
from .visturing.prop10 import OrientationMaskingKendall


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
        # Visturing metrics
        case SpectralSensitivityPearson.name:
            return lambda backend: SpectralSensitivityPearson(backend)
        case WeberLawPearson.name:
            return lambda backend: WeberLawPearson(backend)
        case WeberLawKendall.name:
            return lambda backend: WeberLawKendall(backend)
        case CSFPearson.name:
            return lambda backend: CSFPearson(backend)
        case CSFKendall.name:
            return lambda backend: CSFKendall(backend)
        case CampbellBlakemorePearson.name:
            return lambda backend: CampbellBlakemorePearson(backend)
        case CampbellBlakemoreKendall.name:
            return lambda backend: CampbellBlakemoreKendall(backend)
        case ContrastCurvesPearson.name:
            return lambda backend: ContrastCurvesPearson(backend)
        case ContrastCurvesKendall.name:
            return lambda backend: ContrastCurvesKendall(backend)
        case ContrastMaskingKendall.name:
            return lambda backend: ContrastMaskingKendall(backend)
        case FrequencyMaskingKendall.name:
            return lambda backend: FrequencyMaskingKendall(backend)
        case OrientationMaskingKendall.name:
            return lambda backend: OrientationMaskingKendall(backend)
        case _:
            raise ValueError(f"Metric {metric_name} not supported")
