from .saliency import (
    SaliencyMIT1003DatasetLoader,
    SaliencyMIT1003TorchDatasetLoader,
)
from .tid import (
    TID2013DatasetLoader,
    TID2013TorchDatasetLoader,
)
from .levels import (
    LevelsDatasetLoader,
    LevelsTorchDatasetLoader,
)
from .nights import (
    NightsDatasetLoader,
    NightsTorchDatasetLoader,
)

__all__ = [
    "SaliencyMIT1003DatasetLoader",
    "SaliencyMIT1003TorchDatasetLoader",
    "TID2013DatasetLoader",
    "TID2013TorchDatasetLoader",
    "LevelsDatasetLoader",
    "LevelsTorchDatasetLoader",
    "NightsDatasetLoader",
    "NightsTorchDatasetLoader",
]
