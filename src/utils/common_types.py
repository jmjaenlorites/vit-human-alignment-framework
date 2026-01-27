from typing import TYPE_CHECKING, Any
from dataclasses import dataclass
import numpy as np
import torch
import jax.numpy as jnp

if TYPE_CHECKING:
    from ..metrics.base import BaseMetric

ArrayLike = (
    np.ndarray
    | torch.Tensor
    | jnp.ndarray
    | list[np.ndarray]
    | list[torch.Tensor]
    | list[jnp.ndarray]
)


@dataclass
class ExperimentSpec:
    model_name: str
    metric_columns: list[str]
    pending_metrics: list["BaseMetric"]
    extra_config: dict[str, Any]
    row_index: int
