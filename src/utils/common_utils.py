from ..utils.common_enums import BackendEnum
import torch
import jax

METRIC_PREFIX: str = "metric_"

def get_backend_device(backend: BackendEnum):
    match backend:
        case BackendEnum.TORCH:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case BackendEnum.JAX:
            return jax.devices("gpu")[0]
        case _:
            raise ValueError(f"Backend {backend} not supported")