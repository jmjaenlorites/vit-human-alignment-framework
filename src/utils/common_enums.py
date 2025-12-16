from enum import StrEnum

class BackendEnum(StrEnum):
    TORCH = "torch"
    JAX = "jax"
    NUMPY = "numpy"