from .base import BaseModelAdapter
from .vit_b16 import ViT_B_16

def load_model(model_name: str) -> BaseModelAdapter:
    match model_name:
        case "vit-b16":
            return ViT_B_16()
        case _:
            raise ValueError(f"Model {model_name} not supported")