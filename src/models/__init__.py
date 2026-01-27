from .base import BaseModelAdapter
from .vit_b16 import ViT_B_16, ViT_B_32, ViT_H_14, ViT_L_14


def load_model(model_name: str) -> BaseModelAdapter:
    match model_name:
        case "vit-b16":
            return ViT_B_16()
        case "vit-l14":
            return ViT_L_14()
        case "vit-h14":
            return ViT_H_14()
        case "vit-b32":
            return ViT_B_32()
        case _:
            raise ValueError(f"Model {model_name} not supported")
