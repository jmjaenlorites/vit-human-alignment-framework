from typing import Any, Literal

import torch


RepresentationPooling = Literal["flatten", "cls", "mean_patch"]


def representation_vector(
    features: Any,
    pooling: RepresentationPooling = "flatten",
) -> Any:
    """Return the vector used by representation-based metrics."""
    if pooling == "flatten":
        return features.flatten()

    if features.ndim < 2:
        return features.flatten()

    if pooling == "cls":
        return features[0].flatten()

    if pooling == "mean_patch":
        patch_tokens = features[1:]
        if isinstance(patch_tokens, torch.Tensor):
            return patch_tokens.mean(dim=0).flatten()
        return patch_tokens.mean(axis=0).flatten()

    raise ValueError(f"Unsupported representation_pooling: {pooling}")
