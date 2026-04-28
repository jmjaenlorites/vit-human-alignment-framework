"""Pipeline de preprocesado para los modelos Google ViT.

Replica la receta de evaluación estándar (resize-small → center-crop) y la
normalización a [-1, 1] (`mean=std=0.5`) que usa `vit_jax/preprocess.py`.

Devuelve un `torchvision.transforms.Compose` para mantener la compatibilidad
con la introspección que hace `SaliencyMIT1003Dataset` (filtra `Normalize`
para preparar los GT). El adaptador JAX recibe los tensores torch ya
preprocesados y los convierte a `jnp.ndarray` (NHWC) en el forward.
"""

import torchvision.transforms as T

_RESIZE_RATIO: float = 256 / 224


def make_torchvision_transform(image_size: int) -> T.Compose:
    """Compose: resize-small (ratio 256/224) → center-crop → normalize a [-1, 1]."""
    resize_size = int(round(image_size * _RESIZE_RATIO))
    return T.Compose(
        [
            T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
