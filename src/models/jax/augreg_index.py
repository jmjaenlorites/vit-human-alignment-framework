"""Resolución de modelos Google ViT publicados en la colección AugReg.

Indexa el CSV oficial (`vit_models/augreg/index.csv`) y traduce un filename
(pretrain o adapt) a la información necesaria para instanciar el modelo Flax
y descargar el checkpoint `.npz`.
"""

import io
import logging
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

GOOGLE_AUGREG_BASE_URL = "https://storage.googleapis.com/vit_models/augreg"
GOOGLE_AUGREG_INDEX_URL = f"{GOOGLE_AUGREG_BASE_URL}/index.csv"


@dataclass(frozen=True)
class _ArchSpec:
    patch_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_dim: int


# Dimensiones por arquitectura ViT (token classifier).
# Replica `vision_transformer/vit_jax/configs/models.py` para los archs soportados.
_ARCH_TABLE: dict[str, _ArchSpec] = {
    "Ti/16": _ArchSpec(16, 192, 12, 3, 768),
    "Ti/8": _ArchSpec(8, 192, 12, 3, 768),
    "S/16": _ArchSpec(16, 384, 12, 6, 1536),
    "S/32": _ArchSpec(32, 384, 12, 6, 1536),
    "S/8": _ArchSpec(8, 384, 12, 6, 1536),
    "B/16": _ArchSpec(16, 768, 12, 12, 3072),
    "B/32": _ArchSpec(32, 768, 12, 12, 3072),
    "B/8": _ArchSpec(8, 768, 12, 12, 3072),
    "L/16": _ArchSpec(16, 1024, 24, 16, 4096),
    "L/32": _ArchSpec(32, 1024, 24, 16, 4096),
}

# Núm. clases por dataset, deducido del param-tree publicado.
_PRETRAIN_NUM_CLASSES: dict[str, int] = {
    "i1k": 1000,
    "i21k": 21843,
    "i21k_30": 21843,
}
_ADAPT_NUM_CLASSES: dict[str, int] = {
    "imagenet2012": 1000,
    "cifar100": 100,
    "kitti": 4,
    "oxford_iiit_pet": 37,
    "resisc45": 45,
}


@dataclass(frozen=True)
class GoogleModelInfo:
    arch: str
    patch_size: int
    image_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    num_classes: int
    checkpoint_url: str
    is_adapt: bool


def _download_index_csv() -> bytes:
    logger.info("Downloading AugReg index from %s", GOOGLE_AUGREG_INDEX_URL)
    with urllib.request.urlopen(GOOGLE_AUGREG_INDEX_URL) as response:
        return response.read()


@lru_cache(maxsize=1)
def load_augreg_dataframe() -> pd.DataFrame:
    """Descarga (o recupera de cache de proceso) el index.csv de AugReg."""
    raw = _download_index_csv()
    return pd.read_csv(io.BytesIO(raw))


def _is_supported_arch(arch: str) -> bool:
    return arch in _ARCH_TABLE


@lru_cache(maxsize=1)
def list_available_google_models() -> list[str]:
    """Lista ordenada de filenames AugReg para arquitecturas ViT soportadas."""
    df = load_augreg_dataframe()
    supported = df[df["name"].apply(_is_supported_arch)]
    names: set[str] = set()
    names.update(supported["filename"].dropna().astype(str).tolist())
    names.update(supported["adapt_filename"].dropna().astype(str).tolist())
    return sorted(names)


def lookup_google_model(filename: str) -> GoogleModelInfo:
    """Resuelve un filename AugReg (pretrain o adapt) a su `GoogleModelInfo`."""
    df = load_augreg_dataframe()

    adapt_match = df[df["adapt_filename"] == filename]
    if len(adapt_match) > 0:
        row = adapt_match.iloc[0]
        arch = str(row["name"])
        if not _is_supported_arch(arch):
            raise ValueError(
                f"AugReg arch {arch!r} is not supported (filename={filename!r})"
            )
        adapt_ds = str(row["adapt_ds"])
        if adapt_ds not in _ADAPT_NUM_CLASSES:
            raise ValueError(f"Unknown adapt_ds {adapt_ds!r} for {filename!r}")
        return _build_info(
            arch=arch,
            image_size=int(row["adapt_resolution"]),
            num_classes=_ADAPT_NUM_CLASSES[adapt_ds],
            filename=filename,
            is_adapt=True,
        )

    pretrain_match = df[df["filename"] == filename]
    if len(pretrain_match) > 0:
        row = pretrain_match.iloc[0]
        arch = str(row["name"])
        if not _is_supported_arch(arch):
            raise ValueError(
                f"AugReg arch {arch!r} is not supported (filename={filename!r})"
            )
        ds = str(row["ds"])
        if ds not in _PRETRAIN_NUM_CLASSES:
            raise ValueError(f"Unknown pretrain ds {ds!r} for {filename!r}")
        return _build_info(
            arch=arch,
            image_size=224,
            num_classes=_PRETRAIN_NUM_CLASSES[ds],
            filename=filename,
            is_adapt=False,
        )

    raise ValueError(f"Filename {filename!r} not found in AugReg index")


def _build_info(
    *,
    arch: str,
    image_size: int,
    num_classes: int,
    filename: str,
    is_adapt: bool,
) -> GoogleModelInfo:
    spec = _ARCH_TABLE[arch]
    return GoogleModelInfo(
        arch=arch,
        patch_size=spec.patch_size,
        image_size=image_size,
        hidden_size=spec.hidden_size,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        mlp_dim=spec.mlp_dim,
        num_classes=num_classes,
        checkpoint_url=f"{GOOGLE_AUGREG_BASE_URL}/{filename}.npz",
        is_adapt=is_adapt,
    )
