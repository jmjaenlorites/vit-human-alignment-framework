"""Dataset loader para Experiment_8: Contrast masking."""

import os
import re
from glob import glob
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseVisTuringDatasetLoader, BaseVisTuringTorchDatasetLoader


class Prop8DatasetLoader(BaseVisTuringDatasetLoader):
    """Dataset loader para Experiment_8 (contrast masking)."""

    EXPERIMENT_NAME = "Experiment_8"

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(name="Prop8_ContrastMasking", data_path=data_path)

    def load_data(self) -> tuple[dict, dict, np.ndarray]:
        """
        Carga las imágenes y metadatos del Experiment_8.

        Returns:
            Tupla (gabors, bgs, contrasts):
                - gabors: Dict con arrays de gabor patterns con máscaras
                - bgs: Dict con backgrounds
                - contrasts: Array de contrastes
        """
        experiment_path = self.ensure_data_downloaded()

        # Cargar contrastes (compatibilidad: contrasts.npy/contrast.npy)
        contrasts_path = os.path.join(experiment_path, "contrasts.npy")
        if not os.path.exists(contrasts_path):
            contrasts_path = os.path.join(experiment_path, "contrast.npy")
        contrasts = np.load(contrasts_path)

        # Cargar gabor patterns con máscaras de contraste
        # Formato: gabor_a_C1_mask_0075.npy, gabor_a_C1_mask_0150.npy, etc.
        gabors = {}
        for p in glob(os.path.join(experiment_path, "*.npy")):
            name = p.split("/")[-1]
            m = re.findall(r"gabors_(low|high)_(\w+)\.npy", name)
            if m:
                freq, channel = m[0]
                gabors[f"{freq}_{channel}"] = np.load(p)
                continue
            if name.startswith("gabor_"):
                gabors["_".join(name.split(".")[0].split("_")[1:])] = np.load(p)

        # Compat: experimento nuevo no incluye backgrounds
        bgs = {}

        return gabors, bgs, contrasts


class Prop8Dataset(Dataset):
    """PyTorch Dataset para Experiment_8."""

    def __init__(
        self,
        gabor_key: str,  # ej: 'a_C1_mask_0075' o 'all'
        gabor_patterns: np.ndarray,
        background: np.ndarray,
        transform: Optional[Callable[[Any], Any]] = None,
        samples: Optional[list[tuple[np.ndarray, np.ndarray, str, str, int]]] = None,
    ):
        self.gabor_key = gabor_key
        self.gabor_patterns = gabor_patterns
        self.background = background
        self.transform = transform

        # Aplanar estructura
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            _, c_group, _, mask = gabor_key.split("_", maxsplit=3)
            if gabor_patterns.ndim == 4:
                ref = gabor_patterns[0]
                for contrast_idx, gabor in enumerate(gabor_patterns):
                    self.samples.append((gabor, ref, c_group, mask, contrast_idx))
            elif gabor_patterns.ndim == 5:
                for mask_idx in range(gabor_patterns.shape[0]):
                    for contrast_idx in range(gabor_patterns.shape[1]):
                        self.samples.append(
                            (
                                gabor_patterns[mask_idx, contrast_idx],
                                gabor_patterns[mask_idx, 0],
                                c_group,
                                mask,
                                contrast_idx,
                            )
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        test_img, bg_img, c_group, mask, contrast_idx = self.samples[idx]

        # Convertir a float32 en [0, 1]
        if test_img.dtype == np.uint8:
            test_img = test_img.astype(np.float32) / 255.0
        if bg_img.dtype == np.uint8:
            bg_img = bg_img.astype(np.float32) / 255.0
        if test_img.dtype != np.float32:
            test_img = test_img.astype(np.float32)
        if bg_img.dtype != np.float32:
            bg_img = bg_img.astype(np.float32)

        # De (H, W, C) a (C, H, W)
        test_img = np.transpose(test_img, (2, 0, 1))
        bg_img = np.transpose(bg_img, (2, 0, 1))

        if self.transform:
            if isinstance(test_img, np.ndarray):
                test_img = torch.from_numpy(test_img)
            if isinstance(bg_img, np.ndarray):
                bg_img = torch.from_numpy(bg_img)
            test_img = self.transform(test_img)
            bg_img = self.transform(bg_img)

        return test_img, bg_img, c_group, mask, contrast_idx


class Prop8TorchDatasetLoader(BaseVisTuringTorchDatasetLoader, Prop8DatasetLoader):
    """Torch DataLoader para Experiment_8."""

    def __init__(
        self,
        gabor_key: str = "a_C1_mask_0075",  # Clave del gabor pattern o 'all'
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 2,
        data_path: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        BaseVisTuringTorchDatasetLoader.__init__(
            self,
            name=f"Prop8Torch_{gabor_key}",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            data_path=data_path,
            transform=transform,
        )
        Prop8DatasetLoader.__init__(self, data_path=data_path)
        self.gabor_key = gabor_key

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        """Crea el PyTorch Dataset."""
        gabors, bgs, contrasts = self.load_data()

        # Guardar contrastes como atributo
        self.contrasts = contrasts

        mask_labels = ["nomask", "0075", "0150", "0225", "0300"]
        c_map = {"low": "C1", "high": "C2"}

        if self.gabor_key == "all":
            samples = []
            for freq in ["low", "high"]:
                key = f"{freq}_achrom"
                if key not in gabors:
                    continue
                gabor_patterns = gabors[key]
                c_group = c_map[freq]
                for mask_idx in range(gabor_patterns.shape[0]):
                    mask = mask_labels[mask_idx]
                    for contrast_idx in range(gabor_patterns.shape[1]):
                        samples.append(
                            (
                                gabor_patterns[mask_idx, contrast_idx],
                                gabor_patterns[mask_idx, 0],
                                c_group,
                                mask,
                                contrast_idx,
                            )
                        )
            return Prop8Dataset(
                gabor_key=self.gabor_key,
                gabor_patterns=np.array([]),
                background=np.array([]),
                transform=self.transform,
                samples=samples,
            )

        return Prop8Dataset(
            gabor_key=self.gabor_key,
            gabor_patterns=gabors[self.gabor_key],
            background=np.array([]),
            transform=self.transform,
        )
