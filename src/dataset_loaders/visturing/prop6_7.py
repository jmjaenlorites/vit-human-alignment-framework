"""Dataset loader para Experiment_6_7: Contrast curves without mask."""

import os
import re
from glob import glob
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseVisTuringDatasetLoader, BaseVisTuringTorchDatasetLoader


class Prop6_7DatasetLoader(BaseVisTuringDatasetLoader):
    """Dataset loader para Experiment_6_7 (contrast curves without mask)."""

    EXPERIMENT_NAME = "Experiment_6_7"

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(name="Prop6_7_ContrastCurves", data_path=data_path)

    def load_data(
        self,
    ) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga las imágenes y metadatos del Experiment_6_7.

        Returns:
            Tupla (gabors, bgs, c_a, c_rg, c_yb):
                - gabors: Dict con arrays de gabor patterns
                - bgs: Dict con backgrounds
                - c_a: Contrastes acromáticos
                - c_rg: Contrastes rojo-verde
                - c_yb: Contrastes amarillo-azul
        """
        experiment_path = self.ensure_data_downloaded()

        # Cargar contrastes
        c_a = np.load(os.path.join(experiment_path, "contrast_a.npy"))
        c_rg = np.load(os.path.join(experiment_path, "contrast_rg.npy"))
        c_yb = np.load(os.path.join(experiment_path, "contrast_yb.npy"))

        # Cargar gabor patterns
        gabors = {
            re.findall(r"noise_(\w+)\.", p)[0]: np.load(p)
            for p in glob(os.path.join(experiment_path, "*"))
            if "gabor" in p
        }

        # Cargar backgrounds
        bgs = {
            re.findall(r"background_(\w+)\.", p)[0]: np.load(p)
            for p in glob(os.path.join(experiment_path, "*"))
            if "background" in p
        }

        return gabors, bgs, c_a, c_rg, c_yb


class Prop6_7Dataset(Dataset):
    """PyTorch Dataset para Experiment_6_7."""

    def __init__(
        self,
        gabor_key: str,  # ej: 'a_1p5', 'rg_3', etc. o 'all'
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
            if gabor_patterns.ndim == 4:
                # [contrast, H, W, C]
                ref = gabor_patterns[0]
                channel, freq = gabor_key.split("_", maxsplit=1)
                for contrast_idx, gabor in enumerate(gabor_patterns):
                    self.samples.append((gabor, ref, channel, freq, contrast_idx))
            elif gabor_patterns.ndim == 5:
                # [contrast, sample, H, W, C]
                channel, freq = gabor_key.split("_", maxsplit=1)
                for contrast_idx in range(gabor_patterns.shape[0]):
                    for sample_idx in range(gabor_patterns.shape[1]):
                        self.samples.append(
                            (
                                gabor_patterns[contrast_idx, sample_idx],
                                gabor_patterns[0, sample_idx],
                                channel,
                                freq,
                                contrast_idx,
                            )
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        test_img, bg_img, channel, freq, contrast_idx = self.samples[idx]

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

        return test_img, bg_img, channel, freq, contrast_idx


class Prop6_7TorchDatasetLoader(BaseVisTuringTorchDatasetLoader, Prop6_7DatasetLoader):
    """Torch DataLoader para Experiment_6_7."""

    def __init__(
        self,
        gabor_key: str = "a_1p5",  # ej: 'a_1p5', 'rg_3', 'yb_12', etc. o 'all'
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 2,
        data_path: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        BaseVisTuringTorchDatasetLoader.__init__(
            self,
            name=f"Prop6_7Torch_{gabor_key}",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            data_path=data_path,
            transform=transform,
        )
        Prop6_7DatasetLoader.__init__(self, data_path=data_path)
        self.gabor_key = gabor_key

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        """Crea el PyTorch Dataset."""
        gabors, bgs, c_a, c_rg, c_yb = self.load_data()

        # Guardar contrastes como atributo
        self.contrasts = {"a": c_a, "rg": c_rg, "yb": c_yb}

        freq_labels = ["1p5", "3", "6", "12", "24"]
        key_to_channel = {
            "achrom": "a",
            "red_green": "rg",
            "yellow_blue": "yb",
            "a": "a",
            "rg": "rg",
            "yb": "yb",
        }
        channel_to_key = {
            "a": "achrom" if "achrom" in gabors else "a",
            "rg": "red_green" if "red_green" in gabors else "rg",
            "yb": "yellow_blue" if "yellow_blue" in gabors else "yb",
        }

        if self.gabor_key == "all":
            samples = []
            for key, gabor_patterns in gabors.items():
                channel = key_to_channel.get(key, key)
                if gabor_patterns.ndim != 5:
                    continue
                for freq_idx, freq in enumerate(freq_labels):
                    for contrast_idx in range(gabor_patterns.shape[1]):
                        samples.append(
                            (
                                gabor_patterns[freq_idx, contrast_idx],
                                gabor_patterns[freq_idx, 0],
                                channel,
                                freq,
                                contrast_idx,
                            )
                        )
            return Prop6_7Dataset(
                gabor_key=self.gabor_key,
                gabor_patterns=np.array([]),
                background=np.array([]),
                transform=self.transform,
                samples=samples,
            )

        channel_req, freq_req = self.gabor_key.split("_", maxsplit=1)
        data_key = channel_to_key[channel_req]
        gabor_patterns = gabors[data_key]
        freq_idx = freq_labels.index(freq_req)
        selected = gabor_patterns[freq_idx]

        return Prop6_7Dataset(
            gabor_key=self.gabor_key,
            gabor_patterns=selected,
            background=np.array([]),
            transform=self.transform,
        )
