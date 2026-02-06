"""Dataset loader para Experiment_5: Campbell-Blakemore (frequency masking)."""

import os
from glob import glob
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseVisTuringDatasetLoader, BaseVisTuringTorchDatasetLoader


class Prop5DatasetLoader(BaseVisTuringDatasetLoader):
    """Dataset loader para Experiment_5 (Campbell-Blakemore)."""

    EXPERIMENT_NAME = "Experiment_5"

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(name="Prop5_CampbellBlakemore", data_path=data_path)

    def load_data(self):
        """
        Carga las imágenes y metadatos del Experiment_5.

        Returns:
            Tupla (noises, backgrounds, freqs):
                - noises: Dict con arrays de noise patterns con máscaras
                - backgrounds: Dict con backgrounds por clave
                - freqs: Array de frecuencias espaciales
        """
        experiment_path = self.ensure_data_downloaded()

        # Cargar frecuencias (compatibilidad: freqs.npy/freq.npy)
        freqs_path = os.path.join(experiment_path, "freqs.npy")
        if not os.path.exists(freqs_path):
            freqs_path = os.path.join(experiment_path, "freq.npy")
        freqs = np.load(freqs_path)

        # Cargar noise patterns con máscaras
        # Formato: noises_achrom_Fmask_3.npy, noises_achrom_Fmask_6.npy, etc.
        noises = {
            p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p)
            for p in glob(os.path.join(experiment_path, "noises_*.npy"))
        }

        # Cargar backgrounds por clave (a, 3, 6, 12)
        backgrounds = {
            p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p)
            for p in glob(os.path.join(experiment_path, "background_*.npy"))
        }

        return noises, backgrounds, freqs


class Prop5Dataset(Dataset):
    """PyTorch Dataset para Experiment_5."""

    def __init__(
        self,
        noise_key: str,  # ej: 'achrom_Fmask_3' o 'all'
        noise_patterns: np.ndarray,
        background: np.ndarray,
        transform: Optional[Callable[[Any], Any]] = None,
        samples: Optional[list[tuple[np.ndarray, np.ndarray, str, int]]] = None,
    ):
        self.noise_key = noise_key
        self.noise_patterns = noise_patterns
        self.background = background
        self.transform = transform

        # Aplanar estructura
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            if noise_patterns.ndim == 4:
                # [freq, H, W, C]
                for freq_idx, noise in enumerate(noise_patterns):
                    self.samples.append((noise, background, noise_key, freq_idx))
            elif noise_patterns.ndim == 5:
                # [repeat, freq, H, W, C]
                for repeat_idx in range(noise_patterns.shape[0]):
                    for freq_idx in range(noise_patterns.shape[1]):
                        self.samples.append(
                            (
                                noise_patterns[repeat_idx, freq_idx],
                                background,
                                noise_key,
                                freq_idx,
                            )
                        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        test_img, bg_img, noise_key, freq_idx = self.samples[idx]

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

        return test_img, bg_img, noise_key, freq_idx


class Prop5TorchDatasetLoader(BaseVisTuringTorchDatasetLoader, Prop5DatasetLoader):
    """Torch DataLoader para Experiment_5."""

    def __init__(
        self,
        noise_key: str = "achrom_Fmask_3",  # Clave del noise pattern o 'all'
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 2,
        data_path: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        BaseVisTuringTorchDatasetLoader.__init__(
            self,
            name=f"Prop5Torch_{noise_key}",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            data_path=data_path,
            transform=transform,
        )
        Prop5DatasetLoader.__init__(self, data_path=data_path)
        self.noise_key = noise_key

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        """Crea el PyTorch Dataset."""
        noises, backgrounds, freqs = self.load_data()

        # Guardar frecuencias como atributo
        self.freqs = freqs

        if self.noise_key == "all":
            samples = []
            for key, noise_patterns in noises.items():
                background = backgrounds[key]
                if noise_patterns.ndim == 4:
                    for freq_idx, noise in enumerate(noise_patterns):
                        samples.append((noise, background, key, freq_idx))
                elif noise_patterns.ndim == 5:
                    for repeat_idx in range(noise_patterns.shape[0]):
                        for freq_idx in range(noise_patterns.shape[1]):
                            samples.append(
                                (
                                    noise_patterns[repeat_idx, freq_idx],
                                    background,
                                    key,
                                    freq_idx,
                                )
                            )
            return Prop5Dataset(
                noise_key=self.noise_key,
                noise_patterns=np.array([]),
                background=np.array([]),
                transform=self.transform,
                samples=samples,
            )

        return Prop5Dataset(
            noise_key=self.noise_key,
            noise_patterns=noises[self.noise_key],
            background=backgrounds[self.noise_key],
            transform=self.transform,
        )
