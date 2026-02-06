"""Dataset loader para Experiment_3_4: Contrast Sensitivity Function (CSF)."""

import os
from glob import glob
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseVisTuringDatasetLoader, BaseVisTuringTorchDatasetLoader


class Prop3_4DatasetLoader(BaseVisTuringDatasetLoader):
    """Dataset loader para Experiment_3_4 (CSF)."""

    EXPERIMENT_NAME = "Experiment_3_4"

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(name="Prop3_4_CSF", data_path=data_path)

    def load_data(self) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        Carga las imágenes y metadatos del Experiment_3_4.

        Returns:
            Tupla (noises, background, freqs):
                - noises: Dict con arrays de noise patterns por canal
                - background: Imagen de background [H, W, C]
                - freqs: Array de frecuencias espaciales
        """
        experiment_path = self.ensure_data_downloaded()

        # Cargar background
        background = np.load(os.path.join(experiment_path, "background.npy"))

        # Cargar frecuencias
        freqs = np.load(os.path.join(experiment_path, "freq.npy"))

        # Cargar noise patterns por canal
        noises = {
            p.split("/")[-1].split(".")[0].split("_")[-1]: np.load(p)
            for p in glob(os.path.join(experiment_path, "*"))
            if "noises" in p
        }

        return noises, background, freqs


class Prop3_4Dataset(Dataset):
    """PyTorch Dataset para Experiment_3_4."""

    def __init__(
        self,
        channel: str,  # 'achrom', 'rg', 'yb', 'all'
        noise_patterns: np.ndarray,  # [N_freqs, N_samples, H, W, C]
        background: np.ndarray,  # [H, W, C]
        transform: Optional[Callable[[Any], Any]] = None,
        samples: Optional[list[tuple[np.ndarray, np.ndarray, str, int]]] = None,
    ):
        """
        Args:
            channel: Canal a procesar
            noise_patterns: Arrays de noise patterns
            background: Background común
            transform: Transformación opcional
        """
        self.channel = channel
        self.noise_patterns = noise_patterns
        self.background = background
        self.transform = transform

        # Aplanar estructura: cada pattern es un sample
        if samples is not None:
            self.samples = samples
        else:
            self.samples = []
            for sample_noises in noise_patterns:
                for freq_idx, noise in enumerate(sample_noises):
                    self.samples.append((noise, channel, freq_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        test_img, channel, freq_idx = self.samples[idx]
        bg_img = self.background

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

        return test_img, bg_img, channel, freq_idx


class Prop3_4TorchDatasetLoader(BaseVisTuringTorchDatasetLoader, Prop3_4DatasetLoader):
    """Torch DataLoader para Experiment_3_4."""

    def __init__(
        self,
        channel: str = "achrom",  # 'achrom', 'rg', 'yb', 'all'
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 2,
        data_path: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        BaseVisTuringTorchDatasetLoader.__init__(
            self,
            name=f"Prop3_4Torch_{channel}",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            data_path=data_path,
            transform=transform,
        )
        Prop3_4DatasetLoader.__init__(self, data_path=data_path)
        self.channel = channel

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        """Crea el PyTorch Dataset."""
        noises, background, freqs = self.load_data()

        # Guardar frecuencias como atributo para la métrica
        self.freqs = freqs

        def resolve_channel_key(channel: str) -> str:
            return "a" if channel == "achrom" else channel

        if self.channel == "all":
            samples = []
            for ch in ["achrom", "rg", "yb"]:
                noise_patterns = noises[resolve_channel_key(ch)]
                for sample_noises in noise_patterns:
                    for freq_idx, noise in enumerate(sample_noises):
                        samples.append((noise, ch, freq_idx))
            return Prop3_4Dataset(
                channel=self.channel,
                noise_patterns=np.array([]),
                background=background,
                transform=self.transform,
                samples=samples,
            )

        return Prop3_4Dataset(
            channel=self.channel,
            noise_patterns=noises[resolve_channel_key(self.channel)],
            background=background,
            transform=self.transform,
        )
