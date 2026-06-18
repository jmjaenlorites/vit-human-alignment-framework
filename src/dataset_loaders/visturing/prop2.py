"""Dataset loader para Experiment_2: Weber Law."""

import os
from glob import glob
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BaseVisTuringDatasetLoader, BaseVisTuringTorchDatasetLoader


class Prop2DatasetLoader(BaseVisTuringDatasetLoader):
    """Dataset loader para Experiment_2 (Weber law)."""

    EXPERIMENT_NAME = "Experiment_2"

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(name="Prop2_WeberLaw", data_path=data_path)

    def load_data(self) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga las imágenes y metadatos del Experiment_2.

        Returns:
            Tupla (data, bgs, x_a, x_rg, x_yb):
                - data: Dict con arrays de imágenes por canal
                - bgs: Dict con arrays de backgrounds por canal
                - x_a: Valores x para acromático
                - x_rg: Valores x para rojo-verde
                - x_yb: Valores x para amarillo-azul
        """
        experiment_path = self.ensure_data_downloaded()

        # Cargar valores x
        x_a = np.load(os.path.join(experiment_path, "luminancias.npy"))
        x_rg = np.load(os.path.join(experiment_path, "x_rg.npy"))
        x_yb = np.load(os.path.join(experiment_path, "x_yb.npy"))

        # Cargar datos de imágenes (excluyendo backgrounds y valores x)
        data = {
            p.split("/")[-1].split(".")[0]: np.load(p)
            for p in glob(os.path.join(experiment_path, "*.npy"))
            if "bgs" not in p
            and "luminancias" not in p
            and "x_rg" not in p
            and "x_yb" not in p
        }

        # Cargar backgrounds
        bgs = {
            p.split("/")[-1].split(".")[0][4:]: np.load(p)
            for p in glob(os.path.join(experiment_path, "*.npy"))
            if "bgs" in p
        }

        return data, bgs, x_a, x_rg, x_yb


class Prop2Dataset(Dataset):
    """PyTorch Dataset para Experiment_2."""

    def __init__(
        self,
        channel: str,
        test_images: np.ndarray,
        backgrounds: np.ndarray,
        transform: Optional[Callable[[Any], Any]] = None,
        samples: Optional[
            list[tuple[np.ndarray, np.ndarray, str, int, int, int]]
        ] = None,
    ):
        """
        Args:
            channel: Canal a procesar ('achrom', 'red_green', 'yellow_blue', o 'all')
            test_images: Array de arrays con imágenes de test
            backgrounds: Array con backgrounds correspondientes
            transform: Transformación opcional
            samples: Lista opcional de samples preconstruidos con metadata
        """
        self.channel = channel
        self.test_images = test_images
        self.backgrounds = backgrounds
        self.transform = transform

        if samples is not None:
            self.samples = samples
        else:
            # Aplanar estructura: cada par (img, bg) es un sample
            self.samples = []
            for level_idx, (imgs, bg) in enumerate(zip(test_images, backgrounds)):
                bg_idx = 0
                for sample_idx, img in enumerate(imgs):
                    self.samples.append(
                        (img, bg, channel, level_idx, sample_idx, bg_idx)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        test_img, bg_img, channel, level_idx, sample_idx, bg_idx = self.samples[idx]

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

        return test_img, bg_img, channel, level_idx, sample_idx, bg_idx


class Prop2TorchDatasetLoader(BaseVisTuringTorchDatasetLoader, Prop2DatasetLoader):
    """Torch DataLoader para Experiment_2."""

    def __init__(
        self,
        channel: str = "achrom",  # 'achrom', 'red_green', 'yellow_blue', 'all'
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 2,
        data_path: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        use_torch_upstream_semantics: bool = False,
    ):
        BaseVisTuringTorchDatasetLoader.__init__(
            self,
            name=f"Prop2Torch_{channel}",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            data_path=data_path,
            transform=transform,
        )
        Prop2DatasetLoader.__init__(self, data_path=data_path)
        self.channel = channel
        self.use_torch_upstream_semantics = use_torch_upstream_semantics

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        """Crea el PyTorch Dataset."""
        data, bgs, x_a, x_rg, x_yb = self.load_data()

        # Guardar valores x como atributo para la métrica
        self.x_values = {
            "achrom": x_a,
            "red_green": x_rg,
            "yellow_blue": x_yb,
        }
        self.level_counts = {
            "achrom": data["achrom"].shape[0],
            "red_green": data["red_green"].shape[0],
            "yellow_blue": data["yellow_blue"].shape[0],
        }

        if self.channel == "all":
            samples = []
            for ch in ["achrom", "red_green", "yellow_blue"]:
                test_images = data[ch]
                backgrounds = bgs[ch]
                for level_idx, (imgs, bg) in enumerate(zip(test_images, backgrounds)):
                    bg_idx = 0
                    ref_img = bg
                    if self.use_torch_upstream_semantics:
                        if ch == "achrom":
                            ref_img = imgs[0]
                        else:
                            bg_idx = int(
                                np.argwhere(
                                    np.where(imgs == bg, True, False).all(
                                        axis=(1, 2, 3)
                                    )
                                ).squeeze()
                            )
                    for sample_idx, img in enumerate(imgs):
                        samples.append(
                            (img, ref_img, ch, level_idx, sample_idx, bg_idx)
                        )
            return Prop2Dataset(
                channel=self.channel,
                test_images=np.array([]),
                backgrounds=np.array([]),
                transform=self.transform,
                samples=samples,
            )

        samples = None
        if self.use_torch_upstream_semantics:
            samples = []
            test_images = data[self.channel]
            backgrounds = bgs[self.channel]
            for level_idx, (imgs, bg) in enumerate(zip(test_images, backgrounds)):
                bg_idx = 0
                ref_img = bg
                if self.channel == "achrom":
                    ref_img = imgs[0]
                else:
                    bg_idx = int(
                        np.argwhere(
                            np.where(imgs == bg, True, False).all(axis=(1, 2, 3))
                        ).squeeze()
                    )
                for sample_idx, img in enumerate(imgs):
                    samples.append(
                        (img, ref_img, self.channel, level_idx, sample_idx, bg_idx)
                    )

        return Prop2Dataset(
            channel=self.channel,
            test_images=data[self.channel],
            backgrounds=bgs[self.channel],
            transform=self.transform,
            samples=samples,
        )
