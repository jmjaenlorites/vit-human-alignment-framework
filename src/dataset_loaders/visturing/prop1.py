"""Dataset loader para Experiment_1: Spectral Sensitivities."""

import os
from glob import glob
from typing import Any, Callable, Optional

import cv2
import torch
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset

from .base import (
    BaseVisTuringDatasetLoader,
    BaseVisTuringTorchDatasetLoader,
    load_image_rgb,
)


class Prop1DatasetLoader(BaseVisTuringDatasetLoader):
    """Dataset loader para Experiment_1 (spectral sensitivities)."""

    EXPERIMENT_NAME = "Experiment_1"

    def __init__(self, data_path: Optional[str] = None):
        super().__init__(name="Prop1_SpectralSensitivity", data_path=data_path)

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga las imágenes y metadatos del Experiment_1.

        Returns:
            Tupla (imgs, ref_img, lambdas):
                - imgs: Array de imágenes [N, H, W, C]
                - ref_img: Imagen de referencia [H, W, C]
                - lambdas: Array de longitudes de onda [N]
        """
        experiment_path = self.ensure_data_downloaded()

        # Cargar imagen de referencia
        ref_path = os.path.join(experiment_path, "im_ref.png")
        ref_img = load_image_rgb(ref_path)

        # Cargar longitudes de onda
        lambdas_path = os.path.join(experiment_path, "lambdas.npy")
        lambdas = np.load(lambdas_path)

        # Cargar imágenes de test (todas excepto la de referencia)
        imgs_paths = [
            p for p in glob(os.path.join(experiment_path, "*.png")) if "ref" not in p
        ]
        imgs_paths = list(natsorted(imgs_paths))

        imgs = np.array([load_image_rgb(p) for p in imgs_paths])

        # Interpolar lambdas para tener el mismo número que imágenes
        lambdas = np.linspace(lambdas.min(), lambdas.max(), num=len(imgs))

        return imgs, ref_img, lambdas


class Prop1Dataset(Dataset):
    """PyTorch Dataset para Experiment_1."""

    def __init__(
        self,
        test_images: np.ndarray,
        ref_image: np.ndarray,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        self.test_images = test_images
        self.ref_image = ref_image
        self.transform = transform

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        test_img = self.test_images[idx]
        ref_img = self.ref_image

        # Convertir a float32 en [0, 1]
        if test_img.dtype == np.uint8:
            test_img = test_img.astype(np.float32) / 255.0
        if ref_img.dtype == np.uint8:
            ref_img = ref_img.astype(np.float32) / 255.0
        if test_img.dtype != np.float32:
            test_img = test_img.astype(np.float32)
        if ref_img.dtype != np.float32:
            ref_img = ref_img.astype(np.float32)

        # De (H, W, C) a (C, H, W)
        test_img = np.transpose(test_img, (2, 0, 1))
        ref_img = np.transpose(ref_img, (2, 0, 1))

        if self.transform:
            if isinstance(test_img, np.ndarray):
                test_img = torch.from_numpy(test_img)
            if isinstance(ref_img, np.ndarray):
                ref_img = torch.from_numpy(ref_img)
            test_img = self.transform(test_img)
            ref_img = self.transform(ref_img)

        return test_img, ref_img


class Prop1TorchDatasetLoader(BaseVisTuringTorchDatasetLoader, Prop1DatasetLoader):
    """Torch DataLoader para Experiment_1."""

    def __init__(
        self,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 2,
        data_path: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        BaseVisTuringTorchDatasetLoader.__init__(
            self,
            name="Prop1Torch",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            data_path=data_path,
            transform=transform,
        )
        Prop1DatasetLoader.__init__(self, data_path=data_path)

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        """Crea el PyTorch Dataset."""
        imgs, ref_img, lambdas = self.load_data()

        # Guardar lambdas como atributo para la métrica
        self.lambdas = lambdas

        return Prop1Dataset(imgs, ref_img, transform=self.transform)
