"""Base classes para dataset loaders de visturing."""

import os
from typing import Any, Callable, Optional

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from ..base import BaseDatasetLoader, BaseTorchDatasetLoader
from ...utils.download import download_and_extract


class BaseVisTuringDatasetLoader(BaseDatasetLoader):
    """
    Clase base para todos los dataset loaders de visturing.

    Proporciona funcionalidad común de descarga desde Zenodo.
    """

    # Subclases deben definir esto
    EXPERIMENT_NAME: str = ""  # ej: "Experiment_1"
    DATA_PATH_ENV_VAR: str = "VISTURING_DATA_PATH"
    DEFAULT_DATA_PATH: str = "./data/visturing"

    def __init__(
        self,
        name: str,
        data_path: Optional[str] = None,
    ):
        super().__init__(name)
        self.data_path = data_path or os.environ.get(
            self.DATA_PATH_ENV_VAR, self.DEFAULT_DATA_PATH
        )

    def ensure_data_downloaded(self) -> str:
        """
        Asegura que los datos del experimento estén descargados.

        Returns:
            Ruta al directorio del experimento
        """
        experiment_path = os.path.join(self.data_path, self.EXPERIMENT_NAME)

        if not os.path.exists(experiment_path):
            print(
                f"Dataset {self.EXPERIMENT_NAME} not found. Downloading from Zenodo..."
            )
            download_and_extract(
                f"{self.EXPERIMENT_NAME}.zip", self.data_path, self.EXPERIMENT_NAME
            )

        return experiment_path


class VisTuringDataset(Dataset):
    """
    Dataset genérico para experimentos de visturing que compara pares de imágenes.

    Carga imágenes de referencia y test para calcular diferencias.
    """

    def __init__(
        self,
        test_images: list[np.ndarray] | np.ndarray,
        ref_image: np.ndarray,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Args:
            test_images: Lista o array de imágenes de test (N, H, W, C)
            ref_image: Imagen de referencia (H, W, C)
            transform: Transformación opcional a aplicar
        """
        self.test_images = test_images
        self.ref_image = ref_image
        self.transform = transform

    def __len__(self):
        return len(self.test_images)

    def __getitem__(self, idx):
        test_img = self.test_images[idx]
        ref_img = self.ref_image

        # Convertir a tensor de PyTorch (C, H, W) con valores en [0, 1]
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


class BaseVisTuringTorchDatasetLoader(
    BaseTorchDatasetLoader, BaseVisTuringDatasetLoader
):
    """
    Base class para dataset loaders de visturing que usan PyTorch DataLoader.
    """

    def __init__(
        self,
        name: str,
        batch_size: int = 32,
        shuffle: bool = False,
        num_workers: int = 2,
        data_path: Optional[str] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        BaseTorchDatasetLoader.__init__(
            self,
            name=name,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        BaseVisTuringDatasetLoader.__init__(self, name=name, data_path=data_path)
        self.transform = transform


def load_image_rgb(path: str) -> np.ndarray:
    """
    Carga una imagen y la convierte a RGB.

    Args:
        path: Ruta a la imagen

    Returns:
        Array numpy (H, W, C) en formato RGB
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img
