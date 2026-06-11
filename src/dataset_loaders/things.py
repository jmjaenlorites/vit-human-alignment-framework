import os
from typing import Any, Callable, Optional

import pandas as pd
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .base import BaseDatasetLoader, BaseTorchDatasetLoader


def _strip_relative_prefix(path: str) -> str:
    """Quita el prefijo `./` que usan los paths del CSV de THINGS."""
    if path.startswith("./"):
        return path[2:]
    return path


class BaseThingsDatasetLoader(BaseDatasetLoader):
    pass


class ThingsDatasetLoader(BaseThingsDatasetLoader):
    DATASET_PATH = "/media/disk/vista/BBDD_video_image/Image_Quality/THINGS/"
    CSV_NAME = "Things_correct_triplet_dataset.csv"

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or os.environ.get(
            "THINGS_DATASET_PATH", self.DATASET_PATH
        )
        super().__init__(name="Things")

    def get_paths(self) -> dict[str, Any]:
        """Devuelve paths absolutos de las tres imágenes del triplet y el outlier."""
        csv_path = os.path.join(self.dataset_path, self.CSV_NAME)
        df = pd.read_csv(csv_path)

        image1_paths = [
            os.path.join(self.dataset_path, _strip_relative_prefix(row.Img1Path))
            for _, row in df.iterrows()
        ]
        image2_paths = [
            os.path.join(self.dataset_path, _strip_relative_prefix(row.Img2Path))
            for _, row in df.iterrows()
        ]
        image3_paths = [
            os.path.join(self.dataset_path, _strip_relative_prefix(row.Img3Path))
            for _, row in df.iterrows()
        ]
        # Para la métrica solo necesitamos identificar cuál es el outlier; nos
        # quedamos con el path relativo (idéntico al que aparece en Img{1,2,3}Path
        # tras strip), que es comparable contra los nombres relativos del batch.
        choice_relpaths = [_strip_relative_prefix(p) for p in df.choice_path.values]

        return {
            "image1": image1_paths,
            "image2": image2_paths,
            "image3": image3_paths,
            "choice": choice_relpaths,
            "dataframe": df,
        }


class ThingsDataset(Dataset):
    def __init__(
        self, paths: dict[str, Any], transform: Optional[Callable[[Any], Any]] = None
    ):
        self.image1_paths = paths["image1"]
        self.image2_paths = paths["image2"]
        self.image3_paths = paths["image3"]
        self.choices = paths["choice"]
        self.transform = transform

    def __len__(self):
        return len(self.image1_paths)

    def __getitem__(self, idx):
        # `convert("RGB")` cubre escala de grises, RGBA, paleta, etc., evitando
        # tensores con canales inesperados que romperían el ViT.
        img1 = Image.open(self.image1_paths[idx]).convert("RGB")
        img2 = Image.open(self.image2_paths[idx]).convert("RGB")
        img3 = Image.open(self.image3_paths[idx]).convert("RGB")

        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)
        img3 = F.to_tensor(img3)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Reconstruimos el path relativo (sin `./`) que la métrica comparará
        # contra `choice` para identificar el outlier.
        choice_relpath = self.choices[idx]
        img1_relpath = self._infer_relpath(self.image1_paths[idx])
        img2_relpath = self._infer_relpath(self.image2_paths[idx])
        img3_relpath = self._infer_relpath(self.image3_paths[idx])

        return img1, img2, img3, choice_relpath, img1_relpath, img2_relpath, img3_relpath

    def _infer_relpath(self, abs_path: str) -> str:
        """Devuelve el path relativo a partir del absoluto.

        Como `image{1,2,3}_paths` se construyen con `os.path.join(dataset_path,
        path[2:])`, el sufijo a partir de `THINGS/Images/...` reconstruye el
        relativo que aparece en `choice_path`.
        """
        marker = "THINGS/Images/"
        idx = abs_path.find(marker)
        if idx >= 0:
            return abs_path[idx:]
        return os.path.basename(abs_path)


class ThingsTorchDatasetLoader(BaseTorchDatasetLoader, ThingsDatasetLoader):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        transform: Optional[Callable[[Any], Any]] = None,
        dataset_path: Optional[str] = None,
    ):
        BaseTorchDatasetLoader.__init__(
            self,
            name="ThingsTorch",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        ThingsDatasetLoader.__init__(self, dataset_path=dataset_path)
        self.transform = transform

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        paths = self.get_paths()
        return ThingsDataset(paths, transform=self.transform)
