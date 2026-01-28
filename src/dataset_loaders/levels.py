import os
from typing import Any, Callable, Literal, Optional

import pandas as pd
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from .base import BaseDatasetLoader, BaseTorchDatasetLoader


class BaseLevelsDatasetLoader(BaseDatasetLoader):
    pass


class LevelsDatasetLoader(BaseLevelsDatasetLoader):
    IMAGENET_PATH = "/media/disk/vista/BBDD_video_image/imagenet_complete/ILSVRC/Data/CLS-LOC/train/"
    LEVELS_PATH = "/media/disk/vista/BBDD_video_image/Image_Quality/Levels/"

    def __init__(
        self,
        split: Literal[
            "between_class", "class_border", "within_class"
        ] = "between_class",
        levels_path: Optional[str] = None,
        imagenet_path: Optional[str] = None,
    ):
        self.split = split
        self.levels_path = levels_path or os.environ.get(
            "LEVELS_DATASET_PATH", self.LEVELS_PATH
        )
        self.imagenet_path = imagenet_path or os.environ.get(
            "IMAGENET_PATH", self.IMAGENET_PATH
        )
        super().__init__(name=f"Levels_{split}")

    def get_paths(self) -> dict[str, Any]:
        """Returns paths to images and the dataframe with triplet information."""
        csv_path = os.path.join(self.levels_path, f"{self.split}.csv")
        df = pd.read_csv(csv_path)

        image1_paths = [
            os.path.join(
                self.imagenet_path, row.image1Path.split("_")[0], row.image1Path
            )
            for _, row in df.iterrows()
        ]
        image2_paths = [
            os.path.join(
                self.imagenet_path, row.image2Path.split("_")[0], row.image2Path
            )
            for _, row in df.iterrows()
        ]
        image3_paths = [
            os.path.join(
                self.imagenet_path, row.image3Path.split("_")[0], row.image3Path
            )
            for _, row in df.iterrows()
        ]
        selected_images = df.selected_image.values.tolist()

        return {
            "image1": image1_paths,
            "image2": image2_paths,
            "image3": image3_paths,
            "selected": selected_images,
            "dataframe": df,
        }


class LevelsDataset(Dataset):
    def __init__(
        self, paths: dict[str, Any], transform: Optional[Callable[[Any], Any]] = None
    ):
        self.image1_paths = paths["image1"]
        self.image2_paths = paths["image2"]
        self.image3_paths = paths["image3"]
        self.selected_images = paths["selected"]
        self.transform = transform

    def __len__(self):
        return len(self.image1_paths)

    def __getitem__(self, idx):
        from PIL import Image

        # Load images using PIL to handle grayscale conversion
        img1 = Image.open(self.image1_paths[idx])
        img2 = Image.open(self.image2_paths[idx])
        img3 = Image.open(self.image3_paths[idx])

        # Convert grayscale to RGB if needed
        if img1.mode == "L":
            img1 = Image.merge("RGB", (img1, img1, img1))
        if img2.mode == "L":
            img2 = Image.merge("RGB", (img2, img2, img2))
        if img3.mode == "L":
            img3 = Image.merge("RGB", (img3, img3, img3))

        # Convert to tensor
        img1 = F.to_tensor(img1)
        img2 = F.to_tensor(img2)
        img3 = F.to_tensor(img3)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        # Get the selected image filename (not full path)
        selected = os.path.basename(self.selected_images[idx])
        img1_name = os.path.basename(self.image1_paths[idx])
        img2_name = os.path.basename(self.image2_paths[idx])
        img3_name = os.path.basename(self.image3_paths[idx])

        return img1, img2, img3, selected, img1_name, img2_name, img3_name


class LevelsTorchDatasetLoader(BaseTorchDatasetLoader, LevelsDatasetLoader):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        split: Literal[
            "between_class", "class_border", "within_class"
        ] = "class_border",
        transform: Optional[Callable[[Any], Any]] = None,
        levels_path: Optional[str] = None,
        imagenet_path: Optional[str] = None,
    ):
        BaseTorchDatasetLoader.__init__(
            self,
            name=f"LevelsTorch_{split}",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        LevelsDatasetLoader.__init__(
            self, split=split, levels_path=levels_path, imagenet_path=imagenet_path
        )
        self.transform = transform

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        paths = self.get_paths()
        return LevelsDataset(paths, transform=self.transform)
