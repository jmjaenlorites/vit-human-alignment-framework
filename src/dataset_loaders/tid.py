import os
from typing import Any, Callable, Optional

import pandas as pd
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .base import BaseDatasetLoader, BaseTorchDatasetLoader


class BaseTIDDatasetLoader(BaseDatasetLoader):
    pass


class TID2013DatasetLoader(BaseTIDDatasetLoader):
    DATASET_PATH = "/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2013"

    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset_path = dataset_path or os.environ.get(
            "TID_DATASET_PATH", self.DATASET_PATH
        )
        super().__init__(name="TID2013")

    def get_paths(self) -> dict[str, Any]:
        """Returns paths to images and the dataframe with MOS scores."""
        csv_path = os.path.join(self.dataset_path, "image_pairs_mos.csv")
        df = pd.read_csv(csv_path, index_col=0)

        reference_paths = [
            os.path.join(self.dataset_path, "reference_images", row.Reference)
            for _, row in df.iterrows()
        ]
        distorted_paths = [
            os.path.join(self.dataset_path, "distorted_images", row.Distorted)
            for _, row in df.iterrows()
        ]
        mos_scores = df.MOS.values.tolist()

        return {
            "reference": reference_paths,
            "distorted": distorted_paths,
            "mos": mos_scores,
            "dataframe": df,
        }


class TID2013Dataset(Dataset):
    def __init__(
        self, paths: dict[str, Any], transform: Optional[Callable[[Any], Any]] = None
    ):
        self.reference_paths = paths["reference"]
        self.distorted_paths = paths["distorted"]
        self.mos_scores = paths["mos"]
        self.transform = transform

    def __len__(self):
        return len(self.reference_paths)

    def __getitem__(self, idx):
        reference = Image.open(self.reference_paths[idx]).convert("RGB")
        distorted = Image.open(self.distorted_paths[idx]).convert("RGB")
        mos = self.mos_scores[idx]

        reference = F.to_tensor(reference)
        distorted = F.to_tensor(distorted)

        if self.transform:
            reference = self.transform(reference)
            distorted = self.transform(distorted)

        return reference, distorted, mos


class TID2013TorchDatasetLoader(BaseTorchDatasetLoader, TID2013DatasetLoader):
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
            name="TID2013Torch",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        TID2013DatasetLoader.__init__(self, dataset_path=dataset_path)
        self.transform = transform

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        paths = self.get_paths()
        return TID2013Dataset(paths, transform=self.transform)
