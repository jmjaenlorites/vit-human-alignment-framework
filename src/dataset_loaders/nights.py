import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Optional, Callable, Any
from PIL import Image
import torchvision.transforms.functional as F
from .base import BaseDatasetLoader, BaseTorchDatasetLoader


class BaseNightsDatasetLoader(BaseDatasetLoader):
    pass


class NightsDatasetLoader(BaseNightsDatasetLoader):
    DATASET_PATH = "/media/disk/vista/BBDD_video_image/Image_Quality/nights/"

    def get_paths(self) -> dict[str, Any]:
        """Returns paths to images and the dataframe with voting information."""
        csv_path = os.path.join(self.DATASET_PATH, "data.csv")
        df = pd.read_csv(csv_path)

        ref_paths = [
            os.path.join(self.DATASET_PATH, row.ref_path) for _, row in df.iterrows()
        ]
        left_paths = [
            os.path.join(self.DATASET_PATH, row.left_path) for _, row in df.iterrows()
        ]
        right_paths = [
            os.path.join(self.DATASET_PATH, row.right_path) for _, row in df.iterrows()
        ]
        left_votes = df.left_vote.values.tolist()
        right_votes = df.right_vote.values.tolist()

        return {
            "reference": ref_paths,
            "left": left_paths,
            "right": right_paths,
            "left_vote": left_votes,
            "right_vote": right_votes,
            "dataframe": df,
        }


class NightsDataset(Dataset):
    def __init__(
        self, paths: dict[str, Any], transform: Optional[Callable[[Any], Any]] = None
    ):
        self.ref_paths = paths["reference"]
        self.left_paths = paths["left"]
        self.right_paths = paths["right"]
        self.left_votes = paths["left_vote"]
        self.right_votes = paths["right_vote"]
        self.transform = transform

    def __len__(self):
        return len(self.ref_paths)

    def __getitem__(self, idx):
        reference = Image.open(self.ref_paths[idx]).convert("RGB")
        left = Image.open(self.left_paths[idx]).convert("RGB")
        right = Image.open(self.right_paths[idx]).convert("RGB")

        reference = F.to_tensor(reference)
        left = F.to_tensor(left)
        right = F.to_tensor(right)

        if self.transform:
            reference = self.transform(reference)
            left = self.transform(left)
            right = self.transform(right)

        left_vote = self.left_votes[idx]
        right_vote = self.right_votes[idx]

        return reference, left, right, left_vote, right_vote


class NightsTorchDatasetLoader(BaseTorchDatasetLoader, NightsDatasetLoader):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(
            name="NightsTorch",
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        self.transform = transform

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        paths = self.get_paths()
        return NightsDataset(paths, transform=self.transform)
