from typing import Protocol, Iterator, Any

import torch
from torch.utils.data import Dataset

class BaseDatasetLoader(Protocol):
    def __init__(self, name: str):
        self.name = name

    def get_iterator(self) -> Iterator[Any]:
        """Retorna un iterador sobre el dataset."""



class BaseTorchDatasetLoader(BaseDatasetLoader):
    def __init__(self, name: str, batch_size: int, shuffle: bool, num_workers: int):
        super().__init__(name)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_iterator(self) -> Iterator[Any]:
        """Retorna un iterador sobre el dataset."""
        return torch.utils.data.DataLoader(self.get_dataset(), batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def get_dataset(self) -> Dataset:
        """Retorna un dataset de PyTorch."""
        