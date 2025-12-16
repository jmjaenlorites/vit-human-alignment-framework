import os

from torch.utils.data import Dataset
from typing import Optional, Callable, Any
from .base import BaseDatasetLoader, BaseTorchDatasetLoader
import torchvision


class BaseSaliencyDatasetLoader(BaseDatasetLoader):
    pass

class SaliencyMIT1003DatasetLoader(BaseSaliencyDatasetLoader):
    DATASET_PATH = MIT1003_DATASET_PATH = "/Users/jmjaenlorites/PhD/ViT-Alignment/ViT-saliency-alignment/FixaTons_repo/Datasets/MIT1003/"
    def get_paths(self) -> dict[str, list[str]]:
        return {
            "stimulus": [os.path.join(self.DATASET_PATH, 'STIMULI', f) for f in os.listdir(os.path.join(self.DATASET_PATH, 'STIMULI')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            "saliency": [os.path.join(self.DATASET_PATH, 'SALIENCY_MAPS', f) for f in os.listdir(os.path.join(self.DATASET_PATH, 'SALIENCY_MAPS')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            "fixation": [os.path.join(self.DATASET_PATH, 'FIXATION_MAPS', f) for f in os.listdir(os.path.join(self.DATASET_PATH, 'FIXATION_MAPS')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        }

class SaliencyMIT1003Dataset(Dataset):
            def __init__(self, paths: dict[str, list[str]], transform: Optional[Callable[[Any], Any]] = None):
                self.stimulus_paths = paths['stimulus']
                self.saliency_paths = paths['saliency']
                self.fixation_paths = paths['fixation']
                self.transform = transform

            def __len__(self):
                return len(self.stimulus_paths)

            def __getitem__(self, idx):
                stimulus = torchvision.io.decode_image(self.stimulus_paths[idx]) / 255.0
                if self.transform:
                    stimulus = self.transform(stimulus)
                saliency = torchvision.io.decode_image(self.saliency_paths[idx]) / 255.0
                fixation = torchvision.io.decode_image(self.fixation_paths[idx]) / 255.0
                if self.transform:
                    resize_function = torchvision.transforms.Compose([x for x in self.transform.transforms if not isinstance(x, torchvision.transforms.Normalize)])
                    saliency = resize_function(saliency)
                    fixation = resize_function(fixation)
                return stimulus, saliency, fixation
                
class SaliencyMIT1003TorchDatasetLoader(BaseTorchDatasetLoader, SaliencyMIT1003DatasetLoader):
    def __init__(self, batch_size: int, shuffle: bool, num_workers: int, transform: Optional[Callable[[Any], Any]] = None):
        super().__init__(name="SaliencyMIT1003Torch", batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.transform = transform

    def get_dataset(self, transform: Optional[Callable[[Any], Any]] = None) -> Dataset:
        paths = self.get_paths()
        return SaliencyMIT1003Dataset(paths, transform=self.transform)