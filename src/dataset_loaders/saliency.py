import os

from torch.utils.data import Dataset
from typing import Optional, Callable, Any
from .base import BaseDatasetLoader, BaseTorchDatasetLoader
import torchvision


class BaseSaliencyDatasetLoader(BaseDatasetLoader):
    pass

class SaliencyMIT1003DatasetLoader(BaseSaliencyDatasetLoader):
    DATASET_PATH = MIT1003_DATASET_PATH = "../ViT-saliency-alignment/FixaTons_repo/Datasets/MIT1003/"
    
    def get_paths(self) -> dict[str, list[str]]:
        """
        Obtiene los paths emparejados de stimulus, saliency y fixation maps.
        
        IMPORTANTE: Los archivos deben estar ordenados y emparejados por nombre base,
        ya que los nombres pueden tener extensiones diferentes entre carpetas.
        """
        stimuli_dir = os.path.join(self.DATASET_PATH, 'STIMULI')
        saliency_dir = os.path.join(self.DATASET_PATH, 'SALIENCY_MAPS')
        fixation_dir = os.path.join(self.DATASET_PATH, 'FIXATION_MAPS')
        
        # Obtener todos los stimulus files y ordenarlos
        stimulus_files = sorted([
            f for f in os.listdir(stimuli_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Crear mapeo de nombres base a archivos en saliency y fixation
        saliency_files_map = {}
        for f in os.listdir(saliency_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(f)[0]
                saliency_files_map[base_name] = f
        
        fixation_files_map = {}
        for f in os.listdir(fixation_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                base_name = os.path.splitext(f)[0]
                fixation_files_map[base_name] = f
        
        # Emparejar por nombre base
        stimulus_paths = []
        saliency_paths = []
        fixation_paths = []
        
        for stim_file in stimulus_files:
            base_name = os.path.splitext(stim_file)[0]
            
            if base_name not in saliency_files_map or base_name not in fixation_files_map:
                print(f"[WARNING] No matching saliency/fixation for {stim_file}, skipping...")
                continue
            
            stimulus_paths.append(os.path.join(stimuli_dir, stim_file))
            saliency_paths.append(os.path.join(saliency_dir, saliency_files_map[base_name]))
            fixation_paths.append(os.path.join(fixation_dir, fixation_files_map[base_name]))
        
        return {
            "stimulus": stimulus_paths,
            "saliency": saliency_paths,
            "fixation": fixation_paths
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
                
                # Convertir saliency y fixation a 1 canal (son grayscale guardados como RGB)
                if saliency.shape[0] == 3:
                    saliency = saliency.mean(dim=0, keepdim=True)
                if fixation.shape[0] == 3:
                    fixation = fixation.mean(dim=0, keepdim=True)
                
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