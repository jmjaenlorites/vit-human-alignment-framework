"""Utilidad para descarga de datasets desde Zenodo."""

import os
from zipfile import ZipFile

import wget


ZENODO_BASE_URL = "https://zenodo.org/records/17700252/files/"


def download_and_extract(
    filename: str, target_path: str, extract_folder: str | None = None
) -> str:
    """
    Descarga un archivo .zip desde Zenodo y lo extrae.

    Args:
        filename: Nombre del archivo a descargar (ej: 'Experiment_1.zip')
        target_path: Directorio donde descargar y extraer
        extract_folder: Nombre de la carpeta extraída (por defecto el nombre sin .zip)

    Returns:
        Ruta al directorio extraído
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    url = ZENODO_BASE_URL + filename
    download_path = os.path.join(target_path, filename)

    # Descargar si no existe
    if not os.path.exists(download_path):
        print(f"Downloading {filename} from Zenodo...")
        wget.download(url, download_path)
        print()  # Nueva línea después de la barra de progreso

    # Determinar carpeta de extracción
    if extract_folder is None:
        extract_folder = filename.replace(".zip", "")

    extracted_path = os.path.join(target_path, extract_folder)

    # Extraer si no existe
    if not os.path.exists(extracted_path):
        print(f"Extracting {filename}...")
        with ZipFile(download_path) as zipObj:
            zipObj.extractall(target_path)

    # Eliminar el zip para ahorrar espacio
    if os.path.exists(download_path):
        os.remove(download_path)

    return extracted_path


def ensure_ground_truth(gt_path: str) -> str:
    """
    Asegura que los archivos de ground truth estén disponibles.

    Args:
        gt_path: Directorio base donde descargar

    Returns:
        Ruta al directorio 'ground_truth'
    """
    gt_folder = os.path.join(gt_path, "ground_truth")

    if not os.path.exists(gt_folder):
        print("Ground truth files not found. Downloading...")
        download_and_extract("ground_truth.zip", gt_path, "ground_truth")

    return gt_folder
