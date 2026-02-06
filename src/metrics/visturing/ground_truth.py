"""Utilidades para cargar y manejar archivos de ground truth (.mat)."""

import os
from typing import Any

import scipy.io as sio

from ...utils.download import ensure_ground_truth


def load_ground_truth_file(
    gt_path: str, filename: str, ensure_downloaded: bool = True
) -> dict[str, Any]:
    """
    Carga un archivo .mat de ground truth.

    Args:
        gt_path: Directorio base donde buscar/descargar ground truth
        filename: Nombre del archivo .mat (ej: 'spectral_sensitivities.mat')
        ensure_downloaded: Si True, descarga ground truth si no existe

    Returns:
        Diccionario con los datos del archivo .mat
    """
    if ensure_downloaded:
        gt_folder = ensure_ground_truth(gt_path)
    else:
        gt_folder = os.path.join(gt_path, "ground_truth")

    mat_path = os.path.join(gt_folder, filename)

    if not os.path.exists(mat_path):
        raise FileNotFoundError(
            f"Ground truth file not found: {mat_path}. "
            f"Try setting ensure_downloaded=True"
        )

    return sio.loadmat(mat_path)
