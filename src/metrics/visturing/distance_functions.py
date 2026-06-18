"""
Funciones de distancia y correlación para métricas de visturing.

Adaptado de visturing/ranking.py para trabajar con features de modelos.
"""

import numpy as np
import scipy.stats as stats

from ...utils.common_types import ArrayLike


def euclidean_distance(features1: ArrayLike, features2: ArrayLike) -> ArrayLike:
    """
    Calcula distancia euclidiana entre features.

    Args:
        features1: Features de test [batch, dim]
        features2: Features de referencia [batch, dim] o [1, dim]

    Returns:
        Array de distancias [batch]
    """
    # Aplanar features si es necesario
    feat1 = features1.reshape(features1.shape[0], -1)
    feat2 = features2.reshape(features2.shape[0], -1)

    # Calcular distancia euclidiana
    diff = np.sqrt(((feat1 - feat2) ** 2).sum(axis=-1))

    return diff


def cosine_distance(features1: ArrayLike, features2: ArrayLike) -> ArrayLike:
    """
    Calcula distancia coseno entre features (1 - similitud coseno).

    Args:
        features1: Features de test [batch, dim]
        features2: Features de referencia [batch, dim] o [1, dim]

    Returns:
        Array de distancias [batch]
    """
    # Aplanar features si es necesario
    feat1 = features1.reshape(features1.shape[0], -1)
    feat2 = features2.reshape(features2.shape[0], -1)

    # Calcular similitud coseno
    dot_product = (feat1 * feat2).sum(axis=-1)
    norm1 = np.sqrt((feat1**2).sum(axis=-1))
    norm2 = np.sqrt((feat2**2).sum(axis=-1))

    cosine_sim = dot_product / (norm1 * norm2 + 1e-8)

    # Distancia = 1 - similitud
    return 1.0 - cosine_sim


# Funciones de correlación (adaptadas de visturing/ranking.py)


def calculate_correlations(ground_truth: np.ndarray, experimental: np.ndarray) -> dict:
    """
    Calcula múltiples correlaciones entre ground truth y valores experimentales.

    Args:
        ground_truth: Valores de ground truth
        experimental: Valores experimentales (diferencias del modelo)

    Returns:
        Dict con correlaciones de Spearman, Kendall y Pearson
    """
    return {
        "spearman": stats.spearmanr(ground_truth.ravel(), experimental.ravel())[0],
        "kendall": stats.kendalltau(ground_truth.ravel(), experimental.ravel())[0],
        "pearson": stats.pearsonr(ground_truth.ravel(), experimental.ravel())[0],
    }


def pearson_correlation_jax(vec1: np.ndarray, vec2: np.ndarray):
    """Calcula Pearson con la misma formulacion usada por visturing JAX."""
    import jax.numpy as jnp

    vec1_jax = jnp.asarray(vec1).squeeze()
    vec2_jax = jnp.asarray(vec2).squeeze()
    vec1_mean = vec1_jax.mean()
    vec2_mean = vec2_jax.mean()
    num = vec1_jax - vec1_mean
    num *= vec2_jax - vec2_mean
    num = num.sum()
    denom = ((vec1_jax - vec1_mean) ** 2).sum() ** 0.5
    denom *= ((vec2_jax - vec2_mean) ** 2).sum() ** 0.5
    return num / denom


def kendall_correlation_jax(x: np.ndarray, y: np.ndarray):
    """Calcula Kendall tau con la formulacion usada por visturing JAX."""
    import jax.numpy as jnp

    x_jax = jnp.asarray(x)
    y_jax = jnp.asarray(y)
    n = x_jax.shape[0]
    diff_x = x_jax[:, None] - x_jax[None, :]
    diff_y = y_jax[:, None] - y_jax[None, :]
    soft_sign_x = jnp.tanh(diff_x / 0.1)
    soft_sign_y = jnp.tanh(diff_y / 0.1)
    concordance = soft_sign_x * soft_sign_y
    return jnp.sum(concordance) / (n * (n - 1))


def calculate_correlations_jax(
    ground_truth: np.ndarray, experimental: np.ndarray
) -> dict:
    """Calcula correlaciones con la semantica del backend JAX de visturing."""
    return {
        "kendall": kendall_correlation_jax(
            np.asarray(ground_truth).ravel(), np.asarray(experimental).ravel()
        )
    }


def calculate_spearman(
    experimental_curve: np.ndarray, ideal_ordering: list[int]
) -> dict:
    """
    Calcula correlaciones de orden comparando con un ordenamiento ideal.

    Args:
        experimental_curve: Curva experimental (diferencias del modelo)
        ideal_ordering: Lista con el orden ideal (ej: [0, 1, 2, 3, 4])

    Returns:
        Dict con correlaciones de Spearman y Kendall
    """
    # Ordenar curva experimental (mayor a menor)
    ordered_curves = (-experimental_curve).argsort(axis=0)

    # Crear array ideal con la misma forma
    ideal_ordering_array = np.array(ideal_ordering)[:, None].repeat(
        ordered_curves.shape[1], axis=1
    )

    return calculate_correlations(ordered_curves, ideal_ordering_array)


def calculate_spearman_jax(
    experimental_curve: np.ndarray, ideal_ordering: list[int]
) -> dict:
    """Calcula correlaciones de orden con la semantica del backend JAX."""
    ordered_curves = (-experimental_curve).argsort(axis=0)
    ideal_ordering_array = np.array(ideal_ordering)[:, None].repeat(
        ordered_curves.shape[1], axis=1
    )
    return calculate_correlations_jax(ordered_curves, ideal_ordering_array)


def calculate_correlations_with_ground_truth(
    experimental_curve: np.ndarray, ground_truth: np.ndarray
) -> dict:
    """
    Calcula correlaciones de orden entre curvas experimentales y ground truth.

    Args:
        experimental_curve: Curva experimental (diferencias del modelo)
        ground_truth: Ground truth

    Returns:
        Dict con correlaciones
    """
    gt_ordering = (-ground_truth).argsort(axis=0)
    e_ordering = (-experimental_curve).argsort(axis=0)
    return calculate_correlations(gt_ordering.ravel(), e_ordering.ravel())


def calculate_correlations_with_ground_truth_jax(
    experimental_curve: np.ndarray, ground_truth: np.ndarray
) -> dict:
    """Calcula correlaciones de orden con la semantica del backend JAX."""
    gt_ordering = (-ground_truth).argsort(axis=0)
    e_ordering = (-experimental_curve).argsort(axis=0)
    return calculate_correlations_jax(gt_ordering.ravel(), e_ordering.ravel())


def compare_ranges(x1: np.ndarray, x2: np.ndarray) -> bool:
    """
    Verifica si x1 contiene completamente el rango de x2.

    Args:
        x1: Primer array
        x2: Segundo array

    Returns:
        True si x1 contiene el rango de x2
    """
    m = x1.min() <= x2.min()
    M = x1.max() >= x2.max()
    return all((m, M))


def prepare_data(
    x_e: np.ndarray, y_e: np.ndarray, x_gt: np.ndarray, y_gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepara datos para comparación interpolando al rango común.

    Args:
        x_e: Valores x experimentales
        y_e: Valores y experimentales
        x_gt: Valores x ground truth
        y_gt: Valores y ground truth

    Returns:
        Tupla (x_e, y_e_interp, x_gt, y_gt_interp) en el rango común
    """
    gt_shorter = compare_ranges(x_e, x_gt)

    if gt_shorter:
        x_wider = x_e
        y_wider = y_e
        x_shorter = x_gt
        y_shorter = y_gt
    else:
        x_wider = x_gt
        y_wider = y_gt
        x_shorter = x_e
        y_shorter = y_e

    # Interpolar y_wider a x_shorter
    y_wider_interp = np.interp(x_shorter, x_wider, y_wider)

    if gt_shorter:
        return x_shorter, y_wider_interp, x_shorter, y_shorter
    else:
        return x_shorter, y_shorter, x_shorter, y_wider_interp


def prepare_and_correlate(
    x_e: np.ndarray, y_e: np.ndarray, x_gt: np.ndarray, y_gt: np.ndarray
) -> dict:
    """
    Prepara datos y calcula correlaciones.

    Args:
        x_e: Valores x experimentales
        y_e: Valores y experimentales
        x_gt: Valores x ground truth
        y_gt: Valores y ground truth

    Returns:
        Dict con correlaciones
    """
    x_e, y_e, x_gt, y_gt = prepare_data(x_e, y_e, x_gt, y_gt)
    return calculate_correlations(y_e, y_gt)


def prepare_and_correlate_order(
    x_e: np.ndarray, y_e: np.ndarray, x_gt: np.ndarray, y_gt: np.ndarray
) -> dict:
    """
    Prepara datos y calcula correlaciones de orden.

    Args:
        x_e: Valores x experimentales
        y_e: Valores y experimentales
        x_gt: Valores x ground truth
        y_gt: Valores y ground truth

    Returns:
        Dict con correlaciones de orden
    """
    x_e, y_e, x_gt, y_gt = prepare_data(x_e, y_e, x_gt, y_gt)
    return calculate_correlations_with_ground_truth(y_e, y_gt)


def calculate_pearson_stack(s1: np.ndarray, s2: np.ndarray) -> tuple[float, float]:
    """
    Calcula correlación de Pearson entre dos arrays apilados.

    Args:
        s1: Primer array
        s2: Segundo array

    Returns:
        Tupla (correlación, p-value)
    """
    return stats.pearsonr(s1.ravel(), s2.ravel())
