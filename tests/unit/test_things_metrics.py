"""Unit tests for Things metrics."""

import json

import torch

from src.metrics.things import ThingsTripletAccuracy
from src.utils.common_enums import BackendEnum


def _build_features(num_layers, num_samples, hidden_dim, base_pattern):
    """Helper para construir features per-layer dado un patrón por sample.

    `base_pattern[i]` indica qué par de imágenes deben ser similares para el
    sample `i`: 'pair_12', 'pair_13' o 'pair_23'.
    """
    features_img1 = []
    features_img2 = []
    features_img3 = []

    for _layer_idx in range(num_layers):
        layer_feat1 = []
        layer_feat2 = []
        layer_feat3 = []

        for i in range(num_samples):
            base = torch.randn(hidden_dim)
            pattern = base_pattern[i]
            if pattern == "pair_12":
                feat1 = base + torch.randn(hidden_dim) * 0.01
                feat2 = base + torch.randn(hidden_dim) * 0.01
                feat3 = torch.randn(hidden_dim)
            elif pattern == "pair_13":
                feat1 = base + torch.randn(hidden_dim) * 0.01
                feat2 = torch.randn(hidden_dim)
                feat3 = base + torch.randn(hidden_dim) * 0.01
            elif pattern == "pair_23":
                feat1 = torch.randn(hidden_dim)
                feat2 = base + torch.randn(hidden_dim) * 0.01
                feat3 = base + torch.randn(hidden_dim) * 0.01
            else:
                raise ValueError(f"Unknown pattern: {pattern}")

            layer_feat1.append(feat1)
            layer_feat2.append(feat2)
            layer_feat3.append(feat3)

        features_img1.append(torch.stack(layer_feat1))
        features_img2.append(torch.stack(layer_feat2))
        features_img3.append(torch.stack(layer_feat3))

    return features_img1, features_img2, features_img3


class TestThingsTripletAccuracy:
    """Tests para ThingsTripletAccuracy."""

    def test_outlier_is_img3(self):
        """Cuando img1 e img2 son similares, el outlier es img3."""
        metric = ThingsTripletAccuracy(BackendEnum.TORCH)

        num_samples = 10
        num_layers = 3
        hidden_dim = 768

        patterns = ["pair_12"] * num_samples
        features_img1, features_img2, features_img3 = _build_features(
            num_layers, num_samples, hidden_dim, patterns
        )

        # choice apunta al path relativo de img3 (es el outlier)
        choice_batch = [f"THINGS/Images/cat3/img3_{i}.jpg" for i in range(num_samples)]
        img1_relpaths = [f"THINGS/Images/cat1/img1_{i}.jpg" for i in range(num_samples)]
        img2_relpaths = [f"THINGS/Images/cat2/img2_{i}.jpg" for i in range(num_samples)]
        img3_relpaths = [f"THINGS/Images/cat3/img3_{i}.jpg" for i in range(num_samples)]

        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            choice_batch,
            img1_relpaths,
            img2_relpaths,
            img3_relpaths,
        )
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }

        metric.calculate(batch, batch_results)
        results = metric.finalize()

        accuracies = json.loads(results["things_triplet_accuracy"])
        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected 1.0, got {acc}"

    def test_outlier_is_img1(self):
        """Cuando img2 e img3 son similares, el outlier es img1."""
        metric = ThingsTripletAccuracy(BackendEnum.TORCH)

        num_samples = 10
        num_layers = 2
        hidden_dim = 768

        patterns = ["pair_23"] * num_samples
        features_img1, features_img2, features_img3 = _build_features(
            num_layers, num_samples, hidden_dim, patterns
        )

        choice_batch = [f"THINGS/Images/cat1/img1_{i}.jpg" for i in range(num_samples)]
        img1_relpaths = [f"THINGS/Images/cat1/img1_{i}.jpg" for i in range(num_samples)]
        img2_relpaths = [f"THINGS/Images/cat2/img2_{i}.jpg" for i in range(num_samples)]
        img3_relpaths = [f"THINGS/Images/cat3/img3_{i}.jpg" for i in range(num_samples)]

        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            choice_batch,
            img1_relpaths,
            img2_relpaths,
            img3_relpaths,
        )
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }

        metric.calculate(batch, batch_results)
        accuracies = json.loads(metric.finalize()["things_triplet_accuracy"])

        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected 1.0, got {acc}"

    def test_outlier_is_img2(self):
        """Cuando img1 e img3 son similares, el outlier es img2."""
        metric = ThingsTripletAccuracy(BackendEnum.TORCH)

        num_samples = 10
        num_layers = 2
        hidden_dim = 768

        patterns = ["pair_13"] * num_samples
        features_img1, features_img2, features_img3 = _build_features(
            num_layers, num_samples, hidden_dim, patterns
        )

        choice_batch = [f"THINGS/Images/cat2/img2_{i}.jpg" for i in range(num_samples)]
        img1_relpaths = [f"THINGS/Images/cat1/img1_{i}.jpg" for i in range(num_samples)]
        img2_relpaths = [f"THINGS/Images/cat2/img2_{i}.jpg" for i in range(num_samples)]
        img3_relpaths = [f"THINGS/Images/cat3/img3_{i}.jpg" for i in range(num_samples)]

        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            choice_batch,
            img1_relpaths,
            img2_relpaths,
            img3_relpaths,
        )
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }

        metric.calculate(batch, batch_results)
        accuracies = json.loads(metric.finalize()["things_triplet_accuracy"])

        for layer_idx, acc in enumerate(accuracies):
            assert acc == 1.0, f"Layer {layer_idx}: Expected 1.0, got {acc}"

    def test_random_prediction(self):
        """Features aleatorios → precisión cercana a chance (~1/3)."""
        metric = ThingsTripletAccuracy(BackendEnum.TORCH)

        num_samples = 30
        num_layers = 2
        hidden_dim = 768

        features_img1 = [torch.randn(num_samples, hidden_dim) for _ in range(num_layers)]
        features_img2 = [torch.randn(num_samples, hidden_dim) for _ in range(num_layers)]
        features_img3 = [torch.randn(num_samples, hidden_dim) for _ in range(num_layers)]

        import random

        choice_options = ["img1", "img2", "img3"]
        choice_batch = [
            f"THINGS/Images/cat/{random.choice(choice_options)}_{i}.jpg"
            for i in range(num_samples)
        ]
        img1_relpaths = [f"THINGS/Images/cat/img1_{i}.jpg" for i in range(num_samples)]
        img2_relpaths = [f"THINGS/Images/cat/img2_{i}.jpg" for i in range(num_samples)]
        img3_relpaths = [f"THINGS/Images/cat/img3_{i}.jpg" for i in range(num_samples)]

        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            choice_batch,
            img1_relpaths,
            img2_relpaths,
            img3_relpaths,
        )
        batch_results = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }

        metric.calculate(batch, batch_results)
        accuracies = json.loads(metric.finalize()["things_triplet_accuracy"])

        for layer_idx, acc in enumerate(accuracies):
            assert 0.0 <= acc <= 0.7, (
                f"Layer {layer_idx}: Random expected ~0.33, got {acc}"
            )

    def test_accumulation_across_batches(self):
        """Verifica que el estado se acumula entre llamadas a calculate."""
        metric = ThingsTripletAccuracy(BackendEnum.TORCH)

        num_layers = 3
        hidden_dim = 768

        for batch_size in (5, 7):
            features_img1 = [
                torch.randn(batch_size, hidden_dim) for _ in range(num_layers)
            ]
            features_img2 = [
                torch.randn(batch_size, hidden_dim) for _ in range(num_layers)
            ]
            features_img3 = [
                torch.randn(batch_size, hidden_dim) for _ in range(num_layers)
            ]

            choice_batch = [
                f"THINGS/Images/cat/img1_{i}.jpg" for i in range(batch_size)
            ]
            img1_relpaths = [
                f"THINGS/Images/cat/img1_{i}.jpg" for i in range(batch_size)
            ]
            img2_relpaths = [
                f"THINGS/Images/cat/img2_{i}.jpg" for i in range(batch_size)
            ]
            img3_relpaths = [
                f"THINGS/Images/cat/img3_{i}.jpg" for i in range(batch_size)
            ]

            batch = (
                torch.zeros(batch_size, 3, 224, 224),
                torch.zeros(batch_size, 3, 224, 224),
                torch.zeros(batch_size, 3, 224, 224),
                choice_batch,
                img1_relpaths,
                img2_relpaths,
                img3_relpaths,
            )
            batch_results = {
                "features_img1": features_img1,
                "features_img2": features_img2,
                "features_img3": features_img3,
            }

            metric.calculate(batch, batch_results)

        assert len(metric.correct_per_layer) == num_layers
        assert metric.total == 5 + 7

        accuracies = json.loads(metric.finalize()["things_triplet_accuracy"])
        assert len(accuracies) == num_layers

    def test_reset(self):
        """reset() limpia los acumuladores."""
        metric = ThingsTripletAccuracy(BackendEnum.TORCH)

        num_samples = 5
        num_layers = 2
        hidden_dim = 768

        features = [torch.randn(num_samples, hidden_dim) for _ in range(num_layers)]
        choice_batch = [f"THINGS/Images/cat/img1_{i}.jpg" for i in range(num_samples)]
        img1_relpaths = list(choice_batch)
        img2_relpaths = [f"THINGS/Images/cat/img2_{i}.jpg" for i in range(num_samples)]
        img3_relpaths = [f"THINGS/Images/cat/img3_{i}.jpg" for i in range(num_samples)]

        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            choice_batch,
            img1_relpaths,
            img2_relpaths,
            img3_relpaths,
        )
        batch_results = {
            "features_img1": features,
            "features_img2": features,
            "features_img3": features,
        }

        metric.calculate(batch, batch_results)
        assert len(metric.correct_per_layer) > 0
        assert metric.total > 0

        metric.reset()
        assert metric.correct_per_layer == []
        assert metric.total == 0

    def test_jax_backend_matches_torch(self):
        """Backend JAX produce resultados consistentes con Torch para los
        mismos features deterministas."""
        import jax.numpy as jnp

        num_samples = 6
        num_layers = 2
        hidden_dim = 64

        torch.manual_seed(0)
        patterns = ["pair_12", "pair_23", "pair_13"] * 2
        features_img1, features_img2, features_img3 = _build_features(
            num_layers, num_samples, hidden_dim, patterns
        )

        choice_batch = []
        for pattern in patterns:
            if pattern == "pair_12":
                choice_batch.append("img3")
            elif pattern == "pair_23":
                choice_batch.append("img1")
            else:
                choice_batch.append("img2")
        img1_relpaths = ["img1"] * num_samples
        img2_relpaths = ["img2"] * num_samples
        img3_relpaths = ["img3"] * num_samples

        batch = (
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            torch.zeros(num_samples, 3, 224, 224),
            choice_batch,
            img1_relpaths,
            img2_relpaths,
            img3_relpaths,
        )
        batch_results_torch = {
            "features_img1": features_img1,
            "features_img2": features_img2,
            "features_img3": features_img3,
        }

        torch_metric = ThingsTripletAccuracy(BackendEnum.TORCH)
        torch_metric.calculate(batch, batch_results_torch)
        torch_acc = json.loads(torch_metric.finalize()["things_triplet_accuracy"])

        # Convertimos features per-layer a jnp.ndarray (mismo contenido).
        batch_results_jax = {
            "features_img1": [jnp.asarray(f.numpy()) for f in features_img1],
            "features_img2": [jnp.asarray(f.numpy()) for f in features_img2],
            "features_img3": [jnp.asarray(f.numpy()) for f in features_img3],
        }
        jax_metric = ThingsTripletAccuracy(BackendEnum.JAX)
        jax_metric.calculate(batch, batch_results_jax)
        jax_acc = json.loads(jax_metric.finalize()["things_triplet_accuracy"])

        for layer_idx, (t, j) in enumerate(zip(torch_acc, jax_acc)):
            assert abs(t - j) < 1e-5, (
                f"Layer {layer_idx}: torch={t} vs jax={j}"
            )
