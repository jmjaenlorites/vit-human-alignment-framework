"""Parity checks between framework Visturing metrics and upstream results."""

import json
import math

import jax.numpy as jnp
import pytest
import torch

from src.metrics.visturing import (
    create_prop1_calculator,
    create_prop2_calculator,
    create_prop3_4_calculator,
    create_prop5_calculator,
    create_prop6_7_calculator,
    create_prop8_calculator,
    create_prop9_calculator,
    create_prop10_calculator,
)
from src.utils.common_enums import BackendEnum


class _IdentityTorchModel:
    backend = BackendEnum.TORCH
    device = torch.device("cpu")
    transform = None

    def forward(self, batch, return_features=False, return_saliency=False):
        return {"features": [batch.float()]}


class _IdentityJaxModel:
    backend = BackendEnum.JAX
    device = None
    transform = None

    def forward(self, batch, return_features=False, return_saliency=False):
        if isinstance(batch, torch.Tensor):
            batch = batch.numpy()
        return {"features": [jnp.asarray(batch)]}


def _prepare_calculator(calculator):
    calculator._dataset_loader.num_workers = 0
    calculator._dataset_loader.transform = None
    return calculator


class TestVisturingTorchParity:
    def test_prop1_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop1_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
            )
        )

        results = calculator.run(_IdentityTorchModel())
        pearson = json.loads(results["visturing_spectral_sensitivity"])[0]

        assert pearson == pytest.approx(0.6297438817505754, abs=1e-6)

    def test_prop2_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop2_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
                include_kendall=True,
            )
        )

        results = calculator.run(_IdentityTorchModel())
        pearson = json.loads(results["visturing_weber_law_pearson"])[0]
        kendall = json.loads(results["visturing_weber_law_kendall"])[0]

        assert pearson["pearson_achrom"] == pytest.approx(0.9727145987459574, abs=1e-6)
        assert pearson["pearson_chrom"] == pytest.approx(0.9352674344116237, abs=1e-6)
        assert kendall["achrom"]["kendall"] == pytest.approx(1.0, abs=1e-6)
        assert kendall["red_green"]["kendall"] == pytest.approx(1.0, abs=1e-6)
        assert kendall["yellow_blue"]["kendall"] == pytest.approx(1.0, abs=1e-6)

    def test_prop3_4_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop3_4_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
                include_kendall=True,
            )
        )

        results = calculator.run(_IdentityTorchModel())
        pearson = json.loads(results["visturing_csf_pearson"])[0]
        kendall = json.loads(results["visturing_csf_kendall"])[0]

        assert pearson == pytest.approx(-0.45481510136629494, abs=1e-6)
        assert kendall["kendall"] == pytest.approx(-0.3160416666666666, abs=1e-6)

    def test_prop5_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop5_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
                include_kendall=True,
            )
        )

        results = calculator.run(_IdentityTorchModel())
        pearson = json.loads(results["visturing_campbell_blakemore_pearson"])[0]
        kendall = json.loads(results["visturing_campbell_blakemore_kendall"])[0]

        assert math.isnan(pearson)
        assert kendall["kendall"] == pytest.approx(-0.6241507779969317, abs=1e-6)

    def test_prop6_7_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop6_7_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
                include_kendall=True,
            )
        )

        results = calculator.run(_IdentityTorchModel())
        pearson = json.loads(results["visturing_contrast_curves_pearson"])[0]
        kendall = json.loads(results["visturing_contrast_curves_kendall"])[0]

        assert pearson == pytest.approx(0.5952297324291396, abs=1e-6)
        assert kendall["a"]["kendall"] == pytest.approx(-0.32720661157024794, abs=1e-6)
        assert kendall["rg"]["kendall"] == pytest.approx(0.496793388429752, abs=1e-6)
        assert kendall["yb"]["kendall"] == pytest.approx(-0.2816528925619835, abs=1e-6)

    def test_prop8_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop8_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
            )
        )

        results = calculator.run(_IdentityTorchModel())
        payload = json.loads(results["visturing_contrast_masking_kendall"])[0]

        assert payload["pearson"] == pytest.approx(0.919813887162594, abs=1e-6)
        assert payload["kendall"]["low"]["kendall"] == pytest.approx(
            -0.847801652892562, abs=1e-6
        )
        assert payload["kendall"]["high"]["kendall"] == pytest.approx(
            -0.7721322314049587, abs=1e-6
        )

    def test_prop9_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop9_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
            )
        )

        results = calculator.run(_IdentityTorchModel())
        payload = json.loads(results["visturing_frequency_masking_kendall"])[0]

        assert payload["low"]["kendall"] == pytest.approx(-0.3333333333333333, abs=1e-6)
        assert payload["high"]["kendall"] == pytest.approx(
            -0.6749999999999999, abs=1e-6
        )

    def test_prop10_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop10_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
            )
        )

        results = calculator.run(_IdentityTorchModel())
        payload = json.loads(results["visturing_orientation_masking_kendall"])[0]

        assert payload["low"]["kendall"] == pytest.approx(0.03562499999999995, abs=1e-6)
        assert payload["high"]["kendall"] == pytest.approx(
            -0.6968749999999999, abs=1e-6
        )


class TestVisturingJaxParity:
    def test_prop1_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop1_calculator(
                backend=BackendEnum.JAX,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
            )
        )

        results = calculator.run(_IdentityJaxModel())
        pearson = json.loads(results["visturing_spectral_sensitivity"])[0]

        assert pearson == pytest.approx(0.6297439, abs=1e-6)

    def test_prop2_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop2_calculator(
                backend=BackendEnum.JAX,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
                include_kendall=True,
            )
        )

        results = calculator.run(_IdentityJaxModel())
        pearson = json.loads(results["visturing_weber_law_pearson"])[0]
        kendall = json.loads(results["visturing_weber_law_kendall"])[0]

        assert pearson["pearson_achrom"] == pytest.approx(0.97271484, abs=1e-6)
        assert pearson["pearson_chrom"] == pytest.approx(-0.12529413, abs=1e-6)
        assert kendall["achrom"]["kendall"] == pytest.approx(0.1189899, abs=1e-6)
        assert kendall["red_green"]["kendall"] == pytest.approx(0.11465202, abs=1e-6)
        assert kendall["yellow_blue"]["kendall"] == pytest.approx(-0.06007326, abs=1e-6)

    def test_prop3_4_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop3_4_calculator(
                backend=BackendEnum.JAX,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
                include_kendall=True,
            )
        )

        results = calculator.run(_IdentityJaxModel())
        pearson = json.loads(results["visturing_csf_pearson"])[0]
        kendall = json.loads(results["visturing_csf_kendall"])[0]

        assert pearson == pytest.approx(-0.45481515, abs=1e-6)
        assert kendall["kendall"] == pytest.approx(-0.21246499, abs=1e-6)

    def test_prop5_matches_upstream_reference(self, visturing_test_data_path):
        calculator = _prepare_calculator(
            create_prop5_calculator(
                backend=BackendEnum.JAX,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=8,
                include_kendall=True,
            )
        )

        results = calculator.run(_IdentityJaxModel())
        pearson = json.loads(results["visturing_campbell_blakemore_pearson"])[0]
        kendall = json.loads(results["visturing_campbell_blakemore_kendall"])[0]

        assert pearson == pytest.approx(-0.32044822, abs=1e-5)
        assert kendall["kendall"] == pytest.approx(-0.16136162, abs=1e-6)
