"""Integration tests for visturing calculator factories."""

import os

import pytest
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


def _has_experiment(path: str, experiment_name: str) -> bool:
    return os.path.exists(os.path.join(path, experiment_name))


class TestVisturingCalculatorFactories:
    """Tests that visturing factory functions build calculators correctly."""

    def test_prop1_factory(self, visturing_test_data_path):
        if not _has_experiment(visturing_test_data_path, "Experiment_1"):
            pytest.skip("Experiment_1 not available in tests/data/visturing")

        calculator = create_prop1_calculator(
            backend=BackendEnum.TORCH,
            data_path=visturing_test_data_path,
            gt_path=visturing_test_data_path,
            batch_size=2,
        )

        assert calculator._backend == BackendEnum.TORCH
        assert len(calculator._metrics) == 1

    def test_prop2_factory(self, visturing_test_data_path):
        if not _has_experiment(visturing_test_data_path, "Experiment_2"):
            pytest.skip("Experiment_2 not available in tests/data/visturing")

        calculator = create_prop2_calculator(
            backend=BackendEnum.TORCH,
            data_path=visturing_test_data_path,
            gt_path=visturing_test_data_path,
            batch_size=2,
            include_kendall=True,
        )

        assert calculator._backend == BackendEnum.TORCH
        assert len(calculator._metrics) == 2

    def test_prop3_4_factory(self, visturing_test_data_path):
        if not _has_experiment(visturing_test_data_path, "Experiment_3_4"):
            pytest.skip("Experiment_3_4 not available in tests/data/visturing")

        calculator = create_prop3_4_calculator(
            backend=BackendEnum.TORCH,
            data_path=visturing_test_data_path,
            gt_path=visturing_test_data_path,
            batch_size=2,
            include_kendall=True,
        )

        assert calculator._backend == BackendEnum.TORCH
        assert len(calculator._metrics) == 2

    def test_prop5_to_10_factories(self, visturing_test_data_path):
        required = [
            "Experiment_5",
            "Experiment_6_7",
            "Experiment_8",
            "Experiment_9",
            "Experiment_10",
        ]
        if not all(_has_experiment(visturing_test_data_path, exp) for exp in required):
            pytest.skip(
                "Not all Experiment_5..10 datasets are available in tests/data/visturing"
            )

        calculators = [
            create_prop5_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=2,
                include_kendall=True,
            ),
            create_prop6_7_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=2,
                include_kendall=True,
            ),
            create_prop8_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=2,
            ),
            create_prop9_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=2,
            ),
            create_prop10_calculator(
                backend=BackendEnum.TORCH,
                data_path=visturing_test_data_path,
                gt_path=visturing_test_data_path,
                batch_size=2,
            ),
        ]

        assert all(c._backend == BackendEnum.TORCH for c in calculators)
        assert [len(c._metrics) for c in calculators] == [2, 2, 1, 1, 1]
