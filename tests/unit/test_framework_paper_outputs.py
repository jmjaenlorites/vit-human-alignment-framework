import pandas as pd
import pytest

from scripts.framework_paper_outputs import (
    _support_label,
    summarize_contrast_delta,
)


def _row(
    *,
    model_name: str,
    experiment_id: str,
    metric_name: str,
    layer_index: int,
    unit_index: int,
    unit_value="",
    human_value="",
    is_correct="",
) -> dict[str, object]:
    return {
        "run_key": f"{model_name}__{experiment_id}",
        "model_name": model_name,
        "experiment_id": experiment_id,
        "metric_name": metric_name,
        "config_hash": "abc123",
        "layer_index": layer_index,
        "unit_index": unit_index,
        "unit_value": unit_value,
        "human_value": human_value,
        "prediction": "",
        "is_correct": is_correct,
    }


def test_contrast_delta_reselects_best_layer_inside_bootstrap() -> None:
    reference_rows = []
    comparison_rows = []
    reference_values = {
        0: [1, 0, 1, 0, 1, 0],
        1: [0, 0, 0, 0, 0, 0],
    }
    comparison_values = {
        0: [1, 1, 1, 1, 1, 1],
        1: [1, 1, 0, 1, 1, 0],
    }
    for layer_index in [0, 1]:
        for unit_index, value in enumerate(reference_values[layer_index]):
            reference_rows.append(
                _row(
                    model_name="ref",
                    experiment_id="nights-preference-accuracy",
                    metric_name="nights_preference_accuracy",
                    layer_index=layer_index,
                    unit_index=unit_index,
                    is_correct=value,
                )
            )
        for unit_index, value in enumerate(comparison_values[layer_index]):
            comparison_rows.append(
                _row(
                    model_name="cmp",
                    experiment_id="nights-preference-accuracy",
                    metric_name="nights_preference_accuracy",
                    layer_index=layer_index,
                    unit_index=unit_index,
                    is_correct=value,
                )
            )

    summary = summarize_contrast_delta(
        pd.DataFrame(reference_rows),
        pd.DataFrame(comparison_rows),
        factor="recipe",
        contrast="cmp - ref",
        experiment_id="nights-preference-accuracy",
        reference_model="ref",
        comparison_model="cmp",
        n_bootstraps=200,
        seed=123,
    )

    assert summary["final_delta"] == pytest.approx(4 / 6)
    assert summary["best_delta"] == pytest.approx(0.5)
    assert summary["final_support"] == "directional+"
    assert summary["best_support"] == "directional+"
    assert summary["n_units"] == 6
    assert summary["reference_n_layers"] == 2
    assert summary["comparison_n_layers"] == 2


def test_contrast_delta_uses_rank_correlation_for_tid() -> None:
    reference_rows = []
    comparison_rows = []
    human = [1, 2, 3, 4, 5, 6]
    for unit_index, human_value in enumerate(human):
        reference_rows.append(
            _row(
                model_name="ref",
                experiment_id="tid-spearman-mos",
                metric_name="tid_spearman_mos",
                layer_index=0,
                unit_index=unit_index,
                unit_value=7 - human_value,
                human_value=human_value,
            )
        )
        comparison_rows.append(
            _row(
                model_name="cmp",
                experiment_id="tid-spearman-mos",
                metric_name="tid_spearman_mos",
                layer_index=0,
                unit_index=unit_index,
                unit_value=human_value,
                human_value=human_value,
            )
        )

    summary = summarize_contrast_delta(
        pd.DataFrame(reference_rows),
        pd.DataFrame(comparison_rows),
        factor="recipe",
        contrast="cmp - ref",
        experiment_id="tid-spearman-mos",
        reference_model="ref",
        comparison_model="cmp",
        n_bootstraps=100,
        seed=456,
    )

    assert summary["final_delta"] == pytest.approx(2.0)
    assert summary["best_delta"] == pytest.approx(2.0)
    assert summary["final_support"] == "directional+"


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (0.1, 0.2, "directional+"),
        (-0.2, -0.1, "directional-"),
        (-0.1, 0.2, "ambiguous"),
    ],
)
def test_support_label(lower: float, upper: float, expected: str) -> None:
    assert _support_label(lower, upper) == expected
