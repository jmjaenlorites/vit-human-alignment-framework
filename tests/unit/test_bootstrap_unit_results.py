import json

import pandas as pd
import pandas.testing as pdt
import pytest

from scripts.bootstrap_unit_results import (
    BootstrapConfig,
    main,
    summarize_unit_results,
)


def _row(
    *,
    metric_name: str,
    layer_index: int,
    unit_index: int,
    unit_value="",
    human_value="",
    prediction="",
    is_correct="",
    experiment_id: str = "exp",
) -> dict[str, object]:
    return {
        "run_key": f"vit-b16__{experiment_id}",
        "model_name": "vit-b16",
        "experiment_id": experiment_id,
        "metric_name": metric_name,
        "config_hash": "abc123",
        "layer_index": layer_index,
        "unit_index": unit_index,
        "unit_value": unit_value,
        "human_value": human_value,
        "prediction": prediction,
        "is_correct": is_correct,
    }


def test_accuracy_bootstrap_summary_is_deterministic_and_uses_chance_null() -> None:
    rows = []
    for layer_index, values in {0: [1, 0, 1, 1], 1: [0, 0, 1, 0]}.items():
        for unit_index, is_correct in enumerate(values):
            rows.append(
                _row(
                    metric_name="nights_preference_accuracy",
                    layer_index=layer_index,
                    unit_index=unit_index,
                    is_correct=is_correct,
                )
            )
    sidecar = pd.DataFrame(rows)
    config = BootstrapConfig(n_bootstraps=200, seed=123, include_null=True)

    first = summarize_unit_results(sidecar, config)
    second = summarize_unit_results(sidecar, config)

    pdt.assert_frame_equal(first.layer_summary, second.layer_summary)
    pdt.assert_frame_equal(first.best_layer_summary, second.best_layer_summary)
    pdt.assert_frame_equal(first.null_layer_summary, second.null_layer_summary)
    pdt.assert_frame_equal(
        first.null_best_layer_summary,
        second.null_best_layer_summary,
    )

    layer_summary = first.layer_summary.sort_values("layer_index").reset_index(
        drop=True
    )
    assert layer_summary.loc[0, "aggregate"] == pytest.approx(0.75)
    assert layer_summary.loc[1, "aggregate"] == pytest.approx(0.25)
    assert set(layer_summary["n_units"]) == {4}
    assert (
        layer_summary.loc[0, "ci_lower"]
        <= layer_summary.loc[0, "aggregate"]
        <= layer_summary.loc[0, "ci_upper"]
    )

    best = first.best_layer_summary.iloc[0]
    assert best["best_score"] == pytest.approx(0.75)
    assert best["best_layer_index"] == 0
    assert json.loads(best["best_layer_counts"])

    null_summary = first.null_layer_summary.sort_values("layer_index")
    assert set(null_summary["null_method"]) == {"chance"}
    assert set(null_summary["chance"]) == {0.5}
    assert set(null_summary["n_units"]) == {4}


def test_saliency_bootstrap_uses_unit_value_mean_without_null() -> None:
    sidecar = pd.DataFrame(
        [
            _row(
                metric_name="saliency_auc_judd",
                layer_index=0,
                unit_index=unit_index,
                unit_value=value,
            )
            for unit_index, value in enumerate([0.1, 0.2, 0.3, 0.4])
        ]
    )

    summaries = summarize_unit_results(
        sidecar,
        BootstrapConfig(n_bootstraps=100, seed=456, include_null=True),
    )

    layer = summaries.layer_summary.iloc[0]
    assert layer["mode"] == "saliency"
    assert layer["aggregate"] == pytest.approx(0.25)
    assert layer["value_column"] == "unit_value"
    assert layer["ci_lower"] <= layer["aggregate"] <= layer["ci_upper"]
    assert summaries.null_layer_summary.empty
    assert summaries.null_best_layer_summary.empty


def test_tid_spearman_bootstrap_and_human_value_permutation_null() -> None:
    rows = []
    for unit_index, human_value in enumerate([1, 2, 3, 4, 5, 6]):
        rows.append(
            _row(
                metric_name="tid_spearman_mos",
                layer_index=0,
                unit_index=unit_index,
                unit_value=human_value,
                human_value=human_value,
                experiment_id="tid",
            )
        )
        rows.append(
            _row(
                metric_name="tid_spearman_mos",
                layer_index=1,
                unit_index=unit_index,
                unit_value=7 - human_value,
                human_value=human_value,
                experiment_id="tid",
            )
        )
    sidecar = pd.DataFrame(rows)
    config = BootstrapConfig(n_bootstraps=150, seed=987, include_null=True)

    summaries = summarize_unit_results(sidecar, config)
    layer_summary = summaries.layer_summary.sort_values("layer_index").reset_index(
        drop=True
    )

    assert layer_summary.loc[0, "aggregate"] == pytest.approx(1.0)
    assert layer_summary.loc[1, "aggregate"] == pytest.approx(-1.0)
    assert set(layer_summary["n_units"]) == {6}

    best = summaries.best_layer_summary.iloc[0]
    assert best["best_score"] == pytest.approx(1.0)
    assert best["best_layer_index"] == 0
    assert best["best_relative_depth"] == pytest.approx(0.0)

    null_summary = summaries.null_layer_summary
    assert set(null_summary["null_method"]) == {"human_value_permutation"}
    assert null_summary["aggregate"].between(-1, 1).all()


def test_psychophysics_pearson_bootstrap_uses_unit_and_human_values() -> None:
    sidecar = pd.DataFrame(
        [
            _row(
                metric_name="visturing_weber_law_pearson",
                layer_index=0,
                unit_index=unit_index,
                unit_value=human_value * 2,
                human_value=human_value,
                experiment_id="weber",
            )
            for unit_index, human_value in enumerate([1, 2, 3, 4, 5, 6])
        ]
    )

    summaries = summarize_unit_results(
        sidecar,
        BootstrapConfig(n_bootstraps=100, seed=135, include_null=True),
    )

    layer = summaries.layer_summary.iloc[0]
    assert layer["mode"] == "pearson"
    assert layer["aggregate"] == pytest.approx(1.0)
    assert layer["value_column"] == "unit_value,human_value"
    assert summaries.null_layer_summary.empty


def test_accuracy_null_uses_label_permutation_with_predictions_and_labels() -> None:
    labels = ["a", "b", "c", "a", "b", "c"]
    sidecar = pd.DataFrame(
        [
            _row(
                metric_name="things_triplet_accuracy",
                layer_index=0,
                unit_index=unit_index,
                unit_value=1,
                human_value=label,
                prediction=label,
                is_correct=1,
                experiment_id="things",
            )
            for unit_index, label in enumerate(labels)
        ]
    )
    config = BootstrapConfig(n_bootstraps=100, seed=321, include_null=True)

    summaries = summarize_unit_results(sidecar, config)
    repeat = summarize_unit_results(sidecar, config)

    pdt.assert_frame_equal(summaries.null_layer_summary, repeat.null_layer_summary)
    observed = summaries.layer_summary.iloc[0]
    null = summaries.null_layer_summary.iloc[0]

    assert observed["aggregate"] == pytest.approx(1.0)
    assert null["null_method"] == "label_permutation"
    assert null["chance"] == pytest.approx(1 / 3)
    assert null["value_column"] == "prediction,human_value"
    assert 0 <= null["aggregate"] <= 1


def test_cli_writes_summary_csvs(tmp_path, capsys) -> None:
    sidecar_path = tmp_path / "sidecar.csv"
    pd.DataFrame(
        [
            _row(
                metric_name="levels_triplet_accuracy",
                layer_index=layer_index,
                unit_index=unit_index,
                is_correct=is_correct,
            )
            for layer_index, values in {0: [1, 1, 0], 1: [0, 1, 0]}.items()
            for unit_index, is_correct in enumerate(values)
        ]
    ).to_csv(sidecar_path, index=False)
    output_dir = tmp_path / "summaries"

    exit_code = main(
        [
            str(sidecar_path),
            "--output-dir",
            str(output_dir),
            "--n-bootstraps",
            "50",
            "--seed",
            "7",
            "--include-null",
        ]
    )

    assert exit_code == 0
    capsys.readouterr()
    expected_paths = [
        output_dir / "unit_layer_bootstrap_summary.csv",
        output_dir / "unit_best_layer_bootstrap_summary.csv",
        output_dir / "unit_layer_null_summary.csv",
        output_dir / "unit_best_layer_null_summary.csv",
    ]
    for path in expected_paths:
        assert path.exists()

    layer_summary = pd.read_csv(expected_paths[0])
    assert len(layer_summary) == 2
    assert set(layer_summary["n_bootstraps"]) == {50}
