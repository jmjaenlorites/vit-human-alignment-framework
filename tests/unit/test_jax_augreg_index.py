"""Tests para el resolver del index AugReg.

Monkeypatchea la descarga del CSV con un fixture mínimo. No toca red ni disco.
"""

import io

import pandas as pd
import pytest

from src.models.jax import augreg_index
from src.models.jax.augreg_index import (
    GOOGLE_AUGREG_BASE_URL,
    list_available_google_models,
    load_augreg_dataframe,
    lookup_google_model,
)


_FIXTURE_CSV = """name,ds,epochs,lr,aug,wd,do,sd,best_val,final_val,final_test,adapt_ds,adapt_lr,adapt_steps,adapt_resolution,adapt_final_val,adapt_final_test,params,infer_samples_per_sec,filename,adapt_filename
Ti/16,i1k,300,0.001,light0,0.03,0.1,0.1,0.7,0.7,0.67,imagenet2012,0.03,20000,224,0.75,0.72,5790000,609,Ti_16-i1k-300ep,Ti_16-i1k-300ep--imagenet2012-res_224
B/16,i21k,300,0.001,medium1,0.1,0.1,0.1,0.45,0.45,0.0,imagenet2012,0.03,20000,384,0.85,0.84,86000000,150,B_16-i21k-300ep,B_16-i21k-300ep--imagenet2012-res_384
R+Ti/16,i1k,300,0.001,light0,0.03,0.1,0.1,0.7,0.7,0.65,imagenet2012,0.03,20000,224,0.74,0.72,7000000,500,R_Ti_16-i1k-300ep,R_Ti_16-i1k-300ep--imagenet2012-res_224
"""


@pytest.fixture(autouse=True)
def _patch_index(monkeypatch) -> None:
    load_augreg_dataframe.cache_clear()
    list_available_google_models.cache_clear()
    monkeypatch.setattr(
        augreg_index,
        "_download_index_csv",
        lambda: _FIXTURE_CSV.encode("utf-8"),
    )
    yield
    load_augreg_dataframe.cache_clear()
    list_available_google_models.cache_clear()


def test_load_augreg_dataframe_parses_csv() -> None:
    df = load_augreg_dataframe()
    assert len(df) == 3
    assert set(df["name"]) == {"Ti/16", "B/16", "R+Ti/16"}


def test_list_available_filters_unsupported_archs() -> None:
    names = list_available_google_models()
    # 4 entradas: 2 archs supported × 2 columnas (filename + adapt_filename)
    assert sorted(names) == [
        "B_16-i21k-300ep",
        "B_16-i21k-300ep--imagenet2012-res_384",
        "Ti_16-i1k-300ep",
        "Ti_16-i1k-300ep--imagenet2012-res_224",
    ]


def test_lookup_pretrain_filename() -> None:
    info = lookup_google_model("Ti_16-i1k-300ep")
    assert info.arch == "Ti/16"
    assert info.patch_size == 16
    assert info.image_size == 224
    assert info.hidden_size == 192
    assert info.num_layers == 12
    assert info.num_heads == 3
    assert info.mlp_dim == 768
    assert info.num_classes == 1000
    assert info.is_adapt is False
    assert info.checkpoint_url == f"{GOOGLE_AUGREG_BASE_URL}/Ti_16-i1k-300ep.npz"


def test_lookup_pretrain_i21k_uses_21843_classes() -> None:
    info = lookup_google_model("B_16-i21k-300ep")
    assert info.num_classes == 21843
    assert info.image_size == 224


def test_lookup_adapt_filename_uses_adapt_resolution() -> None:
    info = lookup_google_model("B_16-i21k-300ep--imagenet2012-res_384")
    assert info.arch == "B/16"
    assert info.image_size == 384
    assert info.num_classes == 1000
    assert info.is_adapt is True
    assert (
        info.checkpoint_url
        == f"{GOOGLE_AUGREG_BASE_URL}/B_16-i21k-300ep--imagenet2012-res_384.npz"
    )


def test_lookup_unknown_filename_raises() -> None:
    with pytest.raises(ValueError, match="not found"):
        lookup_google_model("Unknown_filename")


def test_lookup_unsupported_arch_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        lookup_google_model("R_Ti_16-i1k-300ep")
