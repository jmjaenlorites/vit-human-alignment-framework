from pathlib import Path

import pytest

from src.models.checkpoint_cache import cleanup_checkpoint_cache
from src.models.vit_b16 import DynamicTimmModelAdapter
from src.utils.download import download_and_extract


def test_timm_models_use_temporary_checkpoint_cache(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

    def fake_create_model(model_name: str, pretrained: bool, cache_dir: str):
        captured["model_name"] = model_name
        captured["pretrained"] = pretrained
        captured["cache_dir"] = cache_dir
        return FakeModel()

    monkeypatch.setattr(
        "src.models.vit_b16.list_pretrained_timm_models",
        lambda: {"vit_small_patch16_224"},
    )
    monkeypatch.setattr("src.models.vit_b16.timm.create_model", fake_create_model)

    DynamicTimmModelAdapter("vit_small_patch16_224")

    assert captured["model_name"] == "vit_small_patch16_224"
    assert captured["pretrained"] is True
    assert Path(captured["cache_dir"]).exists()

    cleanup_checkpoint_cache()
    assert not Path(captured["cache_dir"]).exists()


def test_timm_models_without_pretrained_weights_warn_and_use_random_init(
    monkeypatch, caplog
) -> None:
    captured = {}

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

    def fake_create_model(model_name: str, pretrained: bool, cache_dir: str):
        captured["model_name"] = model_name
        captured["pretrained"] = pretrained
        captured["cache_dir"] = cache_dir
        return FakeModel()

    monkeypatch.setattr("src.models.vit_b16.list_pretrained_timm_models", lambda: set())
    monkeypatch.setattr("src.models.vit_b16.timm.create_model", fake_create_model)

    DynamicTimmModelAdapter("vit_small_patch16_224")

    assert captured["model_name"] == "vit_small_patch16_224"
    assert captured["pretrained"] is False
    assert "loading random initialization" in caplog.text


def test_timm_transform_uses_loaded_model_config(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

    fake_model = FakeModel()

    def fake_create_model(model_name: str, pretrained: bool, cache_dir: str):
        captured["model_name"] = model_name
        captured["pretrained"] = pretrained
        captured["cache_dir"] = cache_dir
        return fake_model

    def fake_resolve_data_config(_config, model):
        captured["resolve_model_arg"] = model
        return {"input_size": (3, 384, 384)}

    def fake_create_transform(**kwargs):
        captured["transform_kwargs"] = kwargs
        return "fake-transform"

    monkeypatch.setattr(
        "src.models.vit_b16.list_pretrained_timm_models",
        lambda: {"vit_base_patch16_384.augreg_in1k"},
    )
    monkeypatch.setattr("src.models.vit_b16.timm.create_model", fake_create_model)
    monkeypatch.setattr(
        "src.models.vit_b16.timm.data.resolve_data_config",
        fake_resolve_data_config,
    )
    monkeypatch.setattr(
        "src.models.vit_b16.timm.data.create_transform",
        fake_create_transform,
    )

    model = DynamicTimmModelAdapter("vit_base_patch16_384.augreg_in1k")

    assert captured["resolve_model_arg"] is fake_model
    assert captured["transform_kwargs"] == {
        "input_size": (3, 384, 384),
        "is_training": False,
    }
    assert model.transform == "fake-transform"


def test_internal_timm_aliases_require_pretrained_weights(monkeypatch) -> None:
    monkeypatch.setattr("src.models.vit_b16.list_pretrained_timm_models", lambda: set())

    with pytest.raises(
        ValueError,
        match="requires pretrained timm weights",
    ):
        from src.models.vit_b16 import ViT_B_16

        ViT_B_16()


def test_download_and_extract_removes_zip_after_success(tmp_path, monkeypatch) -> None:
    archive_path = tmp_path / "dataset.zip"
    extracted_dir = tmp_path / "dataset"

    def fake_download(_url: str, output_path: str) -> str:
        Path(output_path).write_bytes(b"zip")
        return output_path

    class FakeZipFile:
        def __init__(self, path: str):
            assert path == str(archive_path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, target_path: str) -> None:
            (Path(target_path) / "dataset").mkdir()

    monkeypatch.setattr("src.utils.download.wget.download", fake_download)
    monkeypatch.setattr("src.utils.download.ZipFile", FakeZipFile)

    extracted_path = download_and_extract("dataset.zip", str(tmp_path))

    assert extracted_path == str(extracted_dir)
    assert extracted_dir.exists()
    assert not archive_path.exists()


def test_download_and_extract_removes_zip_after_extract_failure(
    tmp_path, monkeypatch
) -> None:
    archive_path = tmp_path / "broken.zip"

    def fake_download(_url: str, output_path: str) -> str:
        Path(output_path).write_bytes(b"zip")
        return output_path

    class BrokenZipFile:
        def __init__(self, _path: str):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extractall(self, _target_path: str) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr("src.utils.download.wget.download", fake_download)
    monkeypatch.setattr("src.utils.download.ZipFile", BrokenZipFile)

    with pytest.raises(RuntimeError, match="boom"):
        download_and_extract("broken.zip", str(tmp_path))

    assert not archive_path.exists()
