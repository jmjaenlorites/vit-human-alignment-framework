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

    monkeypatch.setattr("src.models.vit_b16.timm.create_model", fake_create_model)

    DynamicTimmModelAdapter("vit_small_patch16_224")

    assert captured["model_name"] == "vit_small_patch16_224"
    assert captured["pretrained"] is True
    assert Path(captured["cache_dir"]).exists()

    cleanup_checkpoint_cache()
    assert not Path(captured["cache_dir"]).exists()


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
