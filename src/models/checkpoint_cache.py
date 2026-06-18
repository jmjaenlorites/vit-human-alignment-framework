"""Gestiona un cache temporal para checkpoints descargados."""

import atexit
from tempfile import TemporaryDirectory


_checkpoint_cache_dir: TemporaryDirectory[str] | None = None
_cleanup_registered = False


def get_checkpoint_cache_dir() -> str:
    """Retorna el directorio temporal compartido para checkpoints."""
    global _checkpoint_cache_dir, _cleanup_registered

    if _checkpoint_cache_dir is None:
        _checkpoint_cache_dir = TemporaryDirectory(
            prefix="vit-human-alignment-framework-checkpoints-"
        )

    if not _cleanup_registered:
        atexit.register(cleanup_checkpoint_cache)
        _cleanup_registered = True

    return _checkpoint_cache_dir.name


def cleanup_checkpoint_cache() -> None:
    """Elimina el cache temporal de checkpoints si existe."""
    global _checkpoint_cache_dir

    if _checkpoint_cache_dir is None:
        return

    _checkpoint_cache_dir.cleanup()
    _checkpoint_cache_dir = None
