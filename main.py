import os

# Configuración de JAX/XLA. Estas variables DEBEN estar antes de CUALQUIER
# import de JAX (directo o transitivo).
#
# - `XLA_PYTHON_CLIENT_PREALLOCATE=false`: evita que JAX agarre el 75% de la
#   GPU al inicializar. Necesario cuando torch comparte la GPU.
# - `XLA_PYTHON_CLIENT_ALLOCATOR=platform`: usa `cudaMalloc` directo en lugar
#   del BFC. Sustituye el riesgo de fragmentación por un coste de ~10-30% en
#   throughput JAX. Necesario en runs largos con varios modelos JAX seguidos:
#   el BFC nunca devuelve chunks al driver, así que tras varias compilaciones
#   distintas el heap acaba fragmentado y allocaciones grandes (workspaces de
#   compile) revientan aunque haya VRAM libre. No-op en CPU/TPU.
#
# Ambas se pueden sobrescribir desde la shell si el usuario quiere otro modo.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

from src.runner.base import Runner  # noqa: E402

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    logging.getLogger(__name__).info("Logging to %s", log_path)


def main() -> None:
    setup_logging()
    try:
        runner = Runner(
            json_path="data/experiments.timm.model-battery.json",
            results_path="results-model-battery.csv",
        )
        runner.execute()
    except Exception:
        logger.exception("Runner failed")
        raise


if __name__ == "__main__":
    main()
