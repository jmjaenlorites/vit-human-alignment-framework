from datetime import datetime
from pathlib import Path
import logging

from src.runner.base import Runner

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
        runner = Runner(csv_path="data/vit-human-alignment-framework-test.csv")
        runner.execute()
    except Exception:
        logger.exception("Runner failed")
        raise


if __name__ == "__main__":
    main()
