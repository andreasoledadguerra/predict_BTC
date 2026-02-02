import logging
import os


def configure_logging():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )