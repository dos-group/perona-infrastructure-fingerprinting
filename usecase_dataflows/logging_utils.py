import logging
from importlib import reload


def init_logging(logging_level: str, save_file: str):
    """
    initialize logging setting
    """
    reload(logging)
    logging.basicConfig(
        level=getattr(logging, logging_level.upper()),
        format="[%(levelname)s %(asctime)s] %(module)s.%(funcName)s: %(message)s",
        datefmt="%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(save_file)
        ]
    )

    logging.info(f"Successfully initialized logging.")
