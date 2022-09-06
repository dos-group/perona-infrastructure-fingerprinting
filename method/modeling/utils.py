import logging
import os
from imp import reload


def create_dirs(path: str):
    try:
        os.makedirs(path)
    except BaseException as exc:
        logging.debug("Could not create dir", exc_info=exc)


def init_logging(logging_level: str):
    """Initializes logging with specific settings.
    Parameters
    ----------
    logging_level : str
        The desired logging level
    """

    reload(logging)
    logging.basicConfig(
        level=getattr(logging, logging_level.upper()),
        format="%(asctime)s [%(levelname)s] %(module)s.%(funcName)s](%(name)s)__[L%(lineno)d] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    logging.info(f"Successfully initialized logging.")
