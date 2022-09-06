import asyncio
import kopf
import logging

# needed to load kopf operators for benchmark handling
from . import benchmark_handlers
# needed to load kopf operators for chaos handling
from . import stress_handlers


@kopf.on.startup()
async def startup(logger, **_):
    file_handler = logging.FileHandler("/tmp/app.log")

    loggers = [logging.getLogger()] + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.addHandler(file_handler)

    logger.info('Starting Operator in 1s...')
    await asyncio.sleep(1)


@kopf.on.login()
async def login(logger, **_):
    return kopf.login_via_pykube(logger=logger)
