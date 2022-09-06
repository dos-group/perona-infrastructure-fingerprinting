from functools import lru_cache

from .client import PrometheusClient
from .schemes import *


@lru_cache()
def get_prometheus_client():
    return PrometheusClient()
