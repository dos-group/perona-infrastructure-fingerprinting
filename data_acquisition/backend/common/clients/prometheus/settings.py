from typing import Optional
from pydantic import BaseSettings


class PrometheusSettings(BaseSettings):
    prometheus_endpoint: Optional[str] = "http://localhost:9090"
    prometheus_query_step_width: Optional[int] = 10  # in seconds
