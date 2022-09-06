from typing import List, Optional
from pydantic import BaseModel
import datetime


class BenchmarkResultMetric(BaseModel):
    name: str
    value: Optional[float]
    unit: str


class BenchmarkResult(BaseModel):
    id: str
    type: str
    resource: str
    started: datetime.datetime

    metrics: List[BenchmarkResultMetric]  # Dict[str, Optional[str]]
