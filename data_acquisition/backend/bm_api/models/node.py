from __future__ import annotations

from typing import Optional

from pydantic import BaseModel
from common import NodeMetricsModel as PrometheusNodeMetricsModel


class NodeLimitations(BaseModel):
    storage_mb: Optional[int]
    memory_mb: Optional[int]
    num_pods: Optional[int]  

# based on our prometheus node metrics model
class NodeMetricsModel(PrometheusNodeMetricsModel):
    pass
