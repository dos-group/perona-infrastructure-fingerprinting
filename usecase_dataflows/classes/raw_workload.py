from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field, validator


class DataRecordingModel(BaseModel):
    time: datetime
    value: float


class NodeMetricsModel(BaseModel):
    node_name: str
    arrow_metrics: List[float]
    memory_used: List[DataRecordingModel] = Field(default=[])
    cpu_busy: List[DataRecordingModel] = Field(default=[])
    disk_io_util: List[DataRecordingModel] = Field(default=[])
    network_util: List[DataRecordingModel] = Field(default=[])

    @staticmethod
    def cap(value: Union[int, float], replace_zero_with: Optional[float] = None):
        new_value = min(100, max(0, value))
        if new_value == 0 and replace_zero_with is not None:
            new_value = replace_zero_with
        return new_value

    @validator("memory_used")
    def normalize_memory_used(cls, memory_used):
        return [DataRecordingModel(time=el.time, value=cls.cap(el.value)) for el in memory_used]

    @validator("cpu_busy")
    def normalize_cpu_busy(cls, cpu_busy):
        return [DataRecordingModel(time=el.time, value=cls.cap(el.value)) for el in cpu_busy]

    @validator("disk_io_util")
    def normalize_disk_io_util(cls, disk_io_util):
        return [DataRecordingModel(time=el.time, value=cls.cap(el.value)) for el in disk_io_util]

    @validator("network_util")
    def normalize_network_util(cls, network_util):
        return [DataRecordingModel(time=el.time, value=cls.cap(el.value)) for el in network_util]


class BaseInformationModel(BaseModel):
    framework_name: str
    algorithm_name: str
    dataset_name: str


class GenericWorkloadModel(BaseInformationModel):
    workload_name: str

    node_count: int
    machine_name: str
    runtime: float
    cost: Optional[float]
    completed: bool
    timeout: bool
    abandon: bool


class RawWorkloadModel(GenericWorkloadModel):
    node_metrics: Dict[str, NodeMetricsModel]
    configuration: Dict[str, str]
    result: Dict[str, str]
