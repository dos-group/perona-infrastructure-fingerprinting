from typing import Optional, Dict
from datetime import datetime
from pydantic import BaseModel


class ScoutResult(BaseModel):
    completed: Optional[bool]
    elapsed_time: Optional[float]
    timestamp: Optional[datetime]


class ScoutConfiguration(BaseModel):
    datasize: Optional[str]
    framework: Optional[str]
    input_size: Optional[float]
    program: Optional[str]
    workload: Optional[str]


class ScoutName(BaseModel):
    workload_name: str
    node_count: int
    machine_name: str
    framework_name: str
    algorithm_name: str
    dataset_name: str

    @classmethod
    def parse_directory_name(cls, subdir_name: str, machine_name_map: Dict[str, str]):
        node_count, machine_name, algorithm_name, framework_name, dataset_name, _ = subdir_name.split("_")
        machine_name = machine_name_map.get(machine_name, "None")

        return cls(
            workload_name="_".join([str(node_count), machine_name, algorithm_name, framework_name, dataset_name]),
            node_count=node_count,
            machine_name=machine_name,
            algorithm_name=algorithm_name,
            framework_name=framework_name,
            dataset_name=dataset_name
        )
