from typing import List

from pydantic import BaseModel


class ExperimentModel(BaseModel):
    bm_types: List[str]
    node_ids: List[str]
    num_each: int
    random_s: int
    num_anom: int = 0

    class Config:
        schema_extra = {
            "example": {
                "bm_types": ["cpu-sysbench", "memory-sysbench", "disk-fio",
                             "disk-ioping", "network-iperf3", "network-qperf"],
                "node_ids": ["benchmark-operator-worker2@@@benchmark-operator-worker"],
                "num_each": 5,
                "random_s": 42,
                "num_anom": 0
            }
        }
