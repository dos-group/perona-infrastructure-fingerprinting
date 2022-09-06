from __future__ import annotations

from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel

from classes.processed_workload import ProcessedWorkloadModel
from classes.raw_workload import BaseInformationModel
from preparation.loader import load_data
from preparation.loader_scout import _resolve_costs


class WorkloadTask(BaseInformationModel):
    workloads: List[ProcessedWorkloadModel]
    is_complete: bool
    percentile: Optional[int] = None
    iteration: Optional[int] = None
    runtime_target: Optional[float] = None

    @property
    def identifier(self):
        return f"task(framework={self.framework_name}, algorithm={self.algorithm_name}, dataset={self.dataset_name})"

    @classmethod
    def create(cls, workloads: List[ProcessedWorkloadModel], percentile: Optional[int] = None,
               iteration: Optional[int] = None, runtime_target: Optional[float] = None):
        return cls(workloads=workloads,
                   framework_name=workloads[0].framework_name,
                   algorithm_name=workloads[0].algorithm_name,
                   dataset_name=workloads[0].dataset_name,
                   is_complete=len(workloads) == 9 or len(workloads) == 69,
                   percentile=percentile,
                   iteration=iteration,
                   runtime_target=runtime_target)

    @staticmethod
    def lower_cost_bound(runtime: float, *args, **kwargs):
        cost: float = _resolve_costs("R4XXL", 48, runtime)
        return ProcessedWorkloadModel.construct(cost=cost, runtime=runtime)


class WorkloadDataset(BaseModel):
    workloads: List[ProcessedWorkloadModel]
    workload_tasks: Dict[Tuple[str, str, str], WorkloadTask]

    @classmethod
    def create_scout_dataset(cls, scope: str, **kwargs):
        all_workloads: List[ProcessedWorkloadModel] = load_data(scope, **kwargs)

        def filter_dataset(f_name: str, alg_name: str, data_name: str):
            return [w for w in all_workloads
                    if w.framework_name == f_name
                    and w.algorithm_name == alg_name
                    and w.dataset_name == data_name]

        workload_tasks: Dict[Tuple[str, str, str], WorkloadTask] = {}
        for tup in list(set([(w.framework_name, w.algorithm_name, w.dataset_name) for w in all_workloads])):
            filtered_workloads = filter_dataset(*tup)
            if len(filtered_workloads):
                task = WorkloadTask.create(filtered_workloads)
                workload_tasks[tup] = task

        remaining_workloads = sum([t.workloads for t in workload_tasks.values()], [])
        print("all workloads:", len(remaining_workloads))
        all_machines: List[str] = list(set([w.machine_name for w in remaining_workloads]))
        print("all machines:", len(all_machines))
        print("distinct workloads:", len(workload_tasks))

        return cls(workload_tasks=workload_tasks,
                   workloads=remaining_workloads)
