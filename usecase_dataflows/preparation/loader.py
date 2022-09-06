import os
import pickle
from typing import List

from classes.processed_workload import ProcessedWorkloadModel
from preparation.loader_scout import load_scout

root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data(dataset_name: str, **kwargs) -> List[ProcessedWorkloadModel]:
    workloads: List[ProcessedWorkloadModel]

    if os.path.exists(os.path.join(root_dir, "data", f"{dataset_name}_raw.p")) and not kwargs.get("overwrite", False):
        with open(os.path.join(root_dir, "data", f"{dataset_name}_raw.p"), "rb") as f:
            workloads = pickle.load(f)
        return workloads
    elif dataset_name == "scout_multiple":
        workloads = load_scout(dataset_name, **kwargs)
    else:
        raise ValueError("Other dataset-names are not allowed")

    # save processed workloads
    with open(os.path.join(root_dir, "data", f"{dataset_name}_raw.p"), "wb") as f:
        pickle.dump(workloads, f)

    return workloads
