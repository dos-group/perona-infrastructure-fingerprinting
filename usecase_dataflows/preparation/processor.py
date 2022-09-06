from typing import List, Dict
from collections import OrderedDict
import numpy as np
from classes.processed_workload import DataStatisticsModel, ProcessedWorkloadModel
from classes.raw_workload import RawWorkloadModel


def _process_metrics(workload: RawWorkloadModel) -> dict:
    metrics_dict: Dict[str, List[float]] = OrderedDict()
    arrow_metrics: List[List[float]] = []
    for node_name, node_metrics in workload.node_metrics.items():
        for metric_name, metric_values in node_metrics.dict().items():
            if isinstance(metric_values, list):
                if metric_name == "arrow_metrics":
                    arrow_metrics.append(metric_values)
                else:
                    metrics_dict[metric_name] = metrics_dict.get(metric_name, []) + [e["value"] for e in metric_values]

    statistics_dict: Dict[str, DataStatisticsModel] = {key: DataStatisticsModel.construct_statistics(values)
                                                       for key, values in metrics_dict.items()}

    return {
        "metrics": statistics_dict,
        "arrow_metrics": np.mean(np.array(arrow_metrics), axis=0).reshape(-1).tolist()
    }


def process_workload(workload: RawWorkloadModel) -> ProcessedWorkloadModel:
    return ProcessedWorkloadModel(**{
        **workload.dict(),
        **_process_metrics(workload),
    })
