import os
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Union, Optional, Tuple
from sklearn.preprocessing import maxabs_scale

import numpy as np
import pandas as pd

from classes.processed_workload import ProcessedWorkloadModel
from optimization.methods.optimizer import OptimizerBO
from preparation.loader_scout import machine_name_map


class OptimizerExt:
    artifact_name: str = "dataflow_scores_from_embeddings.csv"

    @staticmethod
    @lru_cache()
    def get_path_to_artifact_by_name(artifact_name: str):
        artifact_dir = next((parent for parent in Path(__file__).parents if parent.name == "usecase_dataflows"))
        return os.path.join(artifact_dir.absolute(), "artifacts", artifact_name)

    @staticmethod
    @lru_cache()
    def load_and_prepare() -> Dict[str, Dict[str, float]]:
        df: pd.DataFrame = pd.read_csv(OptimizerExt.get_path_to_artifact_by_name(OptimizerExt.artifact_name))
        df.loc[:, ~df.columns.isin(["machine_name"])] = maxabs_scale(df.loc[:, ~df.columns.isin(["machine_name"])])
        cols: List[str] = sorted(list(df.columns))
        id_col: str = cols.pop(cols.index("machine_name"))
        new_metrics_dict: Dict[str, Dict[str, float]] = OrderedDict()
        for _, row in df.iterrows():
            for col in cols:
                new_metrics_dict[row[id_col]] = new_metrics_dict.get(row[id_col], OrderedDict())
                new_metrics_dict[row[id_col]][col] = row[col]
        return new_metrics_dict

    @staticmethod
    def weight_acqf_values(acqf_values: List[float], profiled_workloads: List[ProcessedWorkloadModel],
                           candidates: List[ProcessedWorkloadModel]):
        new_metrics_dict = OptimizerExt.load_and_prepare()
        
        acqf_values = (np.array(acqf_values) + 0.01).tolist()

        stats_dict: Dict[str, Union[float, List[Tuple[float, float]]]] = {}
        for w in profiled_workloads:
            for name, stats_model in w.metrics.items():
                exp = 2 ** (np.log2(w.node_count) - np.log2(4))
                stats_dict[name] = stats_dict.get(name, []) + [(1 / exp, (stats_model.mean / 100) ** exp)]
        for k, v_list in stats_dict.items():
            weights, values = zip(*v_list)
            weights_norm = list(weights) / sum(list(weights))
            stats_dict[k] = float(np.sum(np.array(weights_norm) * np.array(values)))
        
        new_acqf_values: List[float] = []
        for curr_acqf_value, w in zip(acqf_values, candidates):
            resolved_machine_name: str = next((k for k, v in machine_name_map.items() if v == w.machine_name))
            scores: Dict[str, float] = new_metrics_dict[resolved_machine_name]
            new_acqf_values.append(curr_acqf_value * (1 + (sum([
                stats_dict["memory_used"] * scores["memory-sysbench"],
                stats_dict["cpu_busy"] * scores["cpu-sysbench"],
                stats_dict["disk_io_util"] * scores["disk-fio"],
                stats_dict["disk_io_util"] * scores["disk-ioping"],
                stats_dict["network_util"] * scores["network-qperf"],
                stats_dict["network_util"] * scores["network-iperf3"]
            ]) / len(list(scores.keys())) ** max(1, len(profiled_workloads) - 1))))
        assert acqf_values != new_acqf_values

        return new_acqf_values

    def get_educated_guess(self, profiled_workloads: List[ProcessedWorkloadModel],
                           candidates: List[ProcessedWorkloadModel]) -> Optional[ProcessedWorkloadModel]:
        if len([w for w in profiled_workloads if not w.abandon]):
            acqf_values: List[float] = OptimizerExt.weight_acqf_values([1] * len(candidates), profiled_workloads,
                                                                       candidates)
            _, max_index = OptimizerBO.retrieve_candidate(acqf_values)
            return candidates[max_index]
