import copy
import json
import math
import os
import time
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool
from typing import List, Dict, Callable, Optional, Tuple
import tqdm
import numpy as np

from classes.processed_workload import ProcessedWorkloadModel
from classes.raw_workload import DataRecordingModel, RawWorkloadModel, NodeMetricsModel
from classes.scout_workload import ScoutConfiguration, ScoutResult, ScoutName
from preparation.processor import process_workload

root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

machine_name_map: Dict[str, str] = {
    "c4.large": "C4L",
    "c4.xlarge": "C4XL",
    "c4.2xlarge": "C4XXL",
    "m4.large": "M4L",
    "m4.xlarge": "M4XL",
    "m4.2xlarge": "M4XXL",
    "r4.large": "R4L",
    "r4.xlarge": "R4XL",
    "r4.2xlarge": "R4XXL"
}


def _resolve_costs(name: str, num: int, runtime: float):
    # prices according to https://calculator.aws/#/createCalculator/EC2 (10.11.2021)
    hourly_rates: Dict[str, float] = {
        "C4L": 0.1, "C4XL": 0.199, "C4XXL": 0.398,
        "M4L": 0.1, "M4XL": 0.2, "M4XXL": 0.4,
        "R4L": 0.133, "R4XL": 0.266, "R4XXL": 0.532
    }
    # onDemand costs
    return num * hourly_rates[name] * (runtime / 3600)


def _extract_node_metrics(node_name: str, path_to_csv: str, workload_name: str,
                          time_info: Optional[Tuple[Optional[datetime], Optional[float]]]) -> NodeMetricsModel:
    def _get_df_slice(target_df: pd.DataFrame, start: datetime, end: datetime):
        sub_target_df: pd.DataFrame = target_df[(target_df.timestamp >= start) & (target_df.timestamp <= end)]
        if len(sub_target_df) <= 5:
            return sub_target_df, (len(sub_target_df), len(sub_target_df))
        # interpolate possibly missing values
        w_start = sub_target_df.timestamp.iloc[0]
        w_start -= timedelta(seconds=(1 + math.ceil((w_start - start).total_seconds() / 5)) * 5)
        w_end = sub_target_df.timestamp.iloc[-1]
        w_end += timedelta(seconds=(1 + math.ceil((end - w_end).total_seconds() / 5)) * 5)
        mask_range = [w_start + timedelta(seconds=5 * i) for i in range(int((w_end - w_start).total_seconds() / 5))]
        mask_range = [d for d in mask_range if start <= d <= end]

        idx_range = [w_start + timedelta(seconds=i) for i in range(int((w_end - w_start).total_seconds()))]
        idx_range = [d for d in idx_range if start <= d <= end]

        res_df = pd.DataFrame(data=None, columns=list(sub_target_df.columns), dtype=float, index=idx_range)
        res_df.loc[copy.deepcopy(sub_target_df.timestamp), :] = sub_target_df[:].values
        res_df = res_df.ffill(axis=0).bfill(axis=0)
        # only select using the mask
        res_df = res_df.loc[mask_range, :]
        res_df["timestamp"] = res_df.index
        res_df.reset_index(inplace=True, drop=True)
        return res_df, (len(sub_target_df), len(res_df))

    date_parse_func: Callable = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    df = pd.read_csv(path_to_csv, usecols=["timestamp", "cpu.%idle", "memory.%memused", "disk.%util", "network.%ifutil",
                                           # some arrow metrics
                                           "cpu.%usr", "cpu.%iowait", "load.plist-sz", "memory.%commit", "disk.await"],
                     parse_dates=['timestamp'], date_parser=date_parse_func)

    if time_info is not None:
        ref_time_date, offset = time_info
        if ref_time_date is not None and offset is not None:
            # try forward slicing first
            end_time: datetime = ref_time_date + timedelta(seconds=offset)
            sub_df, shape_change = _get_df_slice(df, ref_time_date, end_time)

            if not (math.floor(offset / 5) - 1 <= len(sub_df) <= math.ceil(offset / 5)):
                # second try: try backward slicing
                start_time: datetime = ref_time_date - timedelta(seconds=offset)
                sub_df, shape_change = _get_df_slice(df, start_time, ref_time_date)

                if not (math.floor(offset / 5) - 1 <= len(sub_df) <= math.ceil(offset / 5)):
                    print(f"{workload_name}, {node_name}, backwards, {shape_change}, {offset / 5}")
            df = sub_df

    def _df_extract(target_df: pd.DataFrame, func: Callable):
        return [DataRecordingModel(time=t, value=v) for t, v in zip(target_df["timestamp"].values, func(target_df))]

    return NodeMetricsModel(**{
        "node_name": node_name.split(".csv")[0],
        "cpu_busy": _df_extract(df, lambda dff: 100 - dff["cpu.%idle"].values),
        "memory_used": _df_extract(df, lambda dff: dff["memory.%memused"].values),
        "disk_io_util": _df_extract(df, lambda dff: dff["disk.%util"].values),
        "network_util": _df_extract(df, lambda dff: dff["network.%ifutil"].values),
        "arrow_metrics": [
            df["cpu.%usr"].mean(),  # CPU utilization on user time
            df["cpu.%iowait"].mean(),  # I/O wait time
            df["load.plist-sz"].mean(),  # number of tasks in the task list
            df["memory.%commit"].mean(),  # % of commits in memory
            df["disk.%util"].mean(),  # disk utilization
            df["disk.await"].mean()  # wait time (disk).
        ]
    })


def _load_raw_workload(name_model: ScoutName, subdir_path: str) -> Optional[RawWorkloadModel]:
    result_dict: dict = {**name_model.dict(), "node_metrics": {}}

    time_info: Optional[Tuple[Optional[datetime], Optional[float]]] = None
    for file_name in sorted(list(os.listdir(subdir_path))):
        path_to_file: str = os.path.join(subdir_path, file_name)
        if file_name == "report.json":
            with open(path_to_file) as f:
                data_dict = json.load(f)
                scout_config, scout_result = ScoutConfiguration(**data_dict), ScoutResult(**data_dict)

                time_info = (scout_result.timestamp, scout_result.elapsed_time)
                result_dict["configuration"] = scout_config.dict()
                result_dict["result"] = scout_result.dict(exclude={'timestamp'})
                result_dict["runtime"] = scout_result.elapsed_time
                result_dict["completed"] = scout_result.completed
                result_dict["timeout"] = scout_result.elapsed_time >= 7200
                result_dict["abandon"] = not result_dict["completed"] and not result_dict["timeout"]
        else:
            result_dict["node_metrics"][file_name.split(".csv")[0]] = _extract_node_metrics(file_name, path_to_file,
                                                                                            name_model.workload_name,
                                                                                            time_info)
    workload: RawWorkloadModel = RawWorkloadModel(**result_dict)
    return workload


def _load_processed_workload(tuple_list: List[Tuple[ScoutName, str]]) -> Optional[ProcessedWorkloadModel]:
    raw_workloads: List[RawWorkloadModel] = []
    for tup in tuple_list:
        raw_workload: Optional[RawWorkloadModel] = _load_raw_workload(*tup)
        if raw_workload is not None:
            raw_workloads.append(raw_workload)

    re_workload: Optional[RawWorkloadModel]
    if len(raw_workloads) == 1:
        re_workload = raw_workloads[0]
    else:
        re_workload = None

    if re_workload is not None:
        re_workload.cost = _resolve_costs(re_workload.machine_name, re_workload.node_count, re_workload.runtime)
        return process_workload(re_workload)
    else:
        return None


def load_scout(dataset_name: str, **kwargs) -> List[ProcessedWorkloadModel]:
    verbose: bool = kwargs.get("verbose", False)
    max_parallel: int = kwargs.get("max_parallel", None)
    target_dir: str = os.path.join(root_dir, "data", dataset_name)

    workload_name_dict: Dict[str, List[Tuple[ScoutName, str]]] = {}
    for subdir_name in list(os.listdir(target_dir)):
        if subdir_name.startswith('.'):  # skip hidden files
            continue
        name_model: ScoutName = ScoutName.parse_directory_name(subdir_name, machine_name_map)
        if name_model.machine_name not in list(machine_name_map.values()):
            continue
        workload_name_dict[name_model.workload_name] = workload_name_dict.get(name_model.workload_name, []) + \
                                                       [(name_model, os.path.join(target_dir, subdir_name))]

    args_list: List[List[Tuple[ScoutName, str]]] = list(workload_name_dict.values())
    if verbose:
        print("#" * 5, f"[{dataset_name}] Number of subdirs: {len(args_list)}", "#" * 5)

    workloads: List[ProcessedWorkloadModel]
    with Pool(processes=max_parallel) as p:
        start = time.time()
        workloads = list(tqdm.tqdm(p.imap_unordered(_load_processed_workload, args_list), total=len(args_list)))
        workloads = [w for w in workloads if w is not None]
        print(f"Time it took to load data (#workloads={len(workloads)}): {time.time() - start:.2f}s")

    return workloads
