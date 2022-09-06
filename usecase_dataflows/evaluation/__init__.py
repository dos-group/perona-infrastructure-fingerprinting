from classes.processed_workload import ProcessedWorkloadModel
import pandas as pd
import numpy as np
import ast


def get_workload_cost_valid(workload: ProcessedWorkloadModel, rt_target: float):
    return workload.cost if workload.completed and workload.runtime <= rt_target else np.nan


def eval_objective(my_df: pd.DataFrame, target_col: str, ref_col: str, cond):
    my_df[target_col] = my_df[['runtime_target', 'profiled_workloads']].apply(cond, axis=1)
    my_df[target_col] = my_df[target_col] / my_df[ref_col]
    return my_df


def load_and_prepare(path_to_file: str):
    df = pd.read_csv(path_to_file)
    print(df.shape, path_to_file)

    if "selection_strategy" not in list(df.columns):
        df["selection_strategy"] = ""

    df["optimizer_strategy_sub"] = "NaiveBO" if "cherrypick" in path_to_file else "AugmentedBO"

    cols = ['optimizer_strategy', 'selection_strategy']
    df['identifier'] = df[cols].apply(lambda x: '-'.join([str(e) for e in x.tolist() if len(str(e))]), axis=1)

    df["profiled_workloads"] = df['profiled_workloads'].map(lambda s: [ProcessedWorkloadModel.parse_raw(o)
                                                                       for o in ast.literal_eval(s)])

    df = eval_objective(df, "best_cost_found", "best_cost",
                        lambda row: min([get_workload_cost_valid(w, row.to_list()[0]) for w in row.to_list()[1]]))
    df = eval_objective(df, "total_search_cost", "best_cost", lambda row: sum([w.cost for w in row.to_list()[1]]))
    df["total_search_time"] = df["profiled_workloads"].map(lambda pw: sum([e.runtime for e in pw]))
    df["total_search_cost"] = df["profiled_workloads"].map(lambda pw: sum([e.cost for e in pw]))
    df["profiling_counter_completed"] = df["profiled_workloads"].map(lambda pw: sum([e.completed for e in pw]))
    df["profiling_counter_not_completed_not_abandon"] = df["profiled_workloads"].map(
        lambda pw: sum([(not e.completed and not e.abandon) for e in pw]))

    if "num_tasks" in list(df.columns):
        df["num_tasks"] = df["num_tasks"].astype('int')
    else:
        df["num_tasks"] = None

    return df
