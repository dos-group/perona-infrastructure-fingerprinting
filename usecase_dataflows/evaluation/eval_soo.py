from typing import List
import pandas as pd
import torch.multiprocessing as torch_mp
import os
import torch

import numpy as np

from evaluation.experiment import Experiment, ExperimentConfig
from optimization.methods.baselines.optimizer_arrow import ArrowOptimizer, ArrowExtOptimizer
from optimization.methods.baselines.optimizer_cherrypick import CherryPickOptimizer, CherryPickExtOptimizer


class SooExperiment(Experiment):
    def _extract_tuples(self, *args, **kwargs) -> List[tuple]:
        if os.path.exists(self.save_file):
            cols = ["percentile", "iteration", "framework_name",
                    "algorithm_name", "dataset_name", "optimizer_strategy", "optimizer_strategy_sub"]
            temp_df: pd.DataFrame = pd.read_csv(self.save_file, usecols=cols)
            temp_df = temp_df[cols]
            return list(set([tuple(my_list) for my_list in temp_df.values.tolist()]))
        else:
            return []

    def _run(self, lock):
        for _, task_obj in sorted(self.dataset.workload_tasks.items(), key=lambda tup: tup[0]):
            seed_arr = np.arange(100 * 20).reshape(100, 20).astype(int)
            for iteration in self.experiment_config.iterations:
                opt_specs_list = []
                for percentile in self.experiment_config.percentiles:
                    for optimizer_class, optimizer_class_sub in self.optimization_classes:

                        if (percentile, iteration, task_obj.framework_name,
                            task_obj.algorithm_name, task_obj.dataset_name,
                            optimizer_class.__name__, optimizer_class_sub) in self.existing_tuples:
                            continue

                        seed_val: int = int(seed_arr[percentile, iteration] * 10) + 10

                        completed_workloads = [w for w in task_obj.workloads if w.completed]

                        rt_target: float = np.percentile([w.runtime for w in completed_workloads], percentile)
                            
                        workloads_valid = [w for w in completed_workloads if w.runtime <= rt_target]
                        best_cost: float = min([w.cost for w in completed_workloads if w.runtime <= rt_target])

                        exp_config: dict = {
                            "exp_config": self.optimizer_config.dict(),
                            "seed": seed_val,
                            "num_objectives": 1,
                            "num_init": 1 if optimizer_class_sub.endswith("Ext") else self.experiment_config.num_initial_random_points,
                            "save_file": self.save_file,
                            "base_entry": {
                                "#all_candidates": len(task_obj.workloads),
                                "#valid_candidates": len(workloads_valid),
                                "framework_name": task_obj.framework_name,
                                "algorithm_name": task_obj.algorithm_name,
                                "dataset_name": task_obj.dataset_name,
                                "percentile": percentile,
                                "iteration": iteration,
                                "runtime_target": rt_target,
                                "best_cost": best_cost,
                                "optimizer_strategy": optimizer_class.__name__,
                                "optimizer_strategy_sub": optimizer_class_sub
                            }}

                        opt_specs_list.append((
                            optimizer_class,
                            (task_obj, rt_target, self.experiment_config.num_profilings),
                            exp_config
                        ))
                        
                if len(opt_specs_list):
                    torch.set_num_threads(1)
                    with torch_mp.Pool(self.dataset_config.max_parallel, initializer=self.pool_init) as pool:
                        pool.map(self.run_optimizer, [(lock, ) + opt_specs for opt_specs in opt_specs_list])


if __name__ == '__main__':
    torch_mp.set_sharing_strategy('file_system')
    torch_mp.set_start_method('spawn')

    for opt_class, opt_name in [
        (CherryPickExtOptimizer, "CherryPickExt"),
        (ArrowExtOptimizer, "ArrowExt"),
        (CherryPickOptimizer, "CherryPick"),
        (ArrowOptimizer, "Arrow")
    ]:
        multiple_experiment: SooExperiment = SooExperiment("scout_multiple",
                                                           experiment_config=ExperimentConfig(
                                                               percentiles=[10, 30, 50, 70, 90],
                                                               iterations=[1, 3, 5, 7, 9],
                                                               num_profilings=10)) \
            .set_experiment_name(f"multiple_soo_{opt_name.lower()}") \
            .set_optimization_classes([(opt_class, opt_name)])
        multiple_experiment.run("INFO", f"RQ0_{opt_name.lower()}")
