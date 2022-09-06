import os
import warnings
from typing import List, Optional, Tuple
from pydantic import BaseModel
import logging

# silence warnings, e.g. from sklearn
warnings.filterwarnings("ignore")

import torch
import torch.multiprocessing as torch_mp
import concurrent.futures

from optimization.methods.optimizer import OptimizerBO

torch.set_default_dtype(torch.float64)

from classes.workload_dataset import WorkloadDataset
from logging_utils import init_logging


class DatasetConfig(BaseModel):
    max_parallel: int = 8
    verbose: bool = True
    overwrite: bool = False


class ExperimentConfig(BaseModel):
    percentiles: List[int] = [25, 50, 75]
    iterations: List[int] = list(range(1, 6))
    num_profilings: int = 10
    num_initial_random_points: int = 3


class OptimizationConfig(BaseModel):
    model_kwargs: dict = {
        "fit_out_of_design": True,
        "torch_dtype": torch.double,
        "torch_device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }
    acquisition_options: dict = {
        "cache_root": False
    }


class Experiment:
    def __init__(self, scope: str,
                 dataset_config: DatasetConfig = DatasetConfig(),
                 experiment_config: ExperimentConfig = ExperimentConfig(),
                 optimization_config: OptimizationConfig = OptimizationConfig()):

        self.dataset: WorkloadDataset = WorkloadDataset.create_scout_dataset(scope, **dataset_config.dict())
        self.experiment_name: Optional[str] = None
        self.dataset_config: DatasetConfig = dataset_config
        self.experiment_config: ExperimentConfig = experiment_config
        self.optimizer_config: OptimizationConfig = optimization_config
        self.save_file: Optional[str] = None
        self.optimization_classes: List[Tuple[OptimizerBO, str]] = []
        self.existing_tuples: List[tuple] = []

    def set_experiment_name(self, exp_name: str):
        self.experiment_name = exp_name
        return self
    
    def set_optimization_classes(self, optimization_classes: List[Tuple[OptimizerBO, str]]):
        self.optimization_classes = optimization_classes
        return self

    def run(self, log_level: str = "INFO", suffix: str = ""):
        root_dir: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", suffix)

        try:
            os.makedirs(root_dir)
        except:
            pass

        log_file: str = os.path.join(root_dir, f"{self.experiment_name}.log")
        self.save_file: str = os.path.join(root_dir, f"{self.experiment_name}.csv")
        self.existing_tuples: List[tuple] = self._extract_tuples()
        logging.warning(f"Number of existing tuples: {len(self.existing_tuples)}")
        
        init_logging(log_level, log_file)
        with torch_mp.Manager() as man:
            lock = man.Lock()
            self._run(lock)
    
    def _extract_tuples(self, *args, **kwargs) -> List[tuple]:
        raise NotImplementedError
    
    def _run(self, lock):
        raise NotImplementedError
    
    @staticmethod
    def pool_init():
        torch.set_num_threads(1)
        torch.manual_seed(os.getpid())
    
    @staticmethod
    def run_optimizer(args):
        return concurrent.futures.ThreadPoolExecutor().submit(Experiment.run_optimizer_internal, args).result()
    
    @staticmethod
    def run_optimizer_internal(args):
        lock, opt_class, opt_args, opt_kwargs = args
        try:
            optimizer = opt_class(*opt_args, **opt_kwargs)
            optimizer.run(lock)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        return None

