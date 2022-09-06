import copy

from ax.modelbridge.generation_strategy import GenerationStrategy

from classes.processed_workload import ProcessedWorkloadModel
from classes.workload_dataset import WorkloadTask
from optimization.methods.optimizer_ext import OptimizerExt
from optimization.methods.embedder import ArrowEmbedderBO
from optimization.methods.evaluator import EvaluatorBO
from optimization.methods.optimizer import OptimizerBO
from optimization.methods.optimizer_helpers import setup_soo_experiment, create_sobol_generation_step, \
    create_arrow_generation_step
from preparation.loader_scout import machine_name_map


class ArrowOptimizer(OptimizerBO):
    exp_name: str = "arrow_experiment"

    def __init__(self, task: WorkloadTask, rt_target: float, num_profilings: int, **kwargs):
        super().__init__(task, rt_target, num_profilings, **kwargs)

        # vectorizer and evaluator
        self.embedder: ArrowEmbedderBO = ArrowEmbedderBO(task.workloads)
        self.evaluator: EvaluatorBO = EvaluatorBO(rt_target, task.workloads, self.embedder)
        # setup experiment, get initial starting points
        gen_stra: GenerationStrategy = GenerationStrategy(
            steps=[
                # quasi-random generation of initial points
                create_sobol_generation_step(self.num_init, self.seed),
                # augmented BO
                create_arrow_generation_step(self)
            ]
        )

        self.ax_client = setup_soo_experiment(self.exp_name, gen_stra, self.embedder, self.evaluator, task)
        self.init()


class ArrowExtOptimizer(OptimizerExt, ArrowOptimizer):
    exp_name: str = "arrow_ext_experiment"

    def modify_workload(self, workload: ProcessedWorkloadModel):
        resolved_machine_name: str = next((k for k, v in machine_name_map.items() if v == workload.machine_name))

        workload_copy: ProcessedWorkloadModel = copy.deepcopy(workload)
        new_metrics_dict = ArrowExtOptimizer.load_and_prepare()
        workload_copy.arrow_metrics = list(new_metrics_dict[resolved_machine_name].values())
        assert workload.arrow_metrics != workload_copy.arrow_metrics

        return workload_copy
