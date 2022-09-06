from ax.modelbridge.generation_strategy import GenerationStrategy

from classes.workload_dataset import WorkloadTask
from optimization.methods.optimizer_ext import OptimizerExt
from optimization.methods.embedder import CherryPickEmbedderBO
from optimization.methods.evaluator import EvaluatorBO
from optimization.methods.optimizer import OptimizerBO
from optimization.methods.optimizer_helpers import setup_soo_experiment, create_sobol_generation_step, \
    create_cherrypick_generation_step


class CherryPickOptimizer(OptimizerBO):
    exp_name: str = "cherrypick_experiment"

    def __init__(self, task: WorkloadTask, rt_target: float, num_profilings: int, **kwargs):
        super().__init__(task, rt_target, num_profilings, **kwargs)

        # vectorizer and evaluator
        self.embedder: CherryPickEmbedderBO = CherryPickEmbedderBO(task.workloads)
        self.evaluator: EvaluatorBO = EvaluatorBO(rt_target, task.workloads, self.embedder)
        # setup experiment, get initial starting points
        gen_stra: GenerationStrategy = GenerationStrategy(
            steps=[
                # quasi-random generation of initial points
                create_sobol_generation_step(self.num_init, self.seed),
                # naive BO
                create_cherrypick_generation_step(self)
            ]
        )

        self.ax_client = setup_soo_experiment(self.exp_name, gen_stra, self.embedder, self.evaluator, task)
        self.init()


class CherryPickExtOptimizer(OptimizerExt, CherryPickOptimizer):
    exp_name: str = "cherrypick_ext_experiment"



