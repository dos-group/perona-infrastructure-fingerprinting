import copy
import logging
import time
from typing import List, Optional
import pandas as pd
import os
import numpy as np
import random
from ax.core import ObservationFeatures
from ax.modelbridge.modelbridge_utils import extract_search_space_digest, extract_outcome_constraints, \
    extract_objective_weights, extract_objective_thresholds
from ax.service.ax_client import AxClient

from classes.processed_workload import ProcessedWorkloadModel
from classes.workload_dataset import WorkloadTask
from optimization.methods.embedder import EmbedderBO
from optimization.methods.evaluator import EvaluatorBO
from optimization.methods.optimizer_helpers import update_and_refit_model, manually_attach_trials, \
    handle_sobol_suggestion, handle_edu_suggestion


class OptimizerBO:
    logger = logging.getLogger(__name__)

    def __init__(self, task: WorkloadTask, runtime_target: float, profile_num: int, **kwargs):
        self.runtime_target = runtime_target
        self.profile_num = profile_num
        self.profile_counter = 0
        self.profiled_workloads: List[ProcessedWorkloadModel] = []

        self.best_model = None

        self.task = task

        self.exp_config: dict = kwargs.get("exp_config", {})
        self.normalize: bool = kwargs.get("normalize", True)
        self.num_objectives: int = kwargs.get("num_objectives", 1)

        self.seed: int = kwargs.get("seed", 42)
        np.random.seed(self.seed)

        self.num_init = kwargs.get("num_init", 3)
        self.base_entry = kwargs.get("base_entry", {})
        self.save_file = kwargs.get("save_file", None)
        self.optimizer_strategy_sub = self.base_entry.get("optimizer_strategy_sub", None)

        self.dict_list: List[dict] = []

        # they will all be set in the respective subclasses
        self.ax_client: Optional[AxClient] = None
        self.embedder: Optional[EmbedderBO] = None
        self.evaluator: Optional[EvaluatorBO] = None

    def init(self):
        # for some randomly chosen initial workloads, train the model initially
        num_init = 3 if self.optimizer_strategy_sub.endswith("Ext") else self.num_init
        while len([w for w in self.profiled_workloads if not w.abandon]) < num_init:
            candidates = [c for c in self.task.workloads if not self.already_seen_config(c)]
            workload: Optional[ProcessedWorkloadModel] = self.get_educated_guess(self.profiled_workloads,
                                                                                 candidates)
            if workload is None:
                workload = handle_sobol_suggestion(self.ax_client, self.task.workloads,
                                                   self.embedder, self.evaluator)
            else:
                self.ax_client, workload = handle_edu_suggestion(self.ax_client, workload, self.embedder, self.evaluator)
            if workload is not None:
                self.profile(workload, optional_properties={
                    "generation_strategy": self.ax_client.generation_strategy.model.model.__class__.__name__
                })

    def profile(self, workload: ProcessedWorkloadModel, optional_properties: Optional[dict] = None):
        if optional_properties is None:
            optional_properties = {}

        workload = self.modify_workload(workload)

        self.profiled_workloads.append(workload)
        self.profile_counter += 1
        self.dict_list.append({
            **self.base_entry,
            "seed": self.seed,
            "num_init": self.num_init,
            "normalize": self.normalize,
            "num_objectives": self.num_objectives,
            "profiling_counter": self.profile_counter,
            "trial_index": len(self.ax_client.experiment.trials),
            "profiled_workloads": [w.json() for w in self.profiled_workloads],
            **optional_properties
        })

    def already_seen_config(self, workload: ProcessedWorkloadModel):
        return any([workload.workload_name == w.workload_name for w in self.profiled_workloads])

    def weight_acqf_values(self, acqf_values: List[float], profiled_workloads: List[ProcessedWorkloadModel],
                           candidates: List[ProcessedWorkloadModel]):
        return acqf_values

    def modify_workload(self, workload: ProcessedWorkloadModel):
        return workload

    # override in subclass if necessary
    def get_educated_guess(self, profiled_workloads: List[ProcessedWorkloadModel],
                           candidates: List[ProcessedWorkloadModel]) -> Optional[ProcessedWorkloadModel]:
        return None

    @staticmethod
    def retrieve_candidate(acqf_values: List[float]):
        # determine max acq_value
        max_value = max(acqf_values)
        # get index of max value. In case there are multiple, choice randomly one from them
        max_index = random.choice([idx for idx, val in enumerate(acqf_values) if val == max_value])
        return max_value, max_index

    def run(self, lock=None):
        while self.profile_counter < self.profile_num:
            self._run()

        df: pd.DataFrame = pd.DataFrame(self.dict_list)
        if self.save_file is not None and lock is not None:
            with lock:
                write_header: bool = not os.path.exists(self.save_file)
                df.to_csv(self.save_file, mode="a", index=False, header=write_header)

        self.ax_client = update_and_refit_model(self.ax_client)
        self.best_model = copy.deepcopy(self.ax_client.generation_strategy.model.model.surrogate.model)

    def _run(self):
        self.ax_client = update_and_refit_model(self.ax_client)

        start_eval_time = time.time()
        # need to deepcopy ax_client, so it doesn't change underlying parameter bounds
        ax_client_copy = copy.deepcopy(self.ax_client)

        model_bridge = ax_client_copy.generation_strategy.model
        transformed_gen_args = model_bridge._get_transformed_gen_args(
            search_space=ax_client_copy.experiment.search_space,
        )
        search_space_digest = extract_search_space_digest(
            search_space=transformed_gen_args.search_space,
            param_names=model_bridge.parameters,
        )
        objective_weights = extract_objective_weights(
            objective=ax_client_copy.experiment.optimization_config.objective,
            outcomes=model_bridge.outcomes,
        )
        objective_thresholds = None
        if hasattr(ax_client_copy.experiment.optimization_config, "objective_thresholds"):
            objective_thresholds = extract_objective_thresholds(
                objective_thresholds=ax_client_copy.experiment.optimization_config.objective_thresholds,
                objective=ax_client_copy.experiment.optimization_config.objective,
                outcomes=model_bridge.outcomes
            )
        outcome_constraints = extract_outcome_constraints(
            outcome_constraints=ax_client_copy.experiment.optimization_config.outcome_constraints,
            outcomes=model_bridge.outcomes
        )

        # evaluate list of valid candidates. Exclude those which we already inspected
        prep_tuples = [(ObservationFeatures(self.embedder.construct(self.embedder.vectorize(w))), w)
                       for w in self.task.workloads if not self.already_seen_config(w)]
        # due to a current "bug" in implementation, only one value is returned, so input each observation individually
        acq_options = copy.deepcopy(model_bridge.model.acquisition_options)
        acqf_values = [
            model_bridge.evaluate_acquisition_function(
                # Each `ObservationFeatures` below represents one point in experiment (untransformed) search space:
                observation_features=[observation_feature],
                search_space_digest=search_space_digest,
                objective_weights=objective_weights,
                objective_thresholds=objective_thresholds,
                outcome_constraints=outcome_constraints,
                acq_options=acq_options
            )
            for observation_feature, _ in prep_tuples
        ]
        acqf_values = sum([[v] if isinstance(v, float) else v for v in acqf_values], [])
        # manipulate acqf_values, if required and implemented
        mod_acqf_values = self.weight_acqf_values(acqf_values, self.profiled_workloads, list(zip(*prep_tuples))[-1])
        max_value, max_index = self.retrieve_candidate(mod_acqf_values)
        # get associated observation / experiment / trial
        next_observation = prep_tuples[max_index][0]
        next_experiment = next_observation.parameters

        # get workload associated with encoding (features)
        f_list = self.embedder.reconstruct(next_experiment)
        workload = next((w for w in self.task.workloads if self.embedder.vectorize(w) == f_list))
        # add new trial to ax experiment, according to observation with highest acq_value (expected improvement)
        self.ax_client = manually_attach_trials(self.ax_client, [workload], self.embedder, self.evaluator)
        # actually profile this configuration, i.e. add it to list of profiled workloads
        self.profile(workload, optional_properties={
            "acqf_value": acqf_values[max_index],
            "mod_acqf_value": max_value,
            "acqf_class": self.ax_client.generation_strategy._curr.model_kwargs["botorch_acqf_class"].__name__,
            "generation_strategy": self.ax_client.generation_strategy.model.model.__class__.__name__,
            "fit_time": self.ax_client.generation_strategy.model.fit_time,
            "predict_time": time.time() - start_eval_time
        })
