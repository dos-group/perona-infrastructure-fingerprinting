from typing import List, Optional

from ax import Models
from ax.modelbridge.generation_node import GenerationStep
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
from ax.service.ax_client import AxClient
import numpy as np
from ax.service.utils.instantiation import ObjectiveProperties
from botorch.acquisition import qNoisyExpectedImprovement
from gpytorch.mlls import SumMarginalLogLikelihood

from classes.processed_workload import ProcessedWorkloadModel
from optimization.methods.baselines.utils_arrow import ArrowModel
from optimization.methods.baselines.utils_cherrypick import CherryPickModel


def create_sobol_generation_step(min_trials_observed: int, seed: int):
    return GenerationStep(
        model=Models.SOBOL,
        model_kwargs={"seed": seed},
        max_parallelism=1,  # sequential evaluation
        num_trials=min_trials_observed,
        min_trials_observed=min_trials_observed,
        enforce_num_trials=False,
        should_deduplicate=True
    )


def create_cherrypick_generation_step(optimizer):
    return GenerationStep(
        model=Models.BOTORCH_MODULAR,
        num_trials=-1,
        enforce_num_trials=True,
        max_parallelism=1,  # sequential evaluation
        model_gen_kwargs=optimizer.exp_config.get("model_gen_kwargs", None),
        model_kwargs={
            **optimizer.exp_config.get("model_kwargs", {}),
            "surrogate": ListSurrogate(botorch_submodel_class=CherryPickModel,  # GP for objective + constraint
                                       mll_class=SumMarginalLogLikelihood),
            "botorch_acqf_class": qNoisyExpectedImprovement,  # MC-based batch Noisy Expected Improvement
            "acquisition_options": optimizer.exp_config.get("acquisition_options", None)
        }
    )


def create_arrow_generation_step(optimizer):
    return GenerationStep(
        model=Models.BOTORCH_MODULAR,
        num_trials=-1,
        enforce_num_trials=True,
        max_parallelism=1,  # sequential evaluation
        model_gen_kwargs=optimizer.exp_config.get("model_gen_kwargs", None),
        model_kwargs={
            **optimizer.exp_config.get("model_kwargs", {}),
            "surrogate": ListSurrogate(botorch_submodel_class=ArrowModel,  # GP for objective + constraint
                                       submodel_options={
                                           # this is an important ref-link! DO NOT REMOVE
                                           "profiled_workloads": optimizer.profiled_workloads
                                       },
                                       mll_class=SumMarginalLogLikelihood),
            "botorch_acqf_class": qNoisyExpectedImprovement,  # MC-based batch Noisy Expected Improvement
            "acquisition_options": optimizer.exp_config.get("acquisition_options", None)
        }
    )


def manually_attach_trials(ax_client: AxClient, workloads: List[ProcessedWorkloadModel], embedder, evaluator):
    for workload in workloads:
        parametrization = embedder.construct(embedder.vectorize(workload))
        _, trial_index = ax_client.attach_trial(parametrization)
        if not workload.abandon:
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluator(parametrization))
        else:
            ax_client.abandon_trial(trial_index=trial_index, reason="did not complete (i.e. timeout or failure)")
    return ax_client

def handle_edu_suggestion(ax_client: AxClient, workload: ProcessedWorkloadModel, embedder, evaluator):
    return manually_attach_trials(ax_client, [workload], embedder, evaluator), workload

def handle_sobol_suggestion(ax_client: AxClient, workloads: List[ProcessedWorkloadModel], embedder, evaluator):
    parametrization, trial_index = ax_client.get_next_trial()
    f_list = embedder.reconstruct(parametrization)
    workload: Optional[ProcessedWorkloadModel] = next((w for w in workloads if embedder.vectorize(w) == f_list), None)
    if workload is not None:
        if not workload.abandon:
            ax_client.complete_trial(trial_index=trial_index, raw_data=evaluator(parametrization))
        else:
            ax_client.abandon_trial(trial_index=trial_index, reason="did not complete (i.e. timeout or failure)")
        return workload
    else:
        ax_client.abandon_trial(trial_index=trial_index, reason="no valid configuration")
        return None


def update_and_refit_model(ax_client: AxClient):
    # ensure ax_client.generation_strategy.model gets assigned
    # this triggers a model-retraining, although we do not really need the trial suggestion
    ax_client.generation_strategy.experiment = ax_client.experiment
    ax_client.generation_strategy._maybe_move_to_next_step()
    ax_client.generation_strategy._fit_or_update_current_model(None)
    return ax_client


def get_soo_objective(*args, **kwargs):
    return {"objective_cost": ObjectiveProperties(minimize=True)}


def setup_soo_experiment(name: str, gs: GenerationStrategy, embedder, evaluator, task):
    ax_client = AxClient(generation_strategy=gs,
                         verbose_logging=False,
                         enforce_sequential_optimization=True)

    ax_client.create_experiment(
        name=name,
        parameters=embedder.get_search_space_as_list(),
        # must obey to target time
        outcome_constraints=[f"constraint_rt <= {np.log(evaluator.runtime_target)}"],
        objectives=get_soo_objective(evaluator, task),
        immutable_search_space_and_opt_config=True,
    )
    return ax_client
