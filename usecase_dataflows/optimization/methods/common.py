import dataclasses
from typing import List

from ax.core.search_space import SearchSpaceDigest
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.acquisition import qNoisyExpectedImprovement
from botorch.acquisition.input_constructors import acqf_input_constructor, \
    construct_inputs_qNEI
from botorch.models import ModelListGP
from botorch.utils.containers import TrainingData

from optimization.methods.baselines.utils_cherrypick import CherryPickModel


class CustomModelListGP(ModelListGP):
    def __init__(self, *models: CherryPickModel):
        super().__init__(*models)
        self.custom_models: List[CherryPickModel] = [*models]


class CustomSurrogate(Surrogate):
    def fit(
            self,
            training_data: TrainingData,
            search_space_digest: SearchSpaceDigest,
            metric_names: List[str],
            **kwargs
    ) -> None:
        """We override this function because we don't fit the model the 'classical way'."""
        self.construct(
            training_data=training_data,
            metric_names=metric_names,
            **dataclasses.asdict(search_space_digest),
            **kwargs
        )


class CustomqNoisyExpectedImprovement(qNoisyExpectedImprovement):
    pass


@acqf_input_constructor(CustomqNoisyExpectedImprovement)
def construct_custom_inputs_qNEI(*args, **kwargs):
    inputs = construct_inputs_qNEI(*args, **kwargs)
    inputs["cache_root"] = kwargs.get("cache_root", True)
    return inputs
