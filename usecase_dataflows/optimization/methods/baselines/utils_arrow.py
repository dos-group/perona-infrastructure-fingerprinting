from typing import List, Optional, Tuple
import numpy as np
import torch
from botorch.models import FixedNoiseGP
from botorch.utils.containers import TrainingData
from gpytorch import lazify
from gpytorch.distributions import MultivariateNormal
from hummingbird.ml import convert
from hummingbird.ml.containers import PyTorchSklearnContainerRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor

from classes.processed_workload import ProcessedWorkloadModel


def get_arrow_metrics(profiled_workloads: List[ProcessedWorkloadModel], root_scaler: Optional[MinMaxScaler] = None):
    profiled_workloads = [w for w in profiled_workloads if not w.abandon]
    low_level_metrics: np.ndarray = np.array([w.arrow_metrics for w in profiled_workloads])
    # normalize to [0, 1] interval
    if root_scaler is None:
        root_scaler = MinMaxScaler()
        root_scaler.fit(low_level_metrics)
    low_level_metrics = root_scaler.transform(low_level_metrics)
    return torch.from_numpy(low_level_metrics), root_scaler


class ArrowModel(FixedNoiseGP):
    _num_outputs = 1  # to inform GPyTorchModel API
    opt_class: str = "Arrow"

    TREE_ARGS = {"n_estimators": 101,  # prime number, makes life easier
                 "criterion": "squared_error",
                 "max_depth": 5}  # override some default parameters

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs):
        arrow_metrics, _ = get_arrow_metrics(kwargs["profiled_workloads"])
        return {
            "train_x": training_data.Xs[0],
            "train_y": training_data.Ys[0],
            "train_yvar": training_data.Yvars[0],
            "arrow_metrics": arrow_metrics.to(training_data.Xs[0]),
            **kwargs
        }

    def __init__(self, train_x: Tensor, train_y: Tensor, train_yvar: Tensor, arrow_metrics: Tensor, **kwargs):
        self.custom_y_mean: Optional[torch.Tensor] = None
        self.custom_y_std: Optional[torch.Tensor] = None
        if kwargs.get("normalize", False):
            self.custom_y_mean = train_y.mean(dim=-2, keepdim=True)
            self.custom_y_std = train_y.std(dim=-2, keepdim=True)
            train_y = (train_y - self.custom_y_mean) / self.custom_y_std
        super().__init__(train_x, train_y, train_yvar.expand_as(train_y), **kwargs)

        concat_train_x = torch.cat([train_x, arrow_metrics], dim=-1).to(train_x)
        self.target_dtype = concat_train_x.dtype

        # generate all possible pairs of samples
        sample_dim: int = train_x.size()[-2]
        all_pairs = torch.cartesian_prod(torch.arange(sample_dim), torch.arange(sample_dim)).reshape(-1, 2)
        unique_pairs: Tensor = all_pairs[all_pairs[:, 0] != all_pairs[:, 1]].to(train_x).long()

        # batch-mode LOOCV GP
        if len(train_x.size()) == 2:
            train_x = train_x.unsqueeze(0)
            concat_train_x = concat_train_x.unsqueeze(0)
            train_y = train_y.unsqueeze(0)

        self.all_models: List[Tuple[PyTorchSklearnContainerRegression, torch.Tensor]] = []
        for t_x, t_c_x, t_y in zip(train_x, concat_train_x, train_y):
            # prepare input for training
            t_x_arr: np.ndarray = np.array(torch.cat((t_c_x[unique_pairs[:, 0]],
                                                      t_x[unique_pairs[:, 1]]), -1).tolist())
            t_y_arr: np.ndarray = np.array(torch.atleast_2d(t_y[unique_pairs[:, 1]]).tolist())
            # train extra trees ensemble method
            extra_trees_regressor: ExtraTreesRegressor = ExtraTreesRegressor(**self.TREE_ARGS)
            extra_trees_regressor.fit(t_x_arr, t_y_arr)
            # convert to pytorch
            model: PyTorchSklearnContainerRegression = convert(extra_trees_regressor, 'pytorch').to(t_x.device)
            # 1 x 1 x M x F
            self.all_models.append((model, t_c_x.unsqueeze(0).unsqueeze(0)))

        self.to(concat_train_x)  # make sure we're on the right device/dtype

    @staticmethod
    def mean_cov(predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # B x N x F ---> B x N x 1
        mean_x = torch.mean(predictions, dim=-1, keepdim=True)
        # (implement batch covariance calculation)
        fact = 1.0 / (predictions.size(-1) - 1)
        predictions_cent = predictions - mean_x
        covar_x = fact * torch.matmul(predictions_cent, torch.transpose(predictions_cent, 1, 2))
        return mean_x, covar_x

    def forward(self, x: Tensor):
        if self.training:
            x = self.transform_inputs(x)

        mean_list, covar_list = [], []
        generator = zip([x] if len(self.all_models) == 1 else x, self.all_models)
        for x_input, (model, train_x) in generator:
            # B x N x 1 x F
            if x_input.ndim == 2:
                x_input = x_input.unsqueeze(0).unsqueeze(-2)
            else:
                x_input = x_input.unsqueeze(-2)
            # prepare model input ---> B x N x M x F
            tensor_input = torch.cat([
                train_x.expand(x_input.size()[0], x_input.size()[1], -1, -1),
                x_input.expand(-1, -1, train_x.size()[-2], -1)
            ], dim=-1)
            # get model predictions
            B, N, M, F = tensor_input.size()
            model_inputs = tensor_input.reshape(-1, F)
            model_outputs = model.predict(model_inputs)

            # BNM ---> B x N x M
            model_outputs = torch.from_numpy(model_outputs.reshape(B, N, -1)).to(x_input)
            mean_x_inner, covar_x_inner = ArrowModel.mean_cov(model_outputs)
            # if possible, try to squeeze mean and cov
            mean_list.append(mean_x_inner.squeeze())
            covar_list.append(covar_x_inner.squeeze())

        mean_x = torch.stack(mean_list, dim=0).squeeze(0)
        covar_x = torch.stack(covar_list, dim=0).squeeze(0)

        # we assume multivariate normal distribution
        mvn = MultivariateNormal(mean_x, lazify(covar_x)).add_jitter(noise=1e-8)
        return mvn