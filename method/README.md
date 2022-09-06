# Method: Perona Infrastructure Fingerprinting

In this folder, we present all the code used for our modeling approach. It involves a custom datamodule and associated transforms of the data, the actual model, and code for conducting the hyperparameter optimization (the associated configuration can be found in `config.py`). Here, the file `main.py` is the entry point, where you provide arguments such as training data source, inference data source, and resources usable by the optimization routine. Internally, a hyperparameter optimization on the subdivided training sources will be conducted, with the best model found being eventually used for inference on the respective target data source.

While the expected data sources are expected to be relative paths to the data-directory in the root directory of this repository, preprocessed artifacts as well as results will be written to a local `artifacts` folder in this directory.

All package requirements are defined in `requirements.txt`, where we fix the version for the relevant packages. 

## Technical Details

#### Important Packages
* Python `3.8.0+`
* [PyTorch](https://pytorch.org/) `1.11.0`
* [PyTorch Lightning](https://www.pytorchlightning.ai/) `1.6.4`
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/2.0.4/) `2.0.4`
* [PyTorch Metric Learning](https://kevinmusgrave.github.io/pytorch-metric-learning/) `1.5.2`
* [Ray Tune](https://docs.ray.io/en/releases-1.13.0/tune/index.html) `1.13.0`
* [Optuna](https://optuna.org/) `2.10.1`

It is recommended to install the requirements of our approach (`requirements.txt`) into a virtual environment.

## Hyperparameter Optimization
During model training, we use [Ray Tune](https://docs.ray.io/en/releases-1.13.0/tune/index.html) and [Optuna](https://optuna.org/) to carry out a search for optimized hyperparameters. As of now, we draw 100 trials using [Optuna](https://optuna.org/) from a manageable search space. All details are listed in the `config.py` and are in most cases directly passed to the respective functions / methods / classes in [Ray Tune](https://docs.ray.io/en/releases-1.13.0/tune/index.html). For instance, it can be inferred from the `config.py` file that we used the [ASHAScheduler](https://docs.ray.io/en/releases-1.13.0/tune/api_docs/schedulers.html) implemented in Ray Tune to early terminate bad trials, pause trials, clone trials, and alter hyperparameters of a running trial. Also, we used Optuna's default sampler, namely [TPESampler](https://optuna.readthedocs.io/en/v2.10.1/reference/generated/optuna.samplers.TPESampler.html), to provide trial suggestions.