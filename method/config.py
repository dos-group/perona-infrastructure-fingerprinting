import os

from ray import tune


class GeneralConfig(object):
    epochs: int = 100
    result_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    best_checkpoint_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_checkpoints")
    vector_norm_ord: int = 10
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"
    seed: int = 42


class TuneConfig(object):
    scheduler: dict = {
        "grace_period": 25,
        "reduction_factor": 4
    }
    cli_reporter: dict = {
        "max_progress_rows": 5,
        "max_error_rows": 5,
        "print_intermediate_tables": True,
        "sort_by_metric": True
    }
    tune_best_trial: dict = {
        "scope": "all",
        "filter_nan_and_inf": True
    }
    tune_run: dict = {
        "local_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "ray_results"),  # local result folder
        "mode": "min",
        "metric": "val_loss",
        "checkpoint_score_attr": "min-val_loss",
        "keep_checkpoints_num": 3,
        "verbose": 1,
        "num_samples": 100,
        "max_failures": -1,
        "reuse_actors": True,
        "resume": "AUTO"
    }
    concurrency_limiter: dict = {
        "max_concurrent": 4
    }
    optuna_search: dict = {
        "seed": GeneralConfig.seed
    }
    search_space: dict = {
        # seed config
        "seed": tune.randint(0, 10000),
        # data config
        "batch_size": tune.choice([16]),
        # model config
        "dropout_adj": tune.choice([0.0, 0.1, 0.2]),
        # refer to: https://pytorch-geometric.readthedocs.io/en/2.0.4/modules/nn.html#torch_geometric.nn.conv.TransformerConv
        "hidden_dim": tune.choice([32]),
        "heads": tune.choice([1, 3, 5]),
        "concat": tune.choice([False, True]),
        "beta": tune.choice([False, True]),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        "root_weight": tune.choice([False, True]),
        # loss config
        # --> (controls focal loss)
        "focal_gamma": tune.choice([2]),
        # --> (controls class-balanced loss)
        "classbalanced_beta": tune.choice([0.9999]),
        # --> (controls marginranking loss)
        "ranking_margin_factor": tune.choice([4]),
        # optimizer config
        "learning_rate": tune.choice([0.1, 0.01, 0.001]),
        "weight_decay": tune.choice([0.01, 0.001, 0.0001])
    }



