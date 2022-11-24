import os
from typing import List

import torch
import logging
import time
import pandas as pd
import numpy as np
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities.model_summary import ModelSummary
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from torch_geometric.loader import DataLoader

from config import TuneConfig, GeneralConfig
from modeling.datamodules import PeronaDataModule
from modeling.models import PeronaGraphModel
from pytorch_lightning.utilities.warnings import PossibleUserWarning, LightningDeprecationWarning
import warnings

metric_columns: List[str] = ["val_loss", "loss_curr", "enc_mse", "enc_acc", "nxt_mse", "nxt_acc"]


class HyperOptimizer(object):
    def __init__(self, job_identifier):
        self.job_identifier: str = job_identifier
            
    @staticmethod
    def new_trainer_instance(device: any, **kwargs):
        strategy = SingleDeviceStrategy(device=device)
        return pl.Trainer(strategy=strategy,
                          enable_progress_bar=False,
                          log_every_n_steps=1000,  # we don't want this intermediate logging
                          # clip gradients' global norm to <='gradient_clip_val' using 'gradient_clip_algorithm'
                          gradient_clip_val=GeneralConfig.gradient_clip_val,
                          gradient_clip_algorithm=GeneralConfig.gradient_clip_algorithm,
                          **kwargs)

    @staticmethod
    def prepare_dataset(device: any,
                        data_name: str,
                        data_paths: List[str]):
        # batch size is not relevant here, since we discard this class instance directly
        datamodule: PeronaDataModule = PeronaDataModule(data_name, data_paths, device=device, batch_size=32)
        datamodule.prepare_data()

    @staticmethod
    def test_and_predict_with_best(chpkt_path: str,
                                   device: any,
                                   train_data_name: str,
                                   train_data_paths: List[str],
                                   inference_data_name: str,
                                   inference_data_paths: List[str],
                                   num_gpus: int = torch.cuda.device_count()):
        """Test model with test dataset and retrieve predictions."""
        
        chkpt: dict = torch.load(chpkt_path, map_location=device)
        print(f"Time taken: {chkpt['time_taken']:.2f} seconds.")
        print("Best trial config: {}".format(chkpt["best_trial_config"]))
        print("Checkpoint validation loss: {}".format(chkpt["best_trial_val_loss"]), "\n")

        batch_size: int = chkpt["hyper_parameters"]["batch_size"]
        
        trainer = HyperOptimizer.new_trainer_instance(device,
                                                      gpus=num_gpus,
                                                      max_epochs=-1)

        model = PeronaGraphModel.load_from_checkpoint(chpkt_path).double()
        print(model.hparams)
        print(ModelSummary(model, max_depth=-1), "\n")
        train_datamodule = PeronaDataModule(train_data_name, train_data_paths, device, batch_size=batch_size)
        train_datamodule.prepare_data()
        print(train_datamodule.hparams, "\n")

        train_datamodule.setup(stage="test")
        trainer.test(model, datamodule=train_datamodule)

        inf_datamodule = PeronaDataModule(inference_data_name, inference_data_paths, device, batch_size=batch_size)
        print(inf_datamodule.hparams, "\n")
        if train_data_name == inference_data_name:
            inf_list = train_datamodule.test_data
        else:
            inf_df: pd.DataFrame = inf_datamodule.prepare_data(prepare_data_splits=False)
            inf_list = train_datamodule.transform([inf_df])
        print(f"#Graphs in test set: {len(inf_list)}")

        result_list: List[dict] = []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(DataLoader(inf_list, batch_size=batch_size, shuffle=False)):
                batch = batch.to(model.device)
                enc, enc_norm, enc_dec, enc_cls, nxt, nxt_norm, nxt_dec, nxt_cls, chaos_logits, valid_node_mask = model(batch)
                # get raw embeddings
                output_dict: dict = {
                    "inf_enc": enc,
                    "inf_enc_norm": enc_norm,
                    "inf_enc_dec": enc_dec,
                    "inf_enc_cls": enc_cls,
                    "inf_nxt": nxt,
                    "inf_nxt_norm": nxt_norm,
                    "inf_nxt_dec": nxt_dec,
                    "inf_nxt_cls": nxt_cls,
                    "inf_chaos_logits": chaos_logits,
                    "inf_valid_node_mask": valid_node_mask
                }
                for node_idx in range(len(enc)):
                    intermediate_dict = {"batch_idx": batch_idx, "node_idx": node_idx}
                    temp_dict = {**batch.to_dict(), **output_dict}
                    batch_size = len(temp_dict["batch"].unique())
                    for k, v in temp_dict.items():
                        if any([k.startswith(opt) for opt in ["edge_index", "edge_attr", "input_dim",
                                                              "edge_dim", "output_dim", "ptr", "ranking_"]]):
                            continue
                        resp = v[temp_dict["batch"][node_idx]] if len(v) == batch_size else v[node_idx]
                        resp = resp.tolist() if torch.is_tensor(resp) else resp
                        intermediate_dict[k] = resp
                    result_list.append(intermediate_dict)
        return pd.DataFrame(result_list)

    @staticmethod
    def perform_optimization(hyperoptimizer_instance,
                             device: any,
                             data_name: str,
                             data_paths: List[str],
                             resources_per_trial: dict,
                             exp_suffix: str,
                             num_gpus: int = torch.cuda.device_count()):
        """Perform hyperparameter optimization."""

        # Extract optionally provided configurations #####
        scheduler_config: dict = TuneConfig.scheduler
        optuna_search_config: dict = TuneConfig.optuna_search
        concurrency_limiter_config: dict = TuneConfig.concurrency_limiter
        tune_run_config: dict = TuneConfig.tune_run
        tune_best_trial_config: dict = TuneConfig.tune_best_trial
        search_space_config: dict = TuneConfig.search_space
        cli_reporter_config: dict = TuneConfig.cli_reporter
        ##################################################

        scheduler = ASHAScheduler(max_t=GeneralConfig.epochs, **scheduler_config)

        reporter = CLIReporter(parameter_columns=list(search_space_config.keys()),
                               metric_columns=metric_columns, **cli_reporter_config)

        search_alg = OptunaSearch(**optuna_search_config)
        search_alg = ConcurrencyLimiter(
            search_alg, **concurrency_limiter_config)

        tune_run_name = f"{hyperoptimizer_instance.job_identifier}_{exp_suffix}"
        
        # sets seeds for numpy, torch and python.random.
        pl.seed_everything(GeneralConfig.seed, workers=True)

        start = time.time()

        analysis = tune.run(tune.with_parameters(hyperoptimizer_instance, device=device,
                                                 data_name=data_name,
                                                 data_paths=data_paths,
                                                 num_epochs=GeneralConfig.epochs, num_gpus=num_gpus),
                            name=tune_run_name,
                            scheduler=scheduler,
                            progress_reporter=reporter,
                            search_alg=search_alg,
                            config=search_space_config,
                            resources_per_trial=resources_per_trial,
                            **tune_run_config)

        time_taken = time.time() - start

        # get best trial
        best_trial = analysis.get_best_trial(**tune_best_trial_config)

        # get some information from best trial
        best_trial_val_loss = best_trial.metric_analysis["val_loss"]["min"]

        logging.info(f"Time taken: {time_taken:.2f} seconds.")
        logging.info("Best trial config: {}".format(best_trial.config))
        logging.info("Best trial final validation loss: {}".format(
            best_trial_val_loss))

        # load the best checkpoint of best trial
        best_checkpoint_dir = analysis.get_best_checkpoint(best_trial)

        checkpoint = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"), map_location=device)
        checkpoint["best_trial_config"] = best_trial.config
        checkpoint["best_trial_val_loss"] = best_trial_val_loss
        checkpoint["time_taken"] = time_taken
        return checkpoint

    def __call__(self, config: dict,
                 data_name: str = None,
                 data_paths: List[str] = None,
                 checkpoint_dir=None,
                 device: any = None,
                 num_epochs: int = None,
                 num_gpus: int = 0):
        """Called by 'tune.run' during hyperparameter optimization.
        Check: https://docs.ray.io/en/releases-1.13.0/tune/api_docs/trainable.html"""
        
        warnings.filterwarnings(action="ignore", category=PossibleUserWarning)
        warnings.filterwarnings(action="ignore", category=LightningDeprecationWarning)

        batch_size: int = config["batch_size"]

        datamodule: PeronaDataModule = PeronaDataModule(data_name, data_paths, device=device, batch_size=batch_size)

        datamodule.setup(stage="fit")
        fixed_model_args = {
            "input_dim": datamodule.input_dim,
            "edge_dim": datamodule.edge_dim,
            "output_dim": datamodule.output_dim,
            "neg_sample_count": datamodule.neg_sample_count,
            "pos_sample_count": datamodule.pos_sample_count
        }
        
        # config["seed"] is set deterministically, but differs between training runs
        # sets seeds for numpy, torch and python.random.
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        
        model = PeronaGraphModel(**{**fixed_model_args, **config}).double().to(device)
        print(ModelSummary(model, max_depth=-1), "\n")

        tune_callback = TuneReportCheckpointCallback(
            metrics={v: f"ptl/{'val_' * bool(1 - ('val_' in v))}{v}" for v in metric_columns},
            filename="checkpoint",
            on="validation_end")
        swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)

        resume_from_checkpoint = os.path.join(checkpoint_dir, "checkpoint") if checkpoint_dir is not None else None
        
        trainer = HyperOptimizer.new_trainer_instance(device,
                                                      gpus=num_gpus,
                                                      resume_from_checkpoint=resume_from_checkpoint,
                                                      max_epochs=num_epochs,
                                                      callbacks=[tune_callback, swa_callback])

        trainer.fit(model, datamodule=datamodule)
