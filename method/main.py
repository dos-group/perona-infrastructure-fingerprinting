import argparse
import copy
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import classification_report

from config import GeneralConfig
from modeling.hyper_opt import HyperOptimizer
from modeling.utils import create_dirs

parser = argparse.ArgumentParser()

parser.add_argument("-tdp", "--train-data-paths", type=str, nargs="*",
                    required=True, help="Path(s) to train dataset(s).")
parser.add_argument("-idp", "--inference-data-paths", type=str, nargs="*",
                    required=True, help="Path(s) to dataset(s) for inference.")
parser.add_argument("-d", "--device", type=str, required=True,
                    help="If available, will use this device for training.")
parser.add_argument("-cr", "--cpu-resources", type=int, required=True,
                    help="Number of cores per trial.")
parser.add_argument("-gr", "--gpu-resources", type=float, required=True,
                    help="Fraction of gpu to use per trial.")

args = parser.parse_args()

device: any = torch.device(args.device)

train_data_paths: List[str] = [str(Path(dp).absolute()) for dp in args.train_data_paths]
train_data_name: str = "_AND_".join(Path(dp).stem for dp in args.train_data_paths)

inference_data_paths: List[str] = [str(Path(dp).absolute()) for dp in args.inference_data_paths]
inference_data_name: str = "_AND_".join(Path(dp).stem for dp in args.inference_data_paths)

resources_per_trial: dict = {
    "cpu": args.cpu_resources,  # how many cpu cores per trial?
    "gpu": args.gpu_resources  # needs to be "0" on cpu-only devices. You can also specify fractions
}

job_identifier: str = f"{train_data_name}"
normal_identifier: str = f"{train_data_name}_{args.device}"
exp_suffix: str = datetime.now().strftime('%Y%m%d-%H%M%S')

torch.cuda.empty_cache()
num_gpus: int = int(torch.cuda.is_available())

checkpoint_path: Optional[str] = os.path.join(GeneralConfig.best_checkpoint_dir,
                                              f"{job_identifier}_{exp_suffix}_checkpoint.pt")
if not os.path.exists(checkpoint_path):
    # perform optimization
    HyperOptimizer.prepare_dataset(device, train_data_name, train_data_paths)
    checkpoint = HyperOptimizer.perform_optimization(HyperOptimizer(job_identifier),
                                                     device,
                                                     train_data_name,
                                                     train_data_paths,
                                                     resources_per_trial,
                                                     exp_suffix,
                                                     num_gpus=num_gpus)

    # save checkpoint
    create_dirs(GeneralConfig.best_checkpoint_dir)
    torch.save(checkpoint, checkpoint_path)

summary_df: pd.DataFrame = HyperOptimizer.test_and_predict_with_best(checkpoint_path, device,
                                                                     train_data_name, train_data_paths,
                                                                     inference_data_name, inference_data_paths,
                                                                     num_gpus=num_gpus)

loc_summary_df = copy.deepcopy(summary_df)
print("#" * 30, "Outlier Detection Report", "#" * 30)
y_true = loc_summary_df.chaos.values
y_pred = np.zeros_like(y_true)
y_pred[expit(loc_summary_df.inf_chaos_logits.values) >= 0.5] = 1
target_names = ['Normal', 'Outlier']
print(classification_report(y_true, y_pred, target_names=target_names))

print("#" * 30, "Benchmark-Type Classification Report", "#" * 30)
y_true = loc_summary_df.bm_id.values
y_pred = np.argmax(np.array(loc_summary_df["inf_cls_enc"].tolist()), axis=1)
target_names = list(sorted(list(loc_summary_df["bm_name"].unique())))
print(classification_report(y_true, y_pred, target_names=target_names))

summary_df.to_csv(os.path.join(Path(__file__).parents[0].absolute(), "artifacts",
                               f"{inference_data_name}_{exp_suffix}_results.csv"), index=False)
