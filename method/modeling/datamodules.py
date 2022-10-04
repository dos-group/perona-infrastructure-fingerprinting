import copy
import json
import os.path
import re
import io
from datetime import timedelta

import dill
from typing import Optional, List, Union
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sqlalchemy import inspect, and_, or_, asc, create_engine, Index
from sqlalchemy.orm import Session, subqueryload
from torch_geometric import transforms
from torch_geometric.loader import DataLoader
from modeling.utils import init_logging
from pathlib import Path

from modeling.transforms import GraphCreator, PeronaCompose, FeatureSelector, MinMaxScaler, FeatureUnifier, \
    GraphFinalizer, FeatureRotator, PeronaData, FeaturePreprocessor
from orm.models import Benchmark, BenchmarkMetric, NodeMetric


def object_as_dict(target_obj):
    new_dict: dict = {c.key: getattr(target_obj, c.key)
                      for c in inspect(target_obj).mapper.column_attrs}
    if target_obj.__dict__.get("metrics", None) is not None:
        new_dict["metrics"] = [object_as_dict(o) for o in target_obj.__dict__.get("metrics")]
    return new_dict


class PeronaDataModule(pl.LightningDataModule):
    def __init__(self, data_name: str, data_paths: List[str], device: any, batch_size: int = 32, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.device: any = device

        self.data_paths: List[str] = data_paths
        self.artifact_paths: List[str] = [re.sub(r"\.sqlite$", ".csv", dp) for dp in self.data_paths]
        self.real_artifact_path: str = os.path.join(Path(__file__).absolute().parents[1],
                                                    "artifacts", f"{data_name}.csv")
        Path(self.real_artifact_path).absolute().parents[0].mkdir(parents=True, exist_ok=True)

        self.batch_size: int = batch_size
        self.seed: int = kwargs.get("seed", 42)

        self.input_dim: Optional[int] = None
        self.edge_dim: Optional[int] = None
        self.output_dim: Optional[int] = None

        self.next_pos_sample_count: float = 0
        self.next_neg_sample_count: float = 0
        self.ranking_margin: float = 0
        self.train_data: List[PeronaData] = []
        self.valid_data: List[PeronaData] = []
        self.test_data: List[PeronaData] = []

        self.transform: PeronaCompose = PeronaCompose([
            FeatureUnifier(),
            FeaturePreprocessor(),
            FeatureSelector(min_std=0.01),
            FeatureRotator(),
            GraphCreator(min_graph_size=4, max_graph_size=8),
            MinMaxScaler(target_property_name="x"),
            MinMaxScaler(target_property_name="edge_attr"),
            GraphFinalizer(),
            transforms.ToDevice(device=device)
        ])

    def prepare_data(self, prepare_data_splits: bool = True) -> Union[None, pd.DataFrame]:
        
        def retrieve_and_extract_node_metrics(cond, suffix):
            node_metrics = session \
                            .query(NodeMetric.metric, NodeMetric.value) \
                            .filter(cond) \
                            .order_by(asc(NodeMetric.timestamp)) \
                            .distinct(NodeMetric.metric) \
                            .all()
            agg_dict: dict = {}
            for k, v in node_metrics:
                agg_dict[k] = agg_dict.get(k, []) + [float(v)]
            new_dict: dict = {}
            for k, v_list in agg_dict.items():
                percentiles: List[int] = [10, 50, 90]
                for perc, perc_val in zip(percentiles, np.percentile(np.array(v_list), percentiles)):
                    new_dict[f"node_metric_{k}_{perc}th_{suffix}"] = float(perc_val)
            return new_dict
        
        if not os.path.exists(self.real_artifact_path):
            all_results: List[pd.DataFrame] = []
            for data_path in self.data_paths:
                temp_results = []
                engine = create_engine(f"sqlite+pysqlite:///{data_path}")
                with Session(engine) as session:
                    try:
                        Index('BenchmarkMetric_benchmark_id_idx', BenchmarkMetric.benchmark_id).create(bind=engine)
                    except:
                        pass
                    try:
                        Index('NodeMetric_node_name_idx', NodeMetric.node_name).create(bind=engine)
                    except:
                        pass
                    try:
                        Index('NodeMetric_timestamp_idx', NodeMetric.timestamp).create(bind=engine)
                    except:
                        pass
                    
                    rows = session \
                        .query(Benchmark) \
                        .options(subqueryload(Benchmark.metrics)) \
                        .join(BenchmarkMetric, Benchmark.id == BenchmarkMetric.benchmark_id)
                    for row in rows:
                        row_dict: dict = object_as_dict(row)
                        row_dict["duration"] = (row_dict["finished"] - row_dict["started"]).total_seconds()
                        metrics = row_dict.pop("metrics", [])
                        for m in metrics:
                            row_dict[f"{row_dict['type']}-{m['name']}"] = (m["value"], m["unit"])
                        
                        # ranges with respect to our prometheus config (agg-intervals of 20s, agg-action every 10s)
                        node_id = row_dict["node_id"]
                        # prev_metrics: get start-date and end-date (relaxed range)
                        prev_start_date = row_dict["started"] - timedelta(seconds=50)
                        prev_end_date = row_dict["started"] + timedelta(seconds=10)
                        prev_cond = and_(NodeMetric.timestamp <= prev_end_date,
                                         NodeMetric.timestamp > prev_start_date,
                                         NodeMetric.node_name == node_id)
                        node_metrics_prev = retrieve_and_extract_node_metrics(prev_cond, "prev")
                        # next_metrics: get start-date and end-date (relaxed range)
                        next_start_date = row_dict["finished"] + timedelta(seconds=10)
                        next_end_date = row_dict["finished"] + timedelta(seconds=70)
                        next_cond = and_(NodeMetric.timestamp <= next_end_date,
                                         NodeMetric.timestamp > next_start_date,
                                         NodeMetric.node_name == node_id)
                        node_metrics_next = retrieve_and_extract_node_metrics(next_cond, "next")
                        # surr_metrics: get start-date and end-date (relaxed range)
                        node_metrics_surr = retrieve_and_extract_node_metrics(or_(prev_cond, next_cond), "surr")
                        # curr_metrics: get start-date and end-date (relaxed range)
                        curr_start_date = row_dict["started"] - timedelta(seconds=10)
                        curr_end_date = row_dict["finished"] + timedelta(seconds=10)
                        curr_cond = and_(NodeMetric.timestamp <= curr_end_date,
                                         NodeMetric.timestamp > curr_start_date,
                                         NodeMetric.node_name == node_id)
                        node_metrics_curr = retrieve_and_extract_node_metrics(curr_cond, "curr")
                        # add aggregated metrics
                        row_dict = {**row_dict, **node_metrics_prev, **node_metrics_next,
                                    **node_metrics_surr, **node_metrics_curr}
                        temp_results.append(row_dict)
                temp_df: pd.DataFrame = pd.DataFrame(temp_results)
                all_results.append(temp_df)

            all_df: pd.DataFrame = pd.concat(all_results, ignore_index=True)

            type_encoder = OneHotEncoder(sparse=False)
            transformed = type_encoder.fit_transform(all_df['type'].to_numpy().reshape(-1, 1))
            # Create a Pandas DataFrame of the hot encoded column
            ohe_df = pd.DataFrame(transformed, columns=type_encoder.get_feature_names_out())
            # concat with original data
            all_df = pd.concat([all_df, ohe_df], axis=1)
            all_df["category"] = all_df[["node_id", "type", "chaos_applied"]].apply(lambda r: "-".join([str(el)
                                                                                                        for el in
                                                                                                        r.tolist()]),
                                                                                    axis=1)
            print(all_df.shape, list(all_df.columns))
            all_df = all_df.reset_index(drop=True)
            all_df.to_csv(self.real_artifact_path, index=False)
            init_logging("INFO")
            if prepare_data_splits:
                self.prepare_data_splits(all_df)
            return copy.deepcopy(all_df)
        else:
            all_df = pd.read_csv(self.real_artifact_path, parse_dates=["started"])
            if prepare_data_splits:
                t_path = Path(self.real_artifact_path)
                if len([f for f in os.listdir(t_path.absolute().parents[0]) if f.startswith(t_path.stem)]) == 1:
                    self.prepare_data_splits(all_df)
            return copy.deepcopy(all_df)

    def prepare_data_splits(self, total_df: pd.DataFrame):
        np.random.seed(self.seed)
        all_indices, all_labels = np.arange(len(total_df)), total_df["category"].values
        train_indices, temp, _, _ = train_test_split(all_indices, all_indices, test_size=0.4, train_size=0.6,
                                                     shuffle=True, stratify=all_labels)
        val_indices, test_indices, _, _ = train_test_split(temp, temp, test_size=0.5, train_size=0.5,
                                                           shuffle=True, stratify=all_labels[temp])

        self.transform.fit([copy.deepcopy(total_df.iloc[train_indices])])

        train_list = self.transform([copy.deepcopy(total_df.iloc[train_indices])])
        val_list = self.transform([copy.deepcopy(total_df.iloc[val_indices])])
        test_list = self.transform([copy.deepcopy(total_df.iloc[test_indices])])
        predict_list = self.transform([copy.deepcopy(total_df)])

        next_class_targets = torch.cat([el.chaos[1:] for el in predict_list], dim=-1)
        next_pos_sample_count: float = torch.sum(next_class_targets).item()
        next_neg_sample_count: float = len(next_class_targets) - next_pos_sample_count

        with open(re.sub(r"\.csv$", "_config.json", self.real_artifact_path), "w") as json_file:
            json.dump({
                "next_neg_sample_count": next_neg_sample_count,
                "next_pos_sample_count": next_pos_sample_count,
                "ranking_margin": predict_list[0].ranking_margin,
                "input_dim": predict_list[0].input_dim,
                "edge_dim": predict_list[0].edge_dim,
                "output_dim": predict_list[0].output_dim
            }, json_file)

        for obj, the_suffix in zip([train_list, val_list, test_list, self.transform],
                                   ["train", "val", "test", "transform"]):
            with open(re.sub(r"\.csv$", f"_{the_suffix}.pt", self.real_artifact_path), "wb") as torch_file:
                torch.save(obj, torch_file, pickle_module=dill)

    def setup(self, stage: Optional[str] = None):
        # Load general config
        with open(re.sub(r"\.csv$", "_transform.pt", self.real_artifact_path), "rb") as torch_file:
            self.transform = torch.load(io.BytesIO(torch_file.read()), pickle_module=dill)

        with open(re.sub(r"\.csv$", "_config.json", self.real_artifact_path), "rb") as json_file:
            self.__dict__.update(**json.load(json_file))

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            with open(re.sub(r"\.csv$", "_train.pt", self.real_artifact_path), "rb") as torch_file:
                self.train_data = torch.load(io.BytesIO(torch_file.read()), pickle_module=dill)
            with open(re.sub(r"\.csv$", "_val.pt", self.real_artifact_path), "rb") as torch_file:
                self.valid_data = torch.load(io.BytesIO(torch_file.read()), pickle_module=dill)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            with open(re.sub(r"\.csv$", "_test.pt", self.real_artifact_path), "rb") as torch_file:
                self.test_data = torch.load(io.BytesIO(torch_file.read()), pickle_module=dill)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
