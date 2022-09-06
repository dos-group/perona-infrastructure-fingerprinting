import copy
import logging
import math
import re
from abc import ABC
from itertools import permutations
from typing import List, Any, Union, Dict, Callable, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose

from config import GeneralConfig


class PeronaData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key.startswith('ranking_indices'):
            return self.x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class PeronaBaseTransform(BaseTransform, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__is_fitted__: bool = False

    def fit(self, data: any) -> Union[Any, List[Any]]:
        raise NotImplementedError

    def print(self, *args):
        logging.info(f"{self.__class__.__name__}: {*args,}")

    @staticmethod
    def bm_cols(df: pd.DataFrame):
        bm_types = list(set(df["type"].values.reshape(-1).tolist()))
        return [col_name for col_name in list(df.columns) if
                any([col_name.startswith(bm_type) for bm_type in bm_types])]

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{k}={v}' for k, v in self.__dict__.items()])})"


class PeronaCompose(Compose):
    def __init__(self, transforms: List[PeronaBaseTransform]):
        super().__init__(transforms)
        self.__is_fitted__: bool = False
        self.transforms: List[PeronaBaseTransform] = transforms

    @staticmethod
    def collect_and_merge(nested_list: List[Any]):
        new_list = []
        for element in nested_list:
            new_list += [element] if not isinstance(element, list) else PeronaCompose.collect_and_merge(element)
        return new_list

    def fit(self, data_list: List[any]) -> None:
        if self.__is_fitted__:
            return

        for t in self.transforms:
            if "__is_fitted__" in t.__dict__ and not t.__is_fitted__:
                for data in data_list:
                    t.fit(data)
                t.__is_fitted__ = True
            data_list = PeronaCompose.collect_and_merge([t(data) for data in data_list])
        self.__is_fitted__ = True

    def __call__(self, data_list: List[any]):
        for t in self.transforms:
            data_list = PeronaCompose.collect_and_merge([t(data) for data in data_list])
        return data_list

    def __repr__(self):
        args = ['    {},'.format(t) for t in self.transforms]
        return '{}([\n{}\n])'.format(self.__class__.__name__, '\n'.join(args))


class FeatureSelector(PeronaBaseTransform):
    def __init__(self, min_std: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_std: float = min_std
        self.cols_to_include: List[str] = ["node_id", "started", "type", "chaos_applied", "chaos_desired"]

    def fit(self, data: pd.DataFrame):
        df = copy.deepcopy(data)
        for col_name in list(df.columns):
            series = df[col_name]
            if col_name in self.cols_to_include:
                continue
            elif any([col_name.startswith(opt) for opt in ["x0_"]]):
                self.print(f"Going to include column with name='{col_name}'. Reason: Type Encoding")
                self.cols_to_include.append(col_name)
            elif all([isinstance(el, (int, float, type(None))) for el in series.tolist()]):
                # column should have various values and not only zeros
                if series.nunique(dropna=True) > 1 and (series[series.notnull()] > 0).any():
                    if series.std(skipna=True) >= self.min_std:
                        self.print(f"Going to include column with name='{col_name}'. Reason: Sufficient Std")
                        self.cols_to_include.append(col_name)
        return None

    def __call__(self, data: pd.DataFrame) -> Any:
        df = copy.deepcopy(data)
        if len(self.cols_to_include):
            df = df.iloc[:, df.columns.isin(self.cols_to_include)]
        self.print(data.shape, df.shape)
        return copy.deepcopy(df)


class FeatureUnifier(PeronaBaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cols_to_include: List[str] = []
        self.mappings: Dict[str, Callable] = {
            "(error)": lambda x: None,
            "scalar": lambda x: x,
            "%": lambda x: x,
            # map to seconds
            "s": lambda x: x,
            "sec": lambda x: x,
            "/sec": lambda x: x,
            "ms": lambda x: x / 1000,
            "msec": lambda x: x / 1000,
            "us": lambda x: x / 1000000,
            "usec": lambda x: x / 1000000,
            # map to MB/sec
            "MB/sec": lambda x: x,
            "ms/GB": lambda x: (1000 / x) * 1000,
            "sec/GB": lambda x: (1 / x) * 1000,
            "MiB/s": lambda x: x * 1.04858,
            "MiB/sec": lambda x: x * 1.04858,
            "GBytes/sec": lambda x: x * 1024,
            "KiB/s": lambda x: x / 976.5625,
            # map to steps of thousand (K),
            "K/sec": lambda x: x,
            # map to MB
            "MB": lambda x: x,
            "MBytes": lambda x: x,
            "KiB": lambda x: x / 976.5625,
            "MiB": lambda x: x * 1.04858,
            "KB": lambda x: x / 1024,
            "GB": lambda x: x * 1024,
            "GBytes": lambda x: x * 1024,
            "bytes": lambda x: x / 1048576
        }

    def fit(self, data: pd.DataFrame):
        unified_data: pd.DataFrame = self.__call__(data)
        unified_data = unified_data.loc[:, unified_data.notna().any(axis=0)]
        self.cols_to_include = list(unified_data.columns)
        return None

    def __call__(self, data: pd.DataFrame) -> Any:
        def try_to_parse_value(target_val: any):
            return isinstance(target_val, str) and re.match(r"^\(.*,.*\)$", target_val) and eval(target_val)

        df = copy.deepcopy(data)
        for df_index, row in df.iterrows():
            for col_name, col_value in row.iteritems():
                eval_col_value = try_to_parse_value(col_value) or col_value
                if isinstance(eval_col_value, tuple):
                    df.loc[df_index, col_name] = self.mappings[eval_col_value[-1]](eval_col_value[0])
                elif isinstance(eval_col_value, bool):
                    df.loc[df_index, col_name] = int(eval_col_value)
        if len(self.cols_to_include):
            df = df.iloc[:, df.columns.isin(self.cols_to_include)]
        self.print(data.shape, df.shape)
        return copy.deepcopy(df)


class FeatureRotator(PeronaBaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotate_dict: Dict[str, Tuple[bool, preprocessing.MinMaxScaler]] = {}

    def fit(self, data: pd.DataFrame):
        df = copy.deepcopy(data)
        for col_name in list(df.columns):
            if col_name in PeronaBaseTransform.bm_cols(data):
                series = df[col_name]
                raw_values = series[series.notnull()].values
                values = preprocessing.robust_scale(raw_values, quantile_range=(25.0, 75.0))
                lower, middle, upper = np.percentile(values, [0, 50, 100])
                left, right = abs(middle - lower), abs(upper - middle)
                # check if increase / decrease, fit scaler for usage in __call__
                increase: bool = bool(right <= left)
                scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
                scaler.fit(raw_values.reshape(-1, 1))
                self.rotate_dict[col_name] = (increase, scaler)
        return None

    def __call__(self, data: pd.DataFrame):
        df = copy.deepcopy(data)
        for col_name in list(df.columns):
            my_tuple: Tuple[bool, preprocessing.MinMaxScaler] = self.rotate_dict.get(col_name, None)
            if my_tuple is not None:
                increase, scaler = my_tuple
                series: pd.Series = copy.deepcopy(df[col_name])
                if not increase:
                    trans_values = scaler.transform(series[series.notnull()].values.reshape(-1, 1))
                    series[series.notnull()] = scaler.inverse_transform((trans_values * -1) + 1).reshape(-1).tolist()
                df.loc[:, [col_name]] = series.values
        self.print(data.shape, df.shape)
        return copy.deepcopy(df)


class GraphCreator(PeronaBaseTransform):
    def __init__(self, min_graph_size: int = 4, max_graph_size: int = 4, date_col: str = "started",
                 id_col: str = "type", node_id_col: str = "node_id", chaos_col: str = "chaos_applied", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_graph_size: int = min_graph_size
        self.max_graph_size: int = max_graph_size
        self.date_col: str = date_col
        self.id_col: str = id_col
        self.node_id_col: str = node_id_col
        self.chaos_col: str = chaos_col
        self.cols_to_drop = [self.date_col, self.node_id_col, chaos_col, "chaos_desired"]
        # below: will be set / inferred during fit-procedure
        self.col_means: Optional[Union[pd.DataFrame, pd.Series]] = None
        self.bm_ids: List[str] = []
        self.onehot_cols: List[str] = []
        self.node_metric_cols: List[str] = []
        self.data_notna_mask_dict: Dict[str, List[str]] = {}

    @staticmethod
    def _calculate_edge_feature(dividend: float, divisor: float):
        return 1 / 2 ** math.log1p(dividend / divisor)

    @staticmethod
    def _calculate_edges(node_metrics_slice: np.ndarray,
                         date_series_slice: pd.DataFrame):
        edge_index = []
        edge_attr = []
        for a in range(len(node_metrics_slice)):
            for b in range(a, len(node_metrics_slice)):
                # only forward connections
                if a < b:
                    edge_index.append((a, b))
                    timedelta = date_series_slice.iloc[b] - date_series_slice.iloc[a]
                    edge_attr.append([
                                         1 / max(1, b - a),
                                         GraphCreator._calculate_edge_feature(timedelta.seconds, 60),
                                         GraphCreator._calculate_edge_feature(timedelta.seconds, 3600),
                                         GraphCreator._calculate_edge_feature(timedelta.seconds, 3600 * 24),
                                         GraphCreator._calculate_edge_feature(timedelta.seconds, 3600 * 24 * 7)
                                     ] + node_metrics_slice[a].reshape(-1).tolist())

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.double)
        return edge_index, edge_attr

    def fit(self, df: pd.DataFrame):
        df_copy: pd.DataFrame = copy.deepcopy(df)
        masked_df: pd.DataFrame = df_copy.loc[:, ~df_copy.columns.isin(self.cols_to_drop)]
        self.node_metric_cols = [name for name in list(df_copy.columns) if "node_metric" in name]
        self.onehot_cols = [name for name in list(df_copy.columns) if "x0_" in name]
        self.col_means = masked_df.loc[:, ~masked_df.columns.isin([self.id_col])].mean(skipna=True)
        self.bm_ids = list(sorted(list(df_copy[self.id_col].unique())))
        for bm_name in self.bm_ids:
            bm_rows = masked_df.loc[masked_df[self.id_col] == bm_name, ~masked_df.columns.isin(self.node_metric_cols)]
            col_names = list(bm_rows.loc[:, bm_rows.notna().any(axis=0)].columns)
            self.data_notna_mask_dict[bm_name] = list(sorted(set(col_names).difference({self.id_col})))
        return None

    def __call__(self, df: pd.DataFrame):
        df_copy: pd.DataFrame = copy.deepcopy(df)
        df_copy = df_copy.sort_values(by=[self.date_col])

        data_list: List[PeronaData] = []

        for node_name in list(sorted(list(df_copy[self.node_id_col].unique()))):
            sub_df: pd.DataFrame = copy.deepcopy(df_copy.loc[df_copy[self.node_id_col] == node_name, :])

            for bm_id, bm_name in enumerate(self.bm_ids):
                sub_sub_df: pd.DataFrame = copy.deepcopy(sub_df.loc[sub_df[self.id_col] == bm_name, :])
                sub_sub_df = sub_sub_df.reset_index(drop=True)
                data_df = sub_sub_df.drop(columns=self.cols_to_drop + [self.id_col])
                date_series = sub_sub_df[self.date_col]
                chaos_series = sub_sub_df[self.chaos_col]
                if self.col_means is not None:
                    data_df = data_df.fillna(self.col_means)

                for index in range(self.min_graph_size, len(data_df)):
                    # consider only normal executions as predecessor nodes
                    mask = sub_sub_df.loc[((sub_sub_df.index < index - 1) & (sub_sub_df[self.chaos_col] == 0)) |
                                          (sub_sub_df.index == index - 1), :].tail(self.max_graph_size).index

                    if len(mask) < self.min_graph_size:
                        continue

                    data_arr_slice = data_df.iloc[mask, ~data_df.columns.isin(self.node_metric_cols)]
                    onehot_arr_slice = data_df.iloc[mask, data_df.columns.isin(self.onehot_cols)]
                    node_metrics_slice = data_df.iloc[mask, data_df.columns.isin(self.node_metric_cols)]
                    date_series_slice = date_series.iloc[mask]
                    chaos_series_slice = chaos_series.iloc[mask]

                    edge_index, edge_attr = self._calculate_edges(  # inverse of node metrics (higher is better)
                        1 / (node_metrics_slice.values + 0.001),
                        date_series_slice)
                    notna_mask = torch.from_numpy(data_arr_slice.columns.isin(self.data_notna_mask_dict[bm_name]))
                    # Result:
                    # - graph with self.min_graph_size <= x <= self.max_graph_size nodes
                    # - only forward connections, edges have attributes regarding time and metrics
                    # - no anomalous nodes as source --> Intention: we assume recordings of normal executions
                    # - in case a new run would be anomalous we would very certainly not add it to dataset!
                    data_obj = PeronaData(x=torch.from_numpy(data_arr_slice.values),
                                          onehot=torch.from_numpy(onehot_arr_slice.values),
                                          bm_id=torch.tensor([bm_id] * len(data_arr_slice.values)),
                                          bm_name=bm_name,
                                          node_name=node_name,
                                          chaos=torch.tensor(chaos_series_slice.tolist()),
                                          edge_index=edge_index.t().contiguous(),
                                          edge_attr=edge_attr,
                                          # will be removed later
                                          notna_mask=notna_mask)
                    data_list.append(data_obj)

        return data_list


class MinMaxScaler(PeronaBaseTransform):
    def __init__(self, target_property_name: str, target_min: float = 0., target_max: float = 1.):
        super().__init__()
        self.target_property_name = target_property_name
        self.target_min = target_min
        self.target_max = target_max
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(target_min, target_max))

    def fit(self, data: PeronaData):
        target_arr: np.ndarray = getattr(data, self.target_property_name).cpu().numpy()
        self.min_max_scaler.partial_fit(target_arr)
        return None

    def __call__(self, data: PeronaData):
        target_arr: np.ndarray = getattr(data, self.target_property_name).cpu().numpy()
        setattr(data, self.target_property_name, torch.from_numpy(self.min_max_scaler.transform(target_arr)))
        return data


class GraphFinalizer(PeronaBaseTransform):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bm_types = set()
        self.norm_std_dict: Dict[str, List[float]] = {}

    def fit(self, data: PeronaData):
        dict_key: str = "_".join([data.bm_name, data.node_name])
        x_norm = torch.linalg.vector_norm(data.x[~data.chaos.to(torch.bool)][:, data.notna_mask],
                                          ord=GeneralConfig.vector_norm_ord, dim=-1)
        self.norm_std_dict[dict_key] = self.norm_std_dict.get(dict_key, []) + x_norm.tolist()

        self.bm_types.add(data.bm_name)
        return None

    def __call__(self, data: PeronaData):
        x_norm = torch.linalg.vector_norm(data.x[:, data.notna_mask], ord=GeneralConfig.vector_norm_ord, dim=-1)

        # all valid combinations and associated targets
        all_perms: List[Tuple[int, int]] = list(permutations(range(len(data.x)), 2))
        all_combs: torch.LongTensor = torch.tensor(all_perms).reshape(-1, 2).to(torch.long)
        all_targets: torch.Tensor = torch.sign(x_norm[all_combs[:, 0]] - x_norm[all_combs[:, 1]])
        all_mask: torch.BoolTensor = all_targets != 0
        all_combs = all_combs[all_mask]
        all_targets = all_targets[all_mask]

        # Does chaos exist? What is the id of the last node? What is the default return value?
        chaos_exists = bool(data.chaos.sum())
        max_node_id = torch.max(all_combs).item()
        fallback_tensor = torch.tensor([]).to(torch.long)

        # for all_normal:
        all_normal_mask = (all_combs[:, 0] < max_node_id) & (all_combs[:, 1] < max_node_id)
        all_normal_combs = all_combs[all_normal_mask] if chaos_exists else all_combs
        all_normal_targets = all_targets[all_normal_mask] if chaos_exists else all_targets

        # for all_chaos:
        all_chaos_mask = (all_combs[:, 0] == max_node_id) | (all_combs[:, 1] == max_node_id)
        all_chaos_combs = all_combs[all_chaos_mask] if chaos_exists else fallback_tensor
        all_chaos_targets = all_targets[all_chaos_mask] if chaos_exists else fallback_tensor

        # set new properties
        setattr(data, "ranking_indices_all_normal", all_normal_combs)
        setattr(data, "ranking_targets_all_normal", all_normal_targets)
        setattr(data, "ranking_indices_all_chaos", all_chaos_combs)
        setattr(data, "ranking_targets_all_chaos", all_chaos_targets)
        setattr(data, "ranking_margin", max([np.array(v_l).std() for v_l in self.norm_std_dict.values()]))
        setattr(data, "x_norm", x_norm)
        setattr(data, "input_dim", data.x.size(1))
        setattr(data, "edge_dim", data.edge_attr.size(1))
        setattr(data, "output_dim", len(self.bm_types))
        # delete no longer used property
        delattr(data, "notna_mask")

        return data
