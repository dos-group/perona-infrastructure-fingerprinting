from typing import List, Dict
import numpy as np
from pydantic import BaseModel
from scipy import integrate
from classes.raw_workload import GenericWorkloadModel


class DataStatisticsModel(BaseModel):
    count: int
    min: float
    max: float
    mean: float
    auc_ecdf: float
    percentile_dict: Dict[int, float]

    @classmethod
    def construct_statistics(cls, values: List[float]):
        def calculate_auc_ecdf(x):
            hist, bin_edges = np.histogram(x, bins=100, range=(0, 100), density=True)
            ecdf = np.cumsum(hist) * np.amax(np.diff(bin_edges))
            return integrate.simpson(ecdf)

        percentiles = np.arange(0, 101, 5).astype(int)

        return cls(
            count=len(values),
            min=np.amin(values),
            max=np.amax(values),
            mean=np.mean(values),
            auc_ecdf=calculate_auc_ecdf(values),
            percentile_dict={k: v for k, v in zip(percentiles, np.percentile(values, percentiles))}
        )


class ProcessedWorkloadModel(GenericWorkloadModel):
    arrow_metrics: List[float] = []
    metrics: Dict[str, DataStatisticsModel] = {}