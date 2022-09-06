from typing import Type, Dict

from pykube import Pod
import re

from .common import TMetricClass, read_benchmark_metrics
from .cpu import SysbenchCpuMetrics
from .memory import SysbenchMemoryMetrics
from .network import NetworkIperf3Metrics, QpervesMetrics
from .disk import FiosMetrics, IopingsMetrics


def get_benchmark_metrics(cls: Type[TMetricClass], pod: Pod) -> TMetricClass:
    lgs_str = pod.logs()
    if "disk-fio" in pod.labels.get("perona-benchmark-id", ""):  # dirty hack
        lgs_str = re.sub(r",\n\s*\|", r",", lgs_str)
        lgs_str = re.sub(r":\n\s*\|", r":", lgs_str)
    values = read_benchmark_metrics(cls, lgs_str.splitlines(keepends=False))
    return lgs_str, values


metrics_cls_mapping: Dict[str, any] = {
    "cpu-sysbench": SysbenchCpuMetrics,
    "memory-sysbench": SysbenchMemoryMetrics,
    "disk-fio": FiosMetrics,
    "disk-ioping": IopingsMetrics,
    "network-iperf3": NetworkIperf3Metrics,
    "network-qperf": QpervesMetrics
}