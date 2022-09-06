from typing import Type

from .base import BaseBenchmark, BenchmarkedResource
from .cpu import *
from .memory import *
from .network import *
from .disk import *

benchmark_cls_mappings: Dict[str, Type[BaseBenchmark]] = {
    "cpu-sysbench": CpuSysbenchBenchmark,
    "memory-sysbench": MemorySysbenchBenchmark,
    "network-iperf3": NetworkIperf3Benchmark,
    "network-qperf": NetworkQperfBenchmark,
    "disk-ioping": DiskIopingBenchmark,
    "disk-fio": DiskFioBenchmark
}
