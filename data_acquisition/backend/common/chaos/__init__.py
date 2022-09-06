from typing import Dict, Type

from .base import BaseChaos, ChaosStartupResult
from .cpu import *
from .disk import *
from .memory import *
from .network import *

chaos_cls_mappings: Dict[str, Type[BaseChaos]] = {
    "cpu-sysbench": CpuChaos,
    "memory-sysbench": MemoryChaos,
    "network-iperf3": NetworkChaos,
    "network-qperf": NetworkChaos,
    "disk-ioping": DiskChaos,
    "disk-fio": DiskChaos
}
