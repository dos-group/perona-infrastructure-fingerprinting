from .base import BaseBenchmark, BenchmarkedResourceKind


class MemorySysbenchBenchmark(BaseBenchmark):
    @property
    def kind(self):
        return "Sysbench"

    @property
    def resource_kind(self) -> BenchmarkedResourceKind:
        return BenchmarkedResourceKind.MEMORY_SYSBENCH

    @property
    def config_path(self):
        return "config/benchmarks/memory_sysbench.yaml"
