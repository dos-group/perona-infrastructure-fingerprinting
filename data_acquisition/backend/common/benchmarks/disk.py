from .base import BaseBenchmark, BenchmarkedResourceKind


class DiskIopingBenchmark(BaseBenchmark):
    @property
    def kind(self):
        return "Ioping"

    @property
    def resource_kind(self) -> BenchmarkedResourceKind:
        return BenchmarkedResourceKind.DISK_IOPING

    @property
    def config_path(self):
        return "config/benchmarks/disk_ioping.yaml"

    @property
    def name(self):
        return BenchmarkedResourceKind.DISK_IOPING.value


class DiskFioBenchmark(BaseBenchmark):
    @property
    def kind(self):
        return "Fio"

    @property
    def resource_kind(self) -> BenchmarkedResourceKind:
        return BenchmarkedResourceKind.DISK_FIO

    @property
    def config_path(self):
        return "config/benchmarks/disk_fio.yaml"
