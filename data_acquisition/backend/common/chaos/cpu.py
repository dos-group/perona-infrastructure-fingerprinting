from .base import BaseChaos, StressedResourceKind


class CpuChaos(BaseChaos):
    @property
    def kind(self):
        return "StressChaos"

    @property
    def resource_kind(self) -> StressedResourceKind:
        return StressedResourceKind.CPU_CHAOS

    @property
    def config_path(self):
        return "config/chaos/cpu_chaos.yaml"
