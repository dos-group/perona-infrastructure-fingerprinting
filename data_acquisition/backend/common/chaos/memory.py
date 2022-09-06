from .base import BaseChaos, StressedResourceKind


class MemoryChaos(BaseChaos):
    @property
    def kind(self):
        return "StressChaos"

    @property
    def resource_kind(self) -> StressedResourceKind:
        return StressedResourceKind.MEMORY_CHAOS

    @property
    def config_path(self):
        return "config/chaos/memory_chaos.yaml"
