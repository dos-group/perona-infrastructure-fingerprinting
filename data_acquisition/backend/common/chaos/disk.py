from .base import BaseChaos, StressedResourceKind


class DiskChaos(BaseChaos):
    @property
    def kind(self):
        return "IOChaos"

    @property
    def resource_kind(self) -> StressedResourceKind:
        return StressedResourceKind.DISK_CHAOS

    @property
    def config_path(self):
        return "config/chaos/disk_chaos.yaml"
