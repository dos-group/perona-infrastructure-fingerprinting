from .base import BaseChaos, StressedResourceKind


class NetworkChaos(BaseChaos):
    @property
    def kind(self):
        return "NetworkChaos"

    @property
    def resource_kind(self) -> StressedResourceKind:
        return StressedResourceKind.NETWORK_CHAOS

    @property
    def config_path(self):
        return "config/chaos/network_chaos.yaml"

    def get_label_selector(self, benchmark_id: str) -> dict:
        return {"iperf-mode": "client"} if "iperf" in benchmark_id else {"qperf-mode": "client"}
