from typing import Dict

import pykube

from .base import BaseBenchmark, BenchmarkStartupResult, BenchmarkedResourceKind


class NetworkIperf3Benchmark(BaseBenchmark):
    @property
    def kind(self):
        return "Iperf3"

    @property
    def resource_kind(self) -> BenchmarkedResourceKind:
        return BenchmarkedResourceKind.NETWORK_IPERF3

    @property
    def config_path(self):
        return "config/benchmarks/network_iperf3.yaml"

    @property
    def name(self):
        return BenchmarkedResourceKind.NETWORK_IPERF3.value

    def _run(self, client: pykube.HTTPClient, spec: Dict,
             node_name: str, perona_kwargs: dict) -> BenchmarkStartupResult:
        client_node_name, server_node_name = (node_name.split("@@@") + [None, None])[:2]
        spec = self.merge_dicts(spec, {"spec": {
            "clientConfiguration": {"podScheduling": {"nodeName": client_node_name,
                                                      "nodeSelector": {"mynodetype": "worker"}},
                                    "podLabels": {k.replace("_", "-"): v for k, v in perona_kwargs.items()}},
            "serverConfiguration": {"podScheduling": {"nodeName": server_node_name,
                                                      "nodeSelector": {"mynodetype": "supporter"}},
                                    "podLabels": {k.replace("_", "-"): v for k, v in perona_kwargs.items()}}
        }})
        self.get_factory(client, self.kind)(client, spec).create()
        return BenchmarkStartupResult(success=True, id=spec['metadata']['name'], benchmark_spec=spec)


class NetworkQperfBenchmark(NetworkIperf3Benchmark):
    @property
    def kind(self):
        return "Qperf"

    @property
    def resource_kind(self) -> BenchmarkedResourceKind:
        return BenchmarkedResourceKind.NETWORK_QPERF

    @property
    def config_path(self):
        return "config/benchmarks/network_qperf.yaml"
