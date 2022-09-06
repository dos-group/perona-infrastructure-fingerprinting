from __future__ import annotations

import dataclasses
import enum
import os.path
from abc import ABC
from typing import Optional, Type, Dict

import pykube
import yaml
from pykube.objects import APIObject

from ..shared import BaseRun


class BenchmarkedResourceKind(enum.Enum):
    CPU_SYSBENCH = "cpu-sysbench"
    DISK_FIO = "disk-fio"
    DISK_IOPING = "disk-ioping"
    MEMORY_SYSBENCH = "memory-sysbench"
    NETWORK_IPERF3 = "network-iperf3"
    NETWORK_QPERF = "network-qperf"


class BenchmarkedResource(enum.Enum):
    CPU = "cpu"
    DISK = "disk"
    MEMORY = "memory"
    NETWORK = "network"


class BaseBenchmark(BaseRun, ABC):
    @property
    def config_path(self):
        raise NotImplementedError

    @property
    def resource_kind(self) -> BenchmarkedResourceKind:
        raise NotImplementedError

    @property
    def name(self):
        return self.resource_kind.value

    @classmethod
    def get_factory(cls, client: pykube.HTTPClient, kind: str) -> Type[APIObject]:
        # use object factory:
        # - all kubestone benchmarks use api_version = 'perf.kubestone.xridge.io/v1alpha1'
        # - specify 'kind', e.g. 'Sysbench'
        return pykube.object_factory(client, "perf.kubestone.xridge.io/v1alpha1", kind)

    def run(self, client: pykube.HTTPClient, node_name: str,
            perona_kwargs: Optional[dict] = None) -> BenchmarkStartupResult:
        if perona_kwargs is None:
            perona_kwargs = {}
        perona_kwargs["perona_suffix"] = perona_kwargs.get("perona_suffix", None) or self.generate_suffix()
        perona_kwargs["perona_benchmark_k8s_kind"] = self.kind
        perona_kwargs["perona_benchmark_resource_kind"] = self.resource_kind.value
        perona_kwargs["perona_benchmark_id"] = "-".join([perona_kwargs[k]
                                                         for k in ["perona_benchmark_resource_kind", "perona_suffix"]])

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    spec = yaml.safe_load(f)
                    # make sure to execute in 'kubestone' namespace
                    spec = self.merge_dicts(spec, {"metadata": {"namespace": "kubestone"}})
                    # add suffix to 'name', so that they are different and
                    # one can schedule multiple benchmarks of one kind simultaneously
                    spec['metadata']['name'] = perona_kwargs["perona_benchmark_id"]
                    # now: run custom logic
                    return self._run(client, spec, node_name, perona_kwargs)
            except Exception as e:
                return BenchmarkStartupResult(success=False, error=str(e))
        else:
            return BenchmarkStartupResult(success=False, error="Job specification config file could not be found")

    def _run(self, client: pykube.HTTPClient, spec: Dict,
             node_name: str, perona_kwargs: dict) -> BenchmarkStartupResult:
        real_node_name: str = node_name.split("@@@")[0]
        spec = self.merge_dicts(spec, {"spec": {"podConfig": {
            "podScheduling": {"nodeName": real_node_name, "nodeSelector": {"mynodetype": "worker"}},
            "podLabels": {k.replace("_", "-"): v for k, v in perona_kwargs.items()}}}})
        self.get_factory(client, self.kind)(client, spec).create()
        return BenchmarkStartupResult(success=True, id=spec['metadata']['name'], benchmark_spec=spec)


@dataclasses.dataclass
class BenchmarkStartupResult:
    success: bool
    id: Optional[str] = None
    benchmark_spec: Optional[Dict] = None
    error: Optional[str] = None
