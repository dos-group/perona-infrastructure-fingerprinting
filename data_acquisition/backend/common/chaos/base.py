import dataclasses
import enum
import os
from abc import ABC
from typing import Type, Optional, Dict, List

import pykube
import yaml
from pykube.objects import APIObject

from ..shared import BaseRun
from ..clients import BenchmarkHistoryClient, K8sClient


@dataclasses.dataclass
class ChaosStartupResult:
    success: bool
    id: Optional[str] = None
    chaos_spec: Optional[Dict] = None
    error: Optional[str] = None


class StressedResourceKind(enum.Enum):
    CPU_CHAOS = "cpu-chaos"
    MEMORY_CHAOS = "memory-chaos"
    DISK_CHAOS = "disk-chaos"
    NETWORK_CHAOS = "network-chaos"


class BaseChaos(BaseRun, ABC):
    @property
    def config_path(self):
        raise NotImplementedError

    @property
    def resource_kind(self) -> StressedResourceKind:
        raise NotImplementedError

    def get_label_selector(self, benchmark_id: str) -> dict:
        return {}

    @property
    def name(self):
        return self.resource_kind.value

    @classmethod
    def delete_chaos_object(cls, k8s_client: K8sClient, bm_history_client: BenchmarkHistoryClient,
                            chaos_obj_name: str, chaos_kind: str, bm_obj_name: str):
        factory_instance, chaos_obj = cls.get_obj_for_selector(k8s_client, {"name": chaos_obj_name}, chaos_kind)
        if chaos_obj is not None:
            factory_instance(k8s_client.api, chaos_obj).delete()
            cls.update_benchmark_db_entry(bm_history_client, chaos_obj, bm_obj_name)

    @classmethod
    def update_benchmark_db_entry(cls, bm_history_client: BenchmarkHistoryClient, chaos_obj: dict, bm_obj_name: str):
        chao_obj_name: str = chaos_obj.get("metadata", {}).get("name", "")
        conditions: List[dict] = chaos_obj.get("status", {}).get("conditions", [])
        container_records: List[dict] = chaos_obj.get("status", {}).get("experiment", {}).get("containerRecords", [])
        chaos_applied: bool = False
        if any([bm_obj_name in obj["id"] for obj in container_records]) and \
                any([cond["status"] == "True" and
                     cond["type"] in ["AllInjected", "AllRecovered"] for cond in conditions]):
            chaos_applied = True
        bm_history_client.update_benchmark_object(bm_obj_name,
                                                  {'chaos_applied': chaos_applied, "chaos_id": chao_obj_name})

    @classmethod
    def get_factory(cls, client: pykube.HTTPClient, kind: str) -> Type[APIObject]:
        # use object factory:
        # - all chaos-mesh api_version = 'chaos-mesh.org/v1alpha1'
        # - specify 'kind', e.g. 'StressChaos'
        return pykube.object_factory(client, "chaos-mesh.org/v1alpha1", kind)

    def run(self, client: pykube.HTTPClient, node_name: str, perona_benchmark_id: str,
            perona_kwargs: Optional[dict] = None) -> ChaosStartupResult:
        if perona_kwargs is None:
            perona_kwargs = {}
        perona_kwargs["perona_suffix"] = perona_kwargs.get("perona_suffix", None) or self.generate_suffix()
        perona_kwargs["perona_chaos_k8s_kind"] = self.kind
        perona_kwargs["perona_chaos_resource_kind"] = self.resource_kind.value
        perona_kwargs["perona_chaos_id"] = "-".join([perona_kwargs[k]
                                                     for k in ["perona_chaos_resource_kind", "perona_suffix"]])

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    spec = yaml.safe_load(f)
                    # make sure to execute in 'kubestone' namespace
                    spec = self.merge_dicts(spec, {"metadata": {"namespace": "kubestone"}})
                    # add suffix to 'name', so that they are different and
                    # one can schedule multiple chaos "objects" of one kind simultaneously
                    spec['metadata']['name'] = perona_kwargs["perona_chaos_id"]
                    spec['metadata']['labels'] = {
                        **{k.replace("_", "-"): v for k, v in perona_kwargs.items()},
                        **spec['metadata'].get("labels", {})
                    }
                    # now: run custom logic
                    real_node_name: str = node_name.split("@@@")[0]
                    spec = self.merge_dicts(spec, {"spec": {"selector": {
                        "nodeSelectors": {"mynodetype": "worker"},
                        "nodes": [real_node_name],
                        "namespaces": ["kubestone"],
                        "labelSelectors": {
                            **{"perona-benchmark-id": perona_benchmark_id},
                            **self.get_label_selector(perona_benchmark_id)}
                    }}})
                    self.get_factory(client, self.kind)(client, spec).create()
                    return ChaosStartupResult(success=True, id=spec['metadata']['name'], chaos_spec=spec)
            except Exception as e:
                return ChaosStartupResult(success=False, error=str(e))
        else:
            return ChaosStartupResult(success=False, error="Chaos specification config file could not be found")
