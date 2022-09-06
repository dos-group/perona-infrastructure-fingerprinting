# ######################################
# HANDLERS FOR CHAOS JOBS ###
# ######################################
from typing import Optional

import kopf

from common import BaseChaos, chaos_cls_mappings
from common import ChaosStartupResult
from common import get_benchmark_history_client, BenchmarkHistoryClient
from common import K8sClient, get_k8s_client

k8s_client: K8sClient = get_k8s_client()
bm_history_client: BenchmarkHistoryClient = get_benchmark_history_client()


def benchmark_filter(meta: dict, **_):
    labels_dict: dict = meta.get("labels", {})
    perona_benchmark_id: Optional[str] = labels_dict.get("perona-benchmark-id", None)
    chaos_desired: bool = "True" == labels_dict.get("perona-benchmark-chaos-desired", "False")
    if any([perona_benchmark_id.startswith(key) for key in list(chaos_cls_mappings.keys())]) and chaos_desired:
        if perona_benchmark_id.startswith("network-") and "-client" in meta.get("name", ""):
            return True
        elif not perona_benchmark_id.startswith("network-"):
            return True
    return False


def pod_field_change_filter(meta, old, new, **_):
    old_cond: bool = (old or {}).get("status", {}).get("phase", None) != "Running"
    true_cond: bool = (new or {}).get("status", {}).get("phase", None) == "Running"
    return benchmark_filter(meta) and old_cond and true_cond


def chaos_field_change_filter(old, new, **_):
    old_cond: bool = (old or {}).get("status", {}).get("experiment", {}).get("desiredPhase", None) != "Stop"
    true_cond: bool = (new or {}).get("status", {}).get("experiment", {}).get("desiredPhase", None) == "Stop"
    return old_cond and true_cond


@kopf.on.field("pods", when=pod_field_change_filter, field="status.phase")
def chaosmesh_create(name: str, logger, body: dict, **_):
    spec, meta = body["spec"], body["metadata"]

    node_id: str = spec.get("nodeName", "")

    labels_dict: dict = meta.get("labels", {})
    perona_benchmark_id: Optional[str] = labels_dict.get("perona-benchmark-id", None)
    perona_benchmark_resource_kind: Optional[str] = labels_dict.get("perona-benchmark-resource-kind", None)
    perona_suffix: Optional[str] = labels_dict.get("perona-suffix", None)

    if all([var is not None for var in [perona_benchmark_id, perona_benchmark_resource_kind, perona_suffix]]):
        chaos_inst = chaos_cls_mappings[perona_benchmark_resource_kind]()
        perona_kwargs: dict = {"perona_suffix": perona_suffix, "perona_benchmark_id": perona_benchmark_id}
        chaos_f_resp: ChaosStartupResult = chaos_inst.run(k8s_client.api, node_id,
                                                          perona_benchmark_id, perona_kwargs=perona_kwargs)
        if chaos_f_resp.success:
            chaos_id: str = chaos_f_resp.id
            logger.info(f"Successfully created chaos-job '{chaos_id}' for '{perona_benchmark_id}'")
        else:
            logger.error(f"Error scheduling chaos for {name}: {chaos_f_resp.error}")
