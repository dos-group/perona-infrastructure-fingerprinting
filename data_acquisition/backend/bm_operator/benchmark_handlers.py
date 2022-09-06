# ######################################
# HANDLERS FOR BENCHMARK JOBS ###
# ######################################
import datetime
import re
from typing import Dict, List, Optional, Tuple

import kopf
import pykube
from pykube.objects import Pod
from sqlalchemy.orm import Session

from common import BenchmarkedResourceKind, benchmark_cls_mappings, BaseBenchmark, BenchmarkHistoryClient, \
    get_benchmark_history_client, BaseChaos, chaos_cls_mappings
from common import K8sClient, get_k8s_client
from common import get_benchmark_metrics, metrics_cls_mapping
from common.metrics.common import BMMetricField
from orm import engine
from orm.models import Benchmark, BenchmarkMetric

k8s_client: K8sClient = get_k8s_client()
bm_history_client: BenchmarkHistoryClient = get_benchmark_history_client()


def parse_bm_value(s_val: Optional[str], def_value: float = 0.0,
                   def_unit: str = "scalar") -> Tuple[Optional[float], float, str]:
    value = def_value
    unit = def_unit

    if s_val is not None:
        rx = re.match(r"([\d\.,]+)(.*)", s_val, re.IGNORECASE)

        s_val = s_val.strip("").strip(",")

        try:
            value, unit = float(rx.group(1).strip(",").replace(',', '')), (rx.group(2) or def_unit).strip()
        except:
            unit = "(error)"

    return s_val, value, unit


# Extracts metrics from a metrics container and converts them to BenchmarkMetric
# entries.
def to_metrics_list(benchmark_id: str, o: any) -> List[BenchmarkMetric]:
    metric_fields: Dict[str, BMMetricField] = {
        k: v for k, v in vars(o).items() if isinstance(v, BMMetricField)
    }

    result = []

    for k, v in metric_fields.items():
        mod_value, p_value, p_unit = parse_bm_value(v.value)
        result.append(BenchmarkMetric(
            benchmark_id=benchmark_id,
            name=k,
            text_value=mod_value,
            value=p_value,
            unit=p_unit
        ))

    return result


def benchmark_filter(meta, **_):
    bm_name: str = meta.get("name")
    return any([bm_name.startswith(str(e.value)) for e in BenchmarkedResourceKind])


def benchmark_field_change_filter(meta, old, new, **_):
    old_cond: bool = (old or {}).get("status", {}).get("completed", False) is not True
    true_cond: bool = (new or {}).get("status", {}).get("completed", False) is True
    return benchmark_filter(meta) and old_cond and true_cond


@kopf.on.field("perf.kubestone.xridge.io", "v1alpha1", kopf.EVERYTHING,
               when=benchmark_field_change_filter, field="status.completed")
def kubestone_complete(namespace: str, name: str, logger: any, **_):

    metrics_cls = next((clazz for variant, clazz in metrics_cls_mapping.items() if name.startswith(variant)))
    benchmark_cls = next((clazz for variant, clazz in benchmark_cls_mappings.items() if name.startswith(variant)))
    chaos_cls = next((clazz for variant, clazz in chaos_cls_mappings.items() if name.startswith(variant)))

    selector: dict = {"perona-benchmark-id": name}
    if all([w in name for w in ["network", "iperf"]]):
        selector["iperf-mode"] = "client"
    if all([w in name for w in ["network", "qperf"]]):
        selector["qperf-mode"] = "client"

    pod: Optional[Pod] = pykube.Pod.objects(k8s_client.api, namespace=namespace).get_or_none(selector=selector)

    if pod is not None:
        container_statuses = pod.obj.get("status", {}).get("containerStatuses", [])

        if not any(container_statuses):
            logger.error(f"{name}: No container status found for pod {pod.name}")

        container_start = container_statuses[0].get("state").get("terminated").get("startedAt")
        container_termination = container_statuses[0].get("state").get("terminated").get("finishedAt")

        logs_str, bm_values = get_benchmark_metrics(metrics_cls, pod)

        with Session(engine) as session:

            bm_metric_list: List[BenchmarkMetric] = to_metrics_list(name, bm_values)
            f_bm_metric_list: List[BenchmarkMetric] = [el for el in bm_metric_list if el.text_value is not None]

            logger.info(f"{name}: {len(f_bm_metric_list)}/{len(bm_metric_list)} Non-Null Metrics")

            bm = Benchmark(
                id=name,
                type=pod.labels.get("perona-benchmark-resource-kind", "unknown"),
                chaos_desired="True" == pod.labels.get("perona-benchmark-chaos-desired", "False"),
                node_id=pod.obj["spec"]["nodeName"],
                pod_id=pod.name,
                started=datetime.datetime.strptime(container_start, "%Y-%m-%dT%H:%M:%S%z"),
                finished=datetime.datetime.strptime(container_termination, "%Y-%m-%dT%H:%M:%S%z"),
                metric_collection_status=f"{len(f_bm_metric_list)}/{len(bm_metric_list)}",
                image=pod.obj["spec"]["containers"][0]["image"],
                options=' '.join(pod.obj["spec"]["containers"][0]["args"]),
                logs=logs_str
            )

            session.merge(bm)

            for bm_metric in bm_metric_list:
                session.merge(bm_metric)

            session.commit()

    # finally: delete benchmark object
    benchmark_cls_instance: BaseBenchmark = benchmark_cls()
    bm_factory_instance, job_obj = benchmark_cls_instance.get_obj_for_selector(k8s_client, {"name": name},
                                                                               benchmark_cls_instance.kind, namespace)
    if job_obj is not None:
        bm_factory_instance(k8s_client.api, job_obj).delete()

    # next: delete chaos object, if any
    chaos_cls_instance: BaseChaos = chaos_cls()
    _, chaos_obj = chaos_cls_instance.get_obj_for_selector(k8s_client,
                                                           {"selector": {"perona-benchmark-id": name}},
                                                           chaos_cls_instance.kind, namespace)
    if chaos_obj is not None:
        chaos_id: str = chaos_obj["metadata"]["name"]
        try:
            logger.info(f"Deleting now chaos-job '{chaos_id}'...")
            BaseChaos.delete_chaos_object(k8s_client, bm_history_client, chaos_id, chaos_cls_instance.kind, name)
        except Exception as e:
            logger.error(f"Could not delete chaos-job '{chaos_id}', error={e}")
