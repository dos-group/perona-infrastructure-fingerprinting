import asyncio
import random
import time
from typing import List, Optional, Tuple, Coroutine

import pykube
from fastapi import FastAPI, HTTPException, responses, Depends, BackgroundTasks, status
from pykube import Pod

from bm_api.models.experiment import ExperimentModel
from fastapi.middleware.cors import CORSMiddleware

from common import benchmark_cls_mappings, BaseRun
from common import get_benchmark_history_client
from common import BenchmarkHistoryClient

from common import get_k8s_client
from common import K8sClient
from bm_api.models.benchmark import BenchmarkResult, BenchmarkResultMetric
from bm_api.models.benchmark import BenchmarkResult

import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/version")
async def get_version(k8s_client: K8sClient = Depends(get_k8s_client)):
    return {
        "version": k8s_client.api.version
    }


@app.get("/", include_in_schema=False)
async def redirect():
    response = responses.RedirectResponse(url='/docs')
    return response


@app.get("/benchmarks/name={bm_name}/results", response_model=Optional[BenchmarkResult])
async def get_benchmark_results(bm_name: str,
                                bm_history_client: BenchmarkHistoryClient = Depends(get_benchmark_history_client)):
    try:
        r = bm_history_client.get_benchmark_result(bm_name)
        if r is not None:
            return BenchmarkResult(
                id=r.id,
                type=r.type,
                resource=r.type,
                started=r.started,
                metrics=[BenchmarkResultMetric(name=m.name, value=m.value, unit=m.unit) for m in r.metrics]
            )
        else:
            return r
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=404, detail=f"Benchmark '{bm_name}' not found")


@app.get("/benchmarks/node={node_name}/results", response_model=List[BenchmarkResult])
async def get_benchmark_results_for_node(node_name: str, bm_history_client: BenchmarkHistoryClient = Depends(
    get_benchmark_history_client)):
    try:
        results = bm_history_client.get_benchmarks_results(node_name)
        bm_list = [
            BenchmarkResult(
                id=r.id,
                type=r.type,
                resource=r.type,
                started=r.started,
                metrics=[BenchmarkResultMetric(name=m.name, value=m.value, unit=m.unit) for m in r.metrics]
            ) for r in results
        ]

        return bm_list
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=404, detail=f"Benchmarks for node '{node_name}' not found")


@app.post("/benchmark/{bm_type}/{node_id}")
async def run_benchmark(bm_type: str, node_id: str,
                        k8s_client: K8sClient = Depends(get_k8s_client), perona_kwargs: Optional[dict] = None):
    if perona_kwargs is None:
        perona_kwargs = {}
    bm_type = bm_type.lower()
    if bm_type in benchmark_cls_mappings:
        bm_cls = benchmark_cls_mappings[bm_type]
        bm = bm_cls()

        startup_result = bm.run(k8s_client.api, node_id, perona_kwargs=perona_kwargs)

        if startup_result.success:
            return {
                "id": startup_result.id,
                "spec": startup_result.benchmark_spec
            }
        else:
            raise HTTPException(status_code=500,
                                detail=f"Failed to start benchmark '{bm_type}': {startup_result.error}")
    else:
        raise HTTPException(status_code=404, detail=f"Benchmark '{bm_type}' not found")


async def handle_parallel_runs(parallel_tuple_runs: List[Tuple[str, str, bool, str]],
                               k8s_client: K8sClient, bm_history_client: BenchmarkHistoryClient, message: str = None):
    if message is not None:
        logging.info(f"{'#' * 40} {message} {'#' * 40}")

    to_observe: List[Tuple[str, bool]] = []
    for bm_type, node_id, chaos_desired, suffix in parallel_tuple_runs:
        perona_benchmark_id: str = f"{bm_type}-{suffix}"
        try:
            if not bm_history_client.benchmark_id_exists(perona_benchmark_id):
                perona_kwargs: dict = {
                    "perona_suffix": suffix,
                    "perona_benchmark_id": perona_benchmark_id,
                    "perona_benchmark_chaos_desired": str(chaos_desired)
                }
                to_observe.append((perona_benchmark_id, False))
                await run_benchmark(bm_type, node_id, k8s_client=k8s_client, perona_kwargs=perona_kwargs)
            else:
                logging.info(f"Skip {perona_benchmark_id}, already exists in DB...")
        except HTTPException as http_exc:
            logging.error(http_exc)

    skip_all: bool = len(to_observe) == 0
    while len(to_observe):
        time.sleep(10)
        pods: List[Pod] = pykube.Pod.objects(k8s_client.api, namespace="kubestone").all()
        for benchmark_id, has_been_seen in to_observe:
            if any([benchmark_id in pod.name for pod in pods]) and not has_been_seen:
                to_observe.remove((benchmark_id, has_been_seen))
                to_observe.append((benchmark_id, True))
            elif not any([benchmark_id in pod.name for pod in pods]) and has_been_seen:
                to_observe.remove((benchmark_id, has_been_seen))
            else:
                pass  # wait for next loop iteration

    if not skip_all:
        logging.info("Wait 60s before next benchmark executions so that node metrics can normalize...")
        time.sleep(60)


async def handle_parallel_tasks(list_of_tuple_lists: List[List[Tuple[str, str, bool, str]]],
                                k8s_client: K8sClient, bm_history_client: BenchmarkHistoryClient):
    global_len: int = len(list_of_tuple_lists[0])
    coroutines: List[Coroutine] = []
    max_parallel_network: List[int] = []
    for idx, parallel_tuple_runs in enumerate(zip(*list_of_tuple_lists)):
        max_parallel_network.append(len([t for t in parallel_tuple_runs if "network" in t[0]]))
        coroutines.append(handle_parallel_runs(parallel_tuple_runs, k8s_client, bm_history_client,
                                               message=f"Parallel Tuple Runs: {idx + 1} / {global_len}"))
    logging.info(f"Max parallel network runs: {max(max_parallel_network)}")
    await asyncio.gather(*coroutines)


@app.post("/experiment/schedule")
async def schedule_experiment_runs(exp_model: ExperimentModel,
                                   background_tasks: BackgroundTasks,
                                   k8s_client: K8sClient = Depends(get_k8s_client),
                                   bm_history_client: BenchmarkHistoryClient = Depends(get_benchmark_history_client)):
    num_network_bms: int = len([el for el in exp_model.bm_types if "network" in el])
    if num_network_bms > 0 and ((len(exp_model.bm_types) // num_network_bms) < len(exp_model.node_ids)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Can not distribute network benchmarks!")

    random.seed(exp_model.random_s)

    bm_list: List[Tuple[str, bool]] = sum([[(bm_type, bool(i < exp_model.num_anom))
                                            for i in range(exp_model.num_each)] for bm_type in exp_model.bm_types], [])
    list_of_tuple_lists: List[List[Tuple[str, str, bool, str]]] = []
    for idx, node_id in enumerate(exp_model.node_ids):
        bm_list_copy = list(bm_list)
        random.shuffle(bm_list_copy)
        suffixes: List[str] = [BaseRun.generate_suffix() for _ in range(len(bm_list_copy))]
        tuple_list: List[Tuple[str, str, bool, str]] = [(bm_type, node_id, chaos_desired, suffix)
                                                        for (bm_type, chaos_desired), suffix in
                                                        zip(bm_list_copy, suffixes)]
        if idx == 0:
            list_of_tuple_lists.append(tuple_list)
        else:
            l_net_ind: List[int] = [i for i, t in enumerate(tuple_list) if "network" in t[0]]
            l_res_ind: List[int] = [i for i, t in enumerate(tuple_list) if "network" not in t[0]]
            r_net_indices = sum([[i for i, t in enumerate(ll) if "network" in t[0]] for ll in list_of_tuple_lists], [])
            mod_tuple_list: List[Optional[Tuple[str, str, bool, str]]] = [None] * len(tuple_list)
            for i in range(len(tuple_list)):
                target_idx: Optional[int] = None
                if i not in r_net_indices and len(l_net_ind):
                    target_idx = l_net_ind.pop(0)
                elif len(l_res_ind):
                    target_idx = l_res_ind.pop(0)
                if target_idx is not None:
                    mod_tuple_list[i] = tuple_list[target_idx]
            list_of_tuple_lists.append(mod_tuple_list)

    val_log_list = [[tup[:-1] for tup in slist] for slist in list_of_tuple_lists]
    for e in sorted(sum([[[x, ll.count(x)] for x in set(ll)] for ll in val_log_list], []), key=lambda t: t[0]):
        logging.info(e)
    background_tasks.add_task(handle_parallel_tasks, list_of_tuple_lists, k8s_client, bm_history_client)
    return {"message": "Bechmark runs of this experiment sent in the background"}


@app.get("/experiment/task_count",
         name="Check the number of still running / outstanding tasks.",
         status_code=status.HTTP_200_OK)
async def check_task_count(background_tasks: BackgroundTasks):
    return len(background_tasks.tasks)
