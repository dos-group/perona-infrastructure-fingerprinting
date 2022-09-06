import asyncio
import contextlib
import os
import threading
import argparse
import datetime
from typing import Optional, Any

import kopf
import uvicorn
import uvloop
from uvloop.loop import Loop
from sqlalchemy.orm import Session

import orm
from bm_api import app
from common import get_prometheus_client
from orm.models import NodeMetric


# this thread will start kopf, i.e. our custom kubernetes operator that listens to kubernetes events, acts accordingly
def kopf_thread(stop_me: threading.Event) -> None:
    try:
        # import needed to active kopf operator
        import bm_operator

        kopf_loop = uvloop.new_event_loop()
        asyncio.set_event_loop(kopf_loop)

        with contextlib.closing(kopf_loop):
            kopf.configure(verbose=True)
            # stop-flag for graceful termination
            kopf_loop.run_until_complete(kopf.operator(stop_flag=stop_me, namespace="kubestone"))
    finally:
        stop_me.set()


# this thread will start a fastapi-backend, which can process requests from the frontend
def api_thread(stop_me: threading.Event, host: str, port: int) -> None:
    api_loop: Optional[Loop] = None
    pending: Any = None
    try:
        api_loop = uvloop.new_event_loop()
        asyncio.set_event_loop(api_loop)

        # monitor the flag and stop it somehow. here, disgracefully.
        with contextlib.closing(api_loop):
            config = uvicorn.Config(app=app, loop=api_loop, host=host, port=port)
            server = uvicorn.Server(config)
            server_task = asyncio.gather(server.serve())
            waiter_task = asyncio.gather(api_loop.run_in_executor(None, stop_me.wait))
            done, pending = api_loop.run_until_complete(
                asyncio.wait({server_task, waiter_task}, return_when=asyncio.FIRST_COMPLETED))
    finally:
        stop_me.set()

        if pending is not None:
            for task in pending:
                task.cancel()
        if api_loop is not None and pending is not None:
            api_loop.run_until_complete(asyncio.gather(pending))


# this thread will persist prometheus metrics
async def prometheus_thread(stop_me: threading.Event):
    pm_client = get_prometheus_client()

    try:
        interval = datetime.timedelta(seconds=5)

        prev_end = datetime.datetime.now() - interval

        while not stop_me.is_set():
            now = datetime.datetime.now()

            metrics = await pm_client.get_node_metrics(prev_end, now)

            with Session(orm.engine) as db_session:
                for node_metrics_obj in metrics:
                    node_name = node_metrics_obj.node_name
                    node_metric_groups = {k.lower(): v for k, v in
                                          node_metrics_obj.dict(exclude={"node_name", "join_node_info"}).items()}

                    for metric_group, metric_entries in node_metric_groups.items():
                        for metric_entry in metric_entries:
                            nm = NodeMetric(
                                node_name=node_name,
                                metric=metric_group,
                                timestamp=metric_entry["time"],
                                value=metric_entry["value"]
                            )

                            db_session.merge(nm)

                db_session.commit()

            prev_end = now
            await asyncio.sleep(interval.total_seconds())
    except Exception as e:
        print(e)
    finally:
        stop_me.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmarking Framework HTTP API server")
    parser.add_argument("--host", dest="host", action="store", type=str, default="0.0.0.0")
    parser.add_argument("--port", dest="port", action="store", type=int,
                        default=int(os.environ.get("PERONA_API_PORT", 8080)))

    cli_namespace = parser.parse_args()

    api_host = cli_namespace.host
    api_port = cli_namespace.port

    orm.create_tables()

    stop_me_event: threading.Event = threading.Event()
    t_kopf: threading.Thread = threading.Thread(target=kopf_thread, args=(stop_me_event,))
    t_api: threading.Thread = threading.Thread(target=api_thread, args=(stop_me_event,), kwargs={
        "host": api_host,
        "port": api_port
    })
    t_prometheus: threading.Thread = threading.Thread(target=asyncio.run, args=(prometheus_thread(stop_me_event),))

    t_kopf.start()
    t_api.start()
    t_prometheus.start()

    try:
        t_api.join()
        t_kopf.join()
        t_prometheus.join()
    except KeyboardInterrupt:
        stop_me_event.set()

        t_api.join()
        t_kopf.join()
        t_prometheus.join()
    finally:
        print("benchmarking-framework: Backend has shut down.")
