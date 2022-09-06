import asyncio
import json
import logging
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import aiohttp
from aiohttp import ClientSession

from .schemes import PrometheusApiResponseModel, NodeMetricsModel, DataRecordingModel
from .settings import PrometheusSettings

prometheus_settings: PrometheusSettings = PrometheusSettings()


class PrometheusClient:

    @staticmethod
    async def __fetch__(url: str, session: ClientSession, item_tuple: Tuple[str, dict]):
        metric_name, query_specs = item_tuple
        try:
            async with session.get(url=url, params=query_specs) as response:
                res: dict = json.loads((await response.read()).decode('utf-8'))
                return metric_name, PrometheusApiResponseModel(**res)
        except BaseException as exc:
            logging.error(f"Unable to fetch metric '{metric_name}' from '{url}' with params '{query_specs}'",
                          exc_info=exc)

    @staticmethod
    async def __gather__(query_list: List[Tuple[str, dict]],
                         results: List[Tuple[str, PrometheusApiResponseModel]]):
        url: str = os.path.join(prometheus_settings.prometheus_endpoint, "api/v1/query_range")
        async with aiohttp.ClientSession() as session:
            results += await asyncio.gather(*[PrometheusClient.__fetch__(url, session, item_tuple)
                                              for item_tuple in query_list])

    @staticmethod
    def __transform_metrics__(result_list: List[Tuple[str, PrometheusApiResponseModel]]):
        node_aggregation_dict: Dict[str, NodeMetricsModel] = {}

        for metric_name, api_response in result_list:
            for data_instance in api_response.data.result:
                node_name: str = data_instance.metric.get("nodename", None)
                if node_name:
                    data_list: List[DataRecordingModel] = [DataRecordingModel(time=float(tup[0]), value=float(tup[1]))
                                                           for tup in data_instance.values]
                    new_aggregations = {metric_name: data_list, "node_name": node_name}
                    aggregation_element: Optional[NodeMetricsModel] = node_aggregation_dict.get(node_name, None)
                    aggregation_dict: dict = aggregation_element.dict() if aggregation_element is not None else {}
                    aggregation_dict.update(new_aggregations)
                    node_aggregation_dict[node_name] = NodeMetricsModel(**aggregation_dict)

        return list(node_aggregation_dict.values())

    @staticmethod
    async def __get_metrics__(query_list: List[Tuple[str, str]], start: datetime, end: datetime,
                              node_name: Optional[str] = None):
        new_query_list: List[Tuple[str, dict]] = []
        for metric_name, query_string in query_list:
            query_specs: dict = {
                "query": f"{query_string} * {NodeMetricsModel.__fields__.get('join_node_info').field_info.description}",
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": int(prometheus_settings.prometheus_query_step_width)
            }

            if node_name is not None:
                query_specs["query"] += f" {{nodename=\"{node_name}\"}}"

            new_query_list.append((metric_name, query_specs))
        results: List[Tuple[str, PrometheusApiResponseModel]] = []
        await PrometheusClient.__gather__(new_query_list, results)
        return PrometheusClient.__transform_metrics__(results)

    @staticmethod
    async def get_node_metrics(start: datetime, end: datetime, node_name: Optional[str] = None):
        query_list: List[tuple] = []
        for field_name, field in NodeMetricsModel.__fields__.items():
            if field_name not in ["node_name", "join_node_info"]:
                query_list.append((field_name.lower(), field.field_info.description))

        return await PrometheusClient.__get_metrics__(query_list, start, end, node_name=node_name)
