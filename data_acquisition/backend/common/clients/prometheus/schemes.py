from datetime import datetime
from math import isnan
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, validator


class DataRecordingModel(BaseModel):
    time: datetime = Field(...)
    value: float = Field(...)

    @validator('*')
    def change_nan_to_none(cls, v, values, config, field):
        if field.outer_type_ is float and isnan(v):
            return 0
        return v


class NodeMetricsModel(BaseModel):
    node_name: str = Field(...)
    # percentage values, i.e. 0 <= x <= 100
    memory_used: List[DataRecordingModel] = Field(
        default=[],
        description='100 - ((avg_over_time(node_memory_MemFree_bytes[20s]) / ' \
                    'avg_over_time(node_memory_MemTotal_bytes[20s])) * 100)')
    memory_swap: List[DataRecordingModel] = Field(
        default=[],
        description='100 - ((avg_over_time(node_memory_SwapFree_bytes[20s]) / ' \
                    'avg_over_time(node_memory_SwapTotal_bytes[20s])) * 100)')
    cpu_busy: List[DataRecordingModel] = Field(
        default=[],
        description='100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[20s])) * 100)')
    disk_read_util: List[DataRecordingModel] = Field(
        default=[],
        description='100 * (sum by (instance) (rate(node_disk_read_time_seconds_total[20s])))')
    disk_write_util: List[DataRecordingModel] = Field(
        default=[],
        description='100 * (sum by (instance) (rate(node_disk_write_time_seconds_total[20s])))')
    disk_io_util: List[DataRecordingModel] = Field(
        default=[],
        description='100 * (sum by (instance) (rate(node_disk_io_time_seconds_total[20s])))')
    filesystem_util: List[DataRecordingModel] = Field(
        default=[],
        description='100 - (100 * (' \
                    '(sum by (instance) ' \
                    '(node_filesystem_avail_bytes{mountpoint="/",fstype!="rootfs"}))  / ' \
                    '(sum by (instance)' \
                    '(node_filesystem_size_bytes{mountpoint="/",fstype!="rootfs"})) ))')
    network_in_errors: List[DataRecordingModel] = Field(
        default=[],
        description='((sum by (instance) (rate(node_network_receive_errs_total[20s]))) / ' \
                    '(sum by (instance) (rate(node_network_receive_packets_total[20s])))) * 100')
    network_out_errors: List[DataRecordingModel] = Field(
        default=[],
        description='((sum by (instance) (rate(node_network_transmit_errs_total[20s]))) / ' \
                    '(sum by (instance) (rate(node_network_transmit_packets_total[20s])))) * 100')
    # absolute values
    disk_read_bytes: List[DataRecordingModel] = Field(
        default=[],
        description='(sum by (instance) (rate(node_disk_read_bytes_total[20s])))')
    disk_write_bytes: List[DataRecordingModel] = Field(
        default=[],
        description='(sum by (instance) (rate(node_disk_written_bytes_total[20s])))')
    network_in_bytes: List[DataRecordingModel] = Field(
        default=[],
        description='(sum by (instance) (rate(node_network_receive_bytes_total[20s])))')
    network_out_bytes: List[DataRecordingModel] = Field(
        default=[],
        description='(sum by (instance) (rate(node_network_transmit_bytes_total[20s])))')
    # join information
    join_node_info: Optional[str] = Field(
        default=None,
        description="on(instance) group_right() node_uname_info")


class PrometheusDataInstanceModel(BaseModel):
    metric: dict
    values: List[Tuple[float, str]]  # point in time, value


class PrometheusDataModel(BaseModel):
    resultType: str
    result: List[PrometheusDataInstanceModel]


class PrometheusApiResponseModel(BaseModel):
    status: str
    errorType: Optional[str] = None
    error: Optional[str] = None
    data: Optional[PrometheusDataModel] = None
