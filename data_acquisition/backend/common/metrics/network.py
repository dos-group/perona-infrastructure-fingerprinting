from .common import BMMetricField, num_unit_extractor, num_extractor
import dataclasses


@dataclasses.dataclass
class NetworkIperf3Metrics:
    tx_c_transfer_size_sender: BMMetricField \
        = BMMetricField(rf".*\[TX-C\].*[^\/]sec\s*({num_unit_extractor})\s.*sender")
    tx_c_transfer_size_receiver: BMMetricField \
        = BMMetricField(rf".*\[TX-C\].*[^\/]sec\s*({num_unit_extractor})\s.*receiver")
    tx_c_transfer_bitrate_sender: BMMetricField \
        = BMMetricField(rf".*\[TX-C\].*\s({num_unit_extractor})\s+{num_extractor}\s*sender")
    tx_c_transfer_bitrate_receiver: BMMetricField \
        = BMMetricField(rf".*\[TX-C\].*\s({num_unit_extractor}).*receiver")
    tx_c_transfer_retr_sender: BMMetricField \
        = BMMetricField(rf".*\[TX-C\].*\s({num_extractor})\s*sender")
    tx_c_cpu_util_sender: BMMetricField \
        = BMMetricField(rf".*CPU Utilization\: local\/sender\s({num_unit_extractor}).*")
    tx_c_cpu_util_receiver: BMMetricField \
        = BMMetricField(rf".*CPU Utilization.*remote\/receiver\s({num_unit_extractor}).*")
    rx_c_cpu_util_sender: BMMetricField \
        = BMMetricField(rf".*CPU Utilization.*local\/receiver\s({num_unit_extractor}).*")
    rx_c_cpu_util_receiver: BMMetricField \
        = BMMetricField(rf".*CPU Utilization.*remote\/sender\s({num_unit_extractor}).*")
    rx_c_transfer_size_sender: BMMetricField \
        = BMMetricField(rf".*\[RX-C\].*[^\/]sec\s*({num_unit_extractor})\s.*sender")
    rx_c_transfer_size_receiver: BMMetricField \
        = BMMetricField(rf".*\[RX-C\].*[^\/]sec\s*({num_unit_extractor})\s.*receiver")
    rx_c_transfer_bitrate_sender: BMMetricField \
        = BMMetricField(rf".*\[RX-C\].*\s({num_unit_extractor})\s+{num_extractor}\s*sender")
    rx_c_transfer_bitrate_receiver: BMMetricField \
        = BMMetricField(rf".*\[RX-C\].*\s({num_unit_extractor}).*receiver")
    rx_c_transfer_retr_sender: BMMetricField \
        = BMMetricField(rf".*\[RX-C\].*\s({num_extractor})\s*sender")


@dataclasses.dataclass
class QpervesMetrics:
    tcp_bw_bandwidth: BMMetricField \
        = BMMetricField(rf".*bw\s*=\s*({num_unit_extractor})")
    tcp_bw_msg_rate: BMMetricField \
        = BMMetricField(rf".*msg_rate\s*=\s*({num_unit_extractor})")
    tcp_bw_access_recv: BMMetricField \
        = BMMetricField(rf".*access_recv\s*=\s*({num_extractor})")
    tcp_bw_msg_size: BMMetricField \
        = BMMetricField(rf".*msg_size\s*=\s*({num_unit_extractor})")
    tcp_bw_time: BMMetricField \
        = BMMetricField(rf"^\s*time\s*=\s*({num_unit_extractor})")
    tcp_bw_timeout: BMMetricField \
        = BMMetricField(rf"^\s*timeout\s*=\s*({num_unit_extractor})")
    tcp_bw_send_cost: BMMetricField \
        = BMMetricField(rf".*send_cost\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_cost: BMMetricField \
        = BMMetricField(rf".*recv_cost\s*=\s*({num_unit_extractor})")
    tcp_bw_send_cpus_used: BMMetricField \
        = BMMetricField(rf".*send_cpus_used\s*=\s*({num_unit_extractor})")
    tcp_bw_send_cpus_user: BMMetricField \
        = BMMetricField(rf".*send_cpus_user\s*=\s*({num_unit_extractor})")
    tcp_bw_send_cpus_intr: BMMetricField \
        = BMMetricField(rf".*send_cpus_intr\s*=\s*({num_unit_extractor})")
    tcp_bw_send_cpus_kernel: BMMetricField \
        = BMMetricField(rf".*send_cpus_kernel\s*=\s*({num_unit_extractor})")
    tcp_bw_send_cpus_iowait: BMMetricField \
        = BMMetricField(rf".*send_cpus_iowait\s*=\s*({num_unit_extractor})")
    tcp_bw_send_real_time: BMMetricField \
        = BMMetricField(rf".*send_real_time\s*=\s*({num_unit_extractor})")
    tcp_bw_send_cpu_time: BMMetricField \
        = BMMetricField(rf".*send_cpu_time\s*=\s*({num_unit_extractor})")
    tcp_bw_send_bytes: BMMetricField \
        = BMMetricField(rf".*send_bytes\s*=\s*({num_unit_extractor})")
    tcp_bw_send_msgs: BMMetricField \
        = BMMetricField(rf".*send_msgs\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_cpus_used: BMMetricField \
        = BMMetricField(rf".*recv_cpus_used\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_cpus_user: BMMetricField \
        = BMMetricField(rf".*recv_cpus_user\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_cpus_intr: BMMetricField \
        = BMMetricField(rf".*recv_cpus_intr\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_cpus_kernel: BMMetricField \
        = BMMetricField(rf".*recv_cpus_kernel\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_cpus_iowait: BMMetricField \
        = BMMetricField(rf".*recv_cpus_iowait\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_real_time: BMMetricField \
        = BMMetricField(rf".*recv_real_time\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_cpu_time: BMMetricField \
        = BMMetricField(rf".*recv_cpu_time\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_bytes: BMMetricField \
        = BMMetricField(rf".*recv_bytes\s*=\s*({num_unit_extractor})")
    tcp_bw_recv_msgs: BMMetricField \
        = BMMetricField(rf".*recv_msgs\s*=\s*({num_unit_extractor})")

    tcp_lat_latency: BMMetricField \
        = BMMetricField(rf".*latency\s*=\s*({num_unit_extractor})")
    tcp_lat_msg_rate: BMMetricField \
        = BMMetricField(rf".*msg_rate\s*=\s*({num_unit_extractor})", overwrite=True)
    tcp_lat_msg_size: BMMetricField \
        = BMMetricField(rf".*msg_size\s*=\s*({num_unit_extractor})", overwrite=True)
    tcp_lat_time: BMMetricField \
        = BMMetricField(rf"^\s*time\s*=\s*({num_unit_extractor})", overwrite=True)
    tcp_lat_timeout: BMMetricField \
        = BMMetricField(rf"^\s*timeout\s*=\s*({num_unit_extractor})", overwrite=True)
    tcp_lat_loc_cpus_used: BMMetricField \
        = BMMetricField(rf".*loc_cpus_used\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_cpus_user: BMMetricField \
        = BMMetricField(rf".*loc_cpus_user\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_cpus_intr: BMMetricField \
        = BMMetricField(rf".*loc_cpus_intr\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_cpus_kernel: BMMetricField \
        = BMMetricField(rf".*loc_cpus_kernel\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_cpus_iowait: BMMetricField \
        = BMMetricField(rf".*loc_cpus_iowait\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_real_time: BMMetricField \
        = BMMetricField(rf".*loc_real_time\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_cpu_time: BMMetricField \
        = BMMetricField(rf".*loc_cpu_time\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_send_bytes: BMMetricField \
        = BMMetricField(rf".*loc_send_bytes\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_recv_bytes: BMMetricField \
        = BMMetricField(rf".*loc_recv_bytes\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_send_msgs: BMMetricField \
        = BMMetricField(rf".*loc_send_msgs\s*=\s*({num_unit_extractor})")
    tcp_lat_loc_recv_msgs: BMMetricField \
        = BMMetricField(rf".*loc_recv_msgs\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_cpus_used: BMMetricField \
        = BMMetricField(rf".*rem_cpus_used\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_cpus_user: BMMetricField \
        = BMMetricField(rf".*rem_cpus_user\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_cpus_intr: BMMetricField \
        = BMMetricField(rf".*rem_cpus_intr\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_cpus_kernel: BMMetricField \
        = BMMetricField(rf".*rem_cpus_kernel\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_cpus_iowait: BMMetricField \
        = BMMetricField(rf".*rem_cpus_iowait\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_real_time: BMMetricField \
        = BMMetricField(rf".*rem_real_time\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_cpu_time: BMMetricField \
        = BMMetricField(rf".*rem_cpu_time\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_send_bytes: BMMetricField \
        = BMMetricField(rf".*rem_send_bytes\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_recv_bytes: BMMetricField \
        = BMMetricField(rf".*rem_recv_bytes\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_send_msgs: BMMetricField \
        = BMMetricField(rf".*rem_send_msgs\s*=\s*({num_unit_extractor})")
    tcp_lat_rem_recv_msgs: BMMetricField \
        = BMMetricField(rf".*rem_recv_msgs\s*=\s*({num_unit_extractor})")
