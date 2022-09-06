from .common import BMMetricField, num_extractor, num_unit_extractor
import dataclasses


@dataclasses.dataclass
class SysbenchMemoryMetrics:
    version: BMMetricField\
        = BMMetricField(r"^sysbench (.*)")

    number_of_threads: BMMetricField\
        = BMMetricField(rf"Number of threads: ({num_extractor})")

    block_size: BMMetricField\
        = BMMetricField(r"block size: (.*)")

    total_size: BMMetricField\
        = BMMetricField(r"total size: (.*)")

    operation: BMMetricField\
        = BMMetricField(r"operation: (.*)")

    scope: BMMetricField\
        = BMMetricField(r"scope: (.*)")

    transfer_size: BMMetricField \
        = BMMetricField(rf"({num_unit_extractor})\s*transferred")

    transfer_rate: BMMetricField \
        = BMMetricField(rf".*transferred\s*\(({num_unit_extractor})\)")

    total_time: BMMetricField\
        = BMMetricField(rf"total time:\s*({num_unit_extractor})")

    total_number_of_events: BMMetricField\
        = BMMetricField(rf"total number of events:\s*({num_extractor})")

    latency_min: BMMetricField\
        = BMMetricField(rf"min:\s*({num_extractor})", extraction_args=([1], "ms"))

    latency_avg: BMMetricField\
        = BMMetricField(rf"avg:\s*({num_extractor})", extraction_args=([1], "ms"))

    latency_max: BMMetricField\
        = BMMetricField(rf"max:\s*({num_extractor})", extraction_args=([1], "ms"))

    latency_95p: BMMetricField\
        = BMMetricField(rf"95th percentile:\s*({num_extractor})", extraction_args=([1], "ms"))

    latency_sum: BMMetricField\
        = BMMetricField(rf"sum:\s*({num_extractor})", extraction_args=([1], "ms"))

    fairness_events_avg: BMMetricField \
        = BMMetricField(rf"events \(avg\/stddev\):\s*({num_extractor})")

    fairness_events_stddev: BMMetricField \
        = BMMetricField(rf"events \(avg\/stddev\):\s*{num_extractor}\/({num_extractor})")

    fairness_exctime_avg: BMMetricField \
        = BMMetricField(rf"execution time \(avg\/stddev\):\s*({num_extractor})", extraction_args=([1], "s"))

    fairness_exctime_stddev: BMMetricField \
        = BMMetricField(rf"execution time \(avg\/stddev\):\s*{num_extractor}\/({num_extractor})",
                        extraction_args=([1], "s"))
