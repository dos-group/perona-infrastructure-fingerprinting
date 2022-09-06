from .common import BMMetricField, num_extractor, num_unit_extractor
import dataclasses


@dataclasses.dataclass
class SysbenchCpuMetrics:
    version: BMMetricField \
        = BMMetricField(r"^sysbench (.*)")

    num_threads: BMMetricField \
        = BMMetricField(rf"^Number of threads: ({num_extractor})")

    prime_numbers_limit: BMMetricField \
        = BMMetricField(rf"Prime numbers limit: ({num_extractor})")

    events_per_second: BMMetricField \
        = BMMetricField(rf"events per second:\s*({num_extractor})")

    total_time: BMMetricField \
        = BMMetricField(rf"total time:\s*({num_unit_extractor})")

    total_number_of_events: BMMetricField \
        = BMMetricField(rf"total number of events:\s*({num_extractor})")

    latency_min: BMMetricField \
        = BMMetricField(rf"min:\s*({num_extractor})", extraction_args=([1], "ms"))

    latency_avg: BMMetricField \
        = BMMetricField(rf"avg:\s*({num_extractor})", extraction_args=([1], "ms"))

    latency_max: BMMetricField \
        = BMMetricField(rf"max:\s*({num_extractor})", extraction_args=([1], "ms"))

    latency_95p: BMMetricField \
        = BMMetricField(rf"95th percentile:\s*({num_extractor})", extraction_args=([1], "ms"))

    latency_sum: BMMetricField \
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
