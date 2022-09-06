from .common import BMMetricField, num_extractor, num_unit_extractor, unit_extractor
import dataclasses


@dataclasses.dataclass
class FiosMetrics:
    version: BMMetricField \
        = BMMetricField(r"(fio\-.+)")

    number_of_processes: BMMetricField \
        = BMMetricField(rf".*Starting ({num_extractor}) process.*")

    read_clat_min: BMMetricField \
        = BMMetricField(rf"clat.*\(({unit_extractor})\).*min=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_clat_max: BMMetricField \
        = BMMetricField(rf"clat.*\(({unit_extractor})\).*max=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_clat_avg: BMMetricField \
        = BMMetricField(rf"clat.*\(({unit_extractor})\).*avg=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_clat_stdev: BMMetricField \
        = BMMetricField(rf"clat.*\(({unit_extractor})\).*stdev=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_clat_05p: BMMetricField \
        = BMMetricField(rf"clat percentiles.*\(({unit_extractor})\).*\s5\.00th=\[\s*({num_extractor})\].*",
                        extraction_args=([2, 1], ""))

    read_clat_50p: BMMetricField \
        = BMMetricField(rf"clat percentiles.*\(({unit_extractor})\).*50\.00th=\[\s*({num_extractor})\].*",
                        extraction_args=([2, 1], ""))

    read_clat_95p: BMMetricField \
        = BMMetricField(rf"clat percentiles.*\(({unit_extractor})\).*95\.00th=\[\s*({num_extractor})\].*",
                        extraction_args=([2, 1], ""))

    read_lat_min: BMMetricField \
        = BMMetricField(rf"^lat.*\(({unit_extractor})\).*min=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_lat_max: BMMetricField \
        = BMMetricField(rf"^lat.*\(({unit_extractor})\).*max=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_lat_avg: BMMetricField \
        = BMMetricField(rf"^lat.*\(({unit_extractor})\).*avg=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_lat_stdev: BMMetricField \
        = BMMetricField(rf"^lat.*\(({unit_extractor})\).*stdev=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_bw_min: BMMetricField \
        = BMMetricField(rf"bw\s*\(\s*({unit_extractor})\).*min=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_bw_max: BMMetricField \
        = BMMetricField(rf"bw\s*\(\s*({unit_extractor})\).*max=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_bw_avg: BMMetricField \
        = BMMetricField(rf"bw\s*\(\s*({unit_extractor})\).*avg=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_bw_stdev: BMMetricField \
        = BMMetricField(rf"bw\s*\(\s*({unit_extractor})\).*stdev=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""))

    read_iops_min: BMMetricField \
        = BMMetricField(rf"iops.*min=\s*({num_extractor}).*")

    read_iops_max: BMMetricField \
        = BMMetricField(rf"iops.*max=\s*({num_extractor}).*")

    read_iops_avg: BMMetricField \
        = BMMetricField(rf"iops.*avg=({num_extractor}).*")

    read_iops_stdev: BMMetricField \
        = BMMetricField(rf"iops.*stdev=({num_extractor}).*")

    write_clat_min: BMMetricField \
        = BMMetricField(rf"clat.*\(({unit_extractor})\).*min=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_clat_max: BMMetricField \
        = BMMetricField(rf"clat.*\(({unit_extractor})\).*max=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_clat_avg: BMMetricField \
        = BMMetricField(rf"clat.*\(({unit_extractor})\).*avg=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_clat_stdev: BMMetricField \
        = BMMetricField(rf"clat.*\(({unit_extractor})\).*stdev=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_clat_05p: BMMetricField \
        = BMMetricField(rf"clat percentiles.*\(({unit_extractor})\).*\s5\.00th=\[\s*({num_extractor})\].*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_clat_50p: BMMetricField \
        = BMMetricField(rf"clat percentiles.*\(({unit_extractor})\).*50\.00th=\[\s*({num_extractor})\].*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_clat_95p: BMMetricField \
        = BMMetricField(rf"clat percentiles.*\(({unit_extractor})\).*95\.00th=\[\s*({num_extractor})\].*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_lat_min: BMMetricField \
        = BMMetricField(rf"^lat.*\(({unit_extractor})\).*min=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_lat_max: BMMetricField \
        = BMMetricField(rf"^lat.*\(({unit_extractor})\).*max=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)
    write_lat_avg: BMMetricField \
        = BMMetricField(rf"^lat.*\(({unit_extractor})\).*avg=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_lat_stdev: BMMetricField \
        = BMMetricField(rf"^lat.*\(({unit_extractor})\).*stdev=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_iops_min: BMMetricField \
        = BMMetricField(rf"iops.*min=\s*({num_extractor}).*", overwrite=True)

    write_iops_max: BMMetricField \
        = BMMetricField(rf"iops.*max=\s*({num_extractor}).*", overwrite=True)

    write_iops_avg: BMMetricField \
        = BMMetricField(rf"iops.*avg=({num_extractor}).*", overwrite=True)

    write_iops_stdev: BMMetricField \
        = BMMetricField(rf"iops.*stdev=({num_extractor}).*", overwrite=True)

    write_bw_min: BMMetricField \
        = BMMetricField(rf"bw\s*\(\s*({unit_extractor})\).*min=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_bw_max: BMMetricField \
        = BMMetricField(rf"bw\s*\(\s*({unit_extractor})\).*max=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_bw_avg: BMMetricField \
        = BMMetricField(rf"bw\s*\(\s*({unit_extractor})\).*avg=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    write_bw_stdev: BMMetricField \
        = BMMetricField(rf"bw\s*\(\s*({unit_extractor})\).*stdev=\s*({num_extractor}).*",
                        extraction_args=([2, 1], ""),
                        overwrite=True)

    cpu_usr: BMMetricField \
        = BMMetricField(rf"cpu.*usr=({num_unit_extractor}).*")

    cpu_sys: BMMetricField \
        = BMMetricField(rf"cpu.*sys=({num_unit_extractor}).*")

    disk_util: BMMetricField \
        = BMMetricField(rf".*util=({num_unit_extractor}).*")


@dataclasses.dataclass
class IopingsMetrics:
    number_of_requests: BMMetricField \
        = BMMetricField(rf"({num_extractor}) requests completed.*")

    total_duration: BMMetricField \
        = BMMetricField(rf".*requests completed in ({num_unit_extractor}),")

    transfer_size: BMMetricField \
        = BMMetricField(rf".*requests completed.*, ({num_unit_extractor}),.*iops")

    iops: BMMetricField \
        = BMMetricField(rf".*requests completed.*,\s*({num_extractor})\s*iops,")

    transfer_bitrate: BMMetricField \
        = BMMetricField(rf".*requests completed.*, ({num_unit_extractor})[\s\n]?")

    latency_min: BMMetricField \
        = BMMetricField(rf"min\/avg\/max\/mdev = ({num_unit_extractor})")

    latency_avg: BMMetricField \
        = BMMetricField(rf"min\/avg\/max\/mdev = .+?\/ ({num_unit_extractor})")

    latency_max: BMMetricField \
        = BMMetricField(rf"min\/avg\/max\/mdev = .+?\/ .+?\/ ({num_unit_extractor})")

    latency_mdev: BMMetricField \
        = BMMetricField(rf"min\/avg\/max\/mdev = .+?\/ .+?\/ .+?\/ ({num_unit_extractor})")
