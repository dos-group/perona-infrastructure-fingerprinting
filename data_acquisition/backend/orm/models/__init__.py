from sqlalchemy import Float, Index, ForeignKey, Boolean
from sqlalchemy import Column, String, TIMESTAMP
from sqlalchemy.orm import relationship, registry
import datetime

mapper_registry = registry()

# TODO remove
metadata_obj = mapper_registry.metadata

Base = mapper_registry.generate_base()


class Benchmark(Base):
    __tablename__ = "benchmarks"

    id = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=False)
    chaos_desired = Column(Boolean, nullable=False, default=False)
    chaos_applied = Column(Boolean, nullable=False, default=False)
    chaos_id = Column(String, nullable=True)
    node_id = Column(String, nullable=False)
    pod_id = Column(String, nullable=False)
    started = Column(TIMESTAMP, nullable=False)
    finished = Column(TIMESTAMP, nullable=False)
    metric_collection_status = Column(String, nullable=False)
    image = Column(String)
    options = Column(String)
    logs = Column(String)

    metrics = relationship("BenchmarkMetric", back_populates="benchmark")

    @property
    def duration(self) -> datetime.timedelta:
        return self.finished - self.started


class BenchmarkMetric(Base):
    __tablename__ = "benchmark_metrics_2"

    benchmark_id = Column(ForeignKey("benchmarks.id"), primary_key=True, nullable=False)
    name = Column(String(30), primary_key=True, nullable=False)
    text_value = Column(String, nullable=True)
    value = Column(Float, nullable=True)
    unit = Column(String, nullable=True)

    benchmark = relationship("Benchmark", back_populates="metrics")


class NodeMetric(Base):
    __tablename__ = "node_metrics"

    node_name = Column(String, primary_key=True, nullable=False)
    metric = Column(String, primary_key=True, nullable=False)
    timestamp = Column(TIMESTAMP, primary_key=True, nullable=False)
    value = Column(String, nullable=False)


ix_timestamp = Index("ix_nodemetric_timestamp", NodeMetric.timestamp)

benchmarks_table = Benchmark.__table__
metrics_table = BenchmarkMetric.__table__
node_metrics_table = NodeMetric.__table__
