from .client import BenchmarkHistoryClient


def get_benchmark_history_client() -> BenchmarkHistoryClient:
    return BenchmarkHistoryClient()
