from functools import lru_cache

from .client import K8sClient


@lru_cache()
def get_k8s_client():
    return K8sClient()



