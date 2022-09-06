import string
from abc import ABC
import random
from typing import Type, Tuple, Optional, Any

import pykube
from pykube.objects import APIObject


class BaseRun(ABC):
    @property
    def name(self):
        raise NotImplementedError

    @property
    def kind(self):
        raise NotImplementedError

    @classmethod
    def get_factory(cls, client: pykube.HTTPClient, kind: str) -> Type[APIObject]:
        raise NotImplementedError

    @classmethod
    def get_obj_for_selector(cls, k8s_client: any, selector: dict, kind: str,
                             namespace: Optional[str] = None) -> Tuple[Type[APIObject], Optional[dict]]:
        factory_instance: Type[APIObject] = cls.get_factory(k8s_client.api, kind)
        namespace = namespace or "kubestone"
        resp: Optional[Any] = factory_instance.objects(k8s_client.api, namespace=namespace).get_or_none(**selector)
        return factory_instance, resp.obj if resp is not None else resp

    @classmethod
    def generate_suffix(cls, length: int = 20):
        # return ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(length))
        return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))

    @classmethod
    def merge_dicts(cls, tgt, enhancer):
        for key, val in enhancer.items():
            if key not in tgt:
                tgt[key] = val
                continue

            if isinstance(val, dict):
                if not isinstance(tgt[key], dict):
                    tgt[key] = dict()
                BaseRun.merge_dicts(tgt[key], val)
            else:
                tgt[key] = val
        return tgt
