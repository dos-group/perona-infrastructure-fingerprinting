import copy
import inspect
from typing import Tuple, List, Any, Union

from ax import SearchSpace, Parameter
from ax.core.parameter import PARAMETER_PYTHON_TYPE_MAP, FixedParameter, ChoiceParameter, RangeParameter, \
    _get_parameter_type

from classes.processed_workload import ProcessedWorkloadModel
from optimization.machine_types import aws_machine_specs


class EmbedderBO:
    def __init__(self):
        self.spec_classes: List[Tuple[str, Any]] = [("NodeCount", None)] + list(inspect.getmembers(aws_machine_specs,
                                                                                                   inspect.isclass))
        self.parameters: List[Parameter] = []

    def get_parameters(self, *args, **kwargs):
        raise NotImplementedError

    def vectorize(self, *args, **kwargs):
        raise NotImplementedError

    def reconstruct(self, source_dict: dict):
        return [source_dict[f"x{idx}"] for idx in range(len(self.parameters))]

    def construct(self, source_list: list):
        return {f"x{idx}": value for idx, value in enumerate(source_list)}

    def get_search_space(self):
        return SearchSpace(parameters=self.parameters)

    def get_search_space_as_list(self):
        map_dict = {"ChoiceParameter": "choice", "FixedParameter": "fixed", "RangeParameter": "range"}

        def _parameter_to_dict(param: Parameter):
            value_dict = copy.deepcopy(param.__dict__)
            value_dict.pop("_sort_values", None)
            if param.__class__.__name__ == "RangeParameter":
                value_dict["_bounds"] = [value_dict["_lower"], value_dict["_upper"]]
                value_dict["_log_scale"] = False
                value_dict.pop("_lower", None)
                value_dict.pop("_logit_scale", None)
                value_dict.pop("_upper", None)

            value_dict["_value_type"] = PARAMETER_PYTHON_TYPE_MAP[value_dict.pop("_parameter_type")].__name__
            value_dict["_type"] = map_dict.get(param.__class__.__name__)
            return {k[1:]: v for k, v in value_dict.items()}

        return [_parameter_to_dict(p) for p in self.parameters]


class CherryPickEmbedderBO(EmbedderBO):
    def __init__(self, workloads: List[ProcessedWorkloadModel]):
        super().__init__()
        self.spec_classes = [(name, el) for (name, el) in self.spec_classes if "CpuType" not in name]
        self.parameters: List[Parameter] = self.get_parameters(workloads)

    def get_parameters(self, workloads: List[ProcessedWorkloadModel]):
        parameters: List[Union[ChoiceParameter, FixedParameter, RangeParameter]] = []
        for idx, (_, clazz) in enumerate(self.spec_classes):
            parameter: Union[ChoiceParameter, FixedParameter]
            if idx == 0:
                node_counts: List[int] = sorted(list(set([w.node_count for w in workloads])))
                if len(node_counts) == 1:
                    parameter = FixedParameter(name=f"x{idx}",
                                               parameter_type=_get_parameter_type(int),
                                               value=node_counts[0])
                else:
                    parameter = ChoiceParameter(name=f"x{idx}",
                                                parameter_type=_get_parameter_type(int),
                                                values=sorted(list(set([w.node_count for w in workloads]))),
                                                sort_values=False,
                                                is_ordered=True)
            elif len(clazz.__dict__.get("ORDER")) == 1:
                parameter = FixedParameter(name=f"x{idx}",
                                           parameter_type=_get_parameter_type(type(clazz.__dict__.get("ORDER")[0])),
                                           value=clazz.__dict__.get("ORDER")[0])
            else:
                parameter = ChoiceParameter(name=f"x{idx}",
                                            parameter_type=_get_parameter_type(type(clazz.__dict__.get("ORDER")[0])),
                                            values=clazz.__dict__.get("ORDER"),
                                            sort_values=False,
                                            is_ordered=True)
            parameters.append(parameter)
        return parameters

    def vectorize(self, workload: ProcessedWorkloadModel) -> List[Union[int, float, str]]:
        features: List[Union[int, float, str]] = []
        for idx, (_, clazz) in enumerate(self.spec_classes):
            value: Union[int, float, str]
            if idx == 0:
                value = workload.node_count
            else:
                value = clazz.__dict__.get(workload.machine_name)
            features.append(value)
        return features


class ArrowEmbedderBO(EmbedderBO):
    def __init__(self, workloads: List[ProcessedWorkloadModel]):
        super().__init__()
        filtered_spec_classes = []
        for key in ["CpuType", "CoreCount", "RamPerCore", "NetworkCapacity"]:
            element = next((el for el in self.spec_classes if key in el[0]))
            filtered_spec_classes.append(element)
        self.spec_classes = filtered_spec_classes
        self.parameters: List[Parameter] = self.get_parameters(workloads)

    def get_parameters(self, workloads: List[ProcessedWorkloadModel]):
        parameters: List[Union[ChoiceParameter, FixedParameter]] = []
        for idx, (_, clazz) in enumerate(self.spec_classes):
            parameter: Union[ChoiceParameter, FixedParameter]
            if idx == 0:
                node_counts: List[int] = sorted(list(set([w.node_count for w in workloads])))
                if len(node_counts) == 1:
                    parameter = FixedParameter(name=f"x{idx}",
                                               parameter_type=_get_parameter_type(int),
                                               value=node_counts[0])
                else:
                    parameter = ChoiceParameter(name=f"x{idx}",
                                                parameter_type=_get_parameter_type(int),
                                                values=sorted(list(set([w.node_count for w in workloads]))),
                                                sort_values=False,
                                                is_ordered=True)
            else:
                parameter: ChoiceParameter = ChoiceParameter(name=f"x{idx}",
                                                             parameter_type=_get_parameter_type(
                                                                 type(clazz.__dict__.get("ORDER")[0])),
                                                             values=clazz.__dict__.get("ORDER"),
                                                             sort_values=False,
                                                             is_ordered=True)
            parameters.append(parameter)
        return parameters

    def vectorize(self, workload: ProcessedWorkloadModel) -> List[Union[int, float, str]]:
        features: List[Union[int, float, str]] = []
        for idx, (_, clazz) in enumerate(self.spec_classes):
            value: Union[int, float, str]
            if idx == 0:
                value = workload.node_count
            else:
                value = clazz.__dict__.get(workload.machine_name)
            features.append(value)
        return features
