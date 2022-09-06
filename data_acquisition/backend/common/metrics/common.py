from __future__ import annotations

import re
from typing import TypeVar, Optional, Type, Dict, Iterable, List, Union, Match, Tuple

TMetricClass = TypeVar("TMetricClass")

num_extractor = r"[\d\.,]+"
unit_extractor = r"[\w\/\%]*"
num_unit_extractor = num_extractor + r"\s*" + unit_extractor


class BMMetricField:
    pattern: Optional[str]
    value: Optional[Union[str, List[str]]] = None
    collect_list: bool = False
    overwrite: bool = False
    extraction_args: Tuple[List[int], str] = ([1], "")

    def __init__(self, pattern: Optional[str], collect_list: bool = False,
                 overwrite: bool = False, extraction_args: Tuple[List[int], str] = ([1], "")):
        self.pattern = pattern
        self.collect_list = collect_list
        self.overwrite = overwrite
        self.extraction_args = extraction_args

    def copy(self) -> BMMetricField:
        f = BMMetricField(
            pattern=self.pattern,
            collect_list=self.collect_list,
            overwrite=self.overwrite
        )
        f.value = self.value.copy() if type(self.value) == list else self.value
        return f

    def extract_value(self, regex_match: Match):
        ordered_group_ids, suffix = self.extraction_args
        return "".join([regex_match.group(group_id).strip("").strip(",") for group_id in ordered_group_ids] + [suffix])

    def __str__(self):
        return self.value or repr(self)

    def __repr__(self):
        if self.value is not None:
            return repr(self.value)
        else:
            return f"(no value, pattern: {self.pattern})"


def read_benchmark_metrics(cls: Type[TMetricClass], lines: Iterable[str]) -> TMetricClass:
    metric_fields: Dict[str, BMMetricField] = {
        k: v for k, v in vars(cls).items() if isinstance(v, BMMetricField)
    }

    result = cls()

    for line in lines:
        for msmt, msmt_field in metric_fields.items():
            m: Match = re.match(msmt_field.pattern, line.strip(), re.IGNORECASE)
            if m:
                val = msmt_field.extract_value(m)
                field: BMMetricField = getattr(result, msmt).copy()

                if field.collect_list:
                    if field.value is None:
                        field.value = [val]
                    else:
                        field.value.append(val)
                else:
                    if field.value is None or field.overwrite:
                        field.value = val

                setattr(result, msmt, field)

    return result
