from pydantic import BaseModel, validator


class ProfilingResultModel(BaseModel):
    strategy: str
    name: str
    num_profilings: int
    runtime: float
    cost: float
    util: float

    @validator('runtime')
    def runtime_format(cls, runtime: float):
        return round(runtime, 2)

    @validator('cost')
    def cost_format(cls, cost: float):
        return round(cost, 2)

    @validator('util')
    def util_format(cls, util: float):
        return round(util, 2)
