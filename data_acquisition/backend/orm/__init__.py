import os

from sqlalchemy import create_engine

import orm.models as orm_models

path_to_db: str = "sqlite+pysqlite:///db.sqlite"
# simple check: we are running in kubernetes, i.e., this env variable is defined
if len(os.environ.get("PROMETHEUS_ENDPOINT", "")):
    path_to_db = "sqlite+pysqlite:////tmp/db.sqlite"

engine = create_engine(path_to_db)


def create_tables():
    orm_models.mapper_registry.metadata.create_all(engine)
