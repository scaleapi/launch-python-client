import inspect

from pydantic import BaseModel

from launch.api_client import models
from launch.api_client.api_client import (  # noqa F401
    ApiClient,
    AsyncApis,
    SyncApis,
)

for model in inspect.getmembers(models, inspect.isclass):
    if model[1].__module__ == "launch.api_client.models":
        model_class = model[1]
        if isinstance(model_class, BaseModel):
            model_class.update_forward_refs()
