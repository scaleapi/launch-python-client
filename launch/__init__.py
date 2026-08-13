"""

Scale Launch provides ML engineers with a simple Python interface for turning a local code snippet
into a production service that automatically scales according to traffic.


"""
# pylint: disable=C0413

import warnings
from importlib.metadata import version
from typing import Sequence

import pydantic

if pydantic.VERSION.startswith("2."):
    # HACK: Suppress warning from pydantic v2 about protected namespace, this is due to
    # launch-python-client module is based on v1 and only does minimum to support forward compatibility
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from .client import LaunchClient
from .connection import Connection
from .hooks import PostInferenceHooks
from .model_bundle import ModelBundle
from .model_endpoint import (
    AsyncEndpoint,
    AsyncEndpointBatchResponse,
    EndpointRequest,
    EndpointResponse,
    EndpointResponseFuture,
    EndpointResponseStream,
    SyncEndpoint,
)

__version__ = version("scale-launch")
__all__: Sequence[str] = [
    "AsyncEndpoint",
    "AsyncEndpointBatchResponse",
    "Connection",
    "EndpointRequest",
    "EndpointResponse",
    "EndpointResponseFuture",
    "EndpointResponseStream",
    "LaunchClient",
    "ModelBundle",
    "PostInferenceHooks",
    "SyncEndpoint",
]
