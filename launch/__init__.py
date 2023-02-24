"""

Scale Launch provides ML engineers with a simple Python interface for turning a local code snippet
into a production service that automatically scales according to traffic.


"""

from typing import Sequence

import pkg_resources

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
    SyncEndpoint,
)

__version__ = pkg_resources.get_distribution("scale-launch").version
__all__: Sequence[str] = [
    "AsyncEndpoint",
    "AsyncEndpointBatchResponse",
    "Connection",
    "EndpointRequest",
    "EndpointResponse",
    "EndpointResponseFuture",
    "LaunchClient",
    "ModelBundle",
    "PostInferenceHooks",
    "SyncEndpoint",
]
