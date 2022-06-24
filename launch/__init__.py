"""

Scale Launch provides ML engineers with a simple Python interface for turning a local code snippet into a
production service that automatically scales according to traffic.


"""

import pkg_resources

from .client import LaunchClient
from .connection import Connection
from .constants import DEFAULT_NETWORK_TIMEOUT_SEC
from .logger import logger
from .model_bundle import ModelBundle
from .model_endpoint import (
    AsyncEndpoint,
    AsyncEndpointBatchResponse,
    EndpointRequest,
    EndpointResponse,
    EndpointResponseFuture,
    SyncEndpoint,
)
from .retry_strategy import RetryStrategy

__version__ = pkg_resources.get_distribution("scale-launch").version
