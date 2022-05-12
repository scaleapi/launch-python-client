from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Optional, Sequence

from launch_api.types import I, O

__all__: Sequence[str] = (
    "Service",
    "Runtime",
    "Deployment",
)

##
##
##
##
## WHAT ##
##
##
##
##


class Service(Generic[I, O], ABC):
    """Core user-defined logic abstraction.

    Users *must* implement a Service to their sepecifications! The `Service` class is
    where a Launch user can define all of their custom code.
    """

    @abstractmethod
    def call(self, request: I) -> O:
        raise NotImplementedError()

    def __call__(self, request: I) -> O:
        """Alias to :func:`call`."""
        return self.call(request)


##
##
##
##
## HOW ##
##
##
##
##


class Runtimes(Enum):
    """List of supported runtimes."""

    HTTP, Celery, Triton = auto(), auto(), auto()


class Runtime(ABC):
    runtime_name: Runtimes

    @abstractmethod
    def start(self) -> None:
        """Starts infinite loop: service accepts and responds to requests."""
        raise NotImplementedError()


##
##
##
##
## WHERE ##
##
##
##
##


@dataclass
class Autoscaling:
    min_workers: int
    max_workers: int
    target_pod_deployment: float


@dataclass
class Hardware:
    cpu: float
    mem: float
    gpus: Optional[int]
    machine_type: Optional[str]


@dataclass
class DeploymentOptions(Autoscaling, Hardware, ...):
    """The configuration options that determine where the service is deployed and how it acts in the cluster."""


class Deployment:
    def plan(self) -> str:
        """K8s deployment file contents."""
        raise NotImplementedError

    def service_description(self) -> str:
        """Description of the service(s) to be run."""
        raise NotImplementedError

    def runtime_description(self) -> str:
        """Description of the runtime(s) that the service(s) are being run on."""

    def runnable_service_bundle(self) -> str:
        """Unique identifier of the bundle that the deployment uses.

        e.g. <docker image repository>:<image tag> as place-holder prototype value
        """
        raise NotImplementedError

    def options(self) -> DeploymentOptions:
        """The customizable deployment options, including specifications for autoscaling behavior, hardware, etc."""
        raise NotImplementedError


class Status(Enum):
    unstarted = auto()
    pending = auto()
    in_progress = auto()
    complete = auto()
    failure = auto()


## Scale Launch client has: client.perform_deployment(<a Deployment value>) -> ReferencedDeployment


class ReferencedDeployment(ABC):

    deployment: Deployment

    @abstractmethod
    def status(self) -> Status:
        # DB or k8s call(s)
        raise NotImplementedError

    @abstractmethod
    def call(self, serialized_request):  # -> 'Future[<result_type>]':
        """Perform an inference request against the deployed service(s).

        Handles all routing. Returns a future, which represents the result of the computation to be completed.
        Call .get() to block on result. Note that if the request is going to a syncronous deployment, then
        there will exist a background thread that is responsible for receiving the request. It will persist
        until the caller invokes `.get()`.

        If the call fails, then the result will be an exception.
        If `.status()` is not `complete`, then all `call`s will fail.
        """
        raise NotImplementedError
