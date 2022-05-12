from abc import ABC, abstractmethod
from typing import Generic, Sequence

from launch_api.core import Service
from launch_api.model import B, Model
from launch_api.types import I, O

__all__: Sequence[str] = (
    "Processor",
    "InferenceService",
)


class Processor(Generic[I, O, B], ABC):
    @abstractmethod
    def preprocess(self, request: I) -> B:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, pred: B) -> O:
        raise NotImplementedError


class InferenceService(Service[I, O], Processor[I, O, B], Model[B], ABC):
    """A service that operates on single requests.

    NOTE: This is the "batteries included" service.
    """
