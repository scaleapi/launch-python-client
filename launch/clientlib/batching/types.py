from abc import ABC, abstractmethod
from typing import Generic, List, Sequence

from launch_api.core import Service
from launch_api.model import B, Model
from launch_api.types import I, O

__all__: Sequence[str] = (
    "Batcher",
    "BatchableService",
)


class Batcher(Generic[I, O, B], ABC):
    @abstractmethod
    def batch(self, requests: List[I]) -> B:
        raise NotImplementedError

    @abstractmethod
    def unbatch(self, preds: B) -> List[O]:
        raise NotImplementedError


class BatchableService(
    Service[List[I], List[O]], Batcher[I, O, B], Model[B], ABC
):
    """A service that operates on batches of requests.

    The objective is to provide optimized throughput for model inference services.
    Many models predictions are embarrassingly parallel. Thus they can operate efficeintly
    on a batch of inputs, producing a batch of outputs.

    This interface's objective is to provide a `call_batch` function that uses
    a `Batcher` and a `Model`

    NOTE: This is the "batteries included" service.
    """
