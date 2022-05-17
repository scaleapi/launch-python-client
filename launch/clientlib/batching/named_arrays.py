from dataclasses import dataclass
from typing import Sequence

from launch_api.batching.implementation import LoaderBatchableServiceImpl
from launch_api.batching.types import Batcher
from launch_api.loader import LoaderSpec
from launch_api.model import ModelNamed, NamedArrays
from launch_api.types import I, O
from scaleml.utils.logging import logger_name, make_logger

logger = make_logger(logger_name())

__all__: Sequence[str] = (
    "BatcherNamed",
    "LoaderNamedBatchableService",
)

BatcherNamed = Batcher[I, O, NamedArrays]


@dataclass(frozen=True, init=False)
class LoaderNamedBatchableService(
    LoaderBatchableServiceImpl[I, O, NamedArrays]
):
    def __init__(
        self,
        batcher: LoaderSpec[BatcherNamed[I, O]],
        model: LoaderSpec[ModelNamed],
    ) -> None:
        super().__init__(batcher, model, logger)
