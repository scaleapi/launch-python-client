from dataclasses import dataclass
from typing import Sequence

from launch_api.loader import LoaderSpec
from launch_api.model import ModelNamed, NamedArrays
from launch_api.model_service import Processor
from launch_api.model_service.implementation import LoaderInferenceServiceImpl
from launch_api.types import I, O
from scaleml.utils.logging import logger_name, make_logger

logger = make_logger(logger_name())

__all__: Sequence[str] = (
    "ProcessorNamed",
    "LoaderNamedInferenceService",
)

ProcessorNamed = Processor[I, O, NamedArrays]


@dataclass(frozen=True, init=False)
class LoaderNamedInferenceService(
    LoaderInferenceServiceImpl[I, O, NamedArrays]
):
    def __init__(
        self,
        Processor: LoaderSpec[ProcessorNamed[I, O]],
        model: LoaderSpec[ModelNamed],
    ) -> None:
        super().__init__(Processor, model, logger)
