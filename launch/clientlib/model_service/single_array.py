from dataclasses import dataclass
from typing import Sequence

import numpy as np
from launch_api.loader import LoaderSpec
from launch_api.model import ModelSingle
from launch_api.model_service import Processor
from launch_api.model_service.implementation import LoaderInferenceServiceImpl
from launch_api.types import I, O
from scaleml.utils.logging import logger_name, make_logger

logger = make_logger(logger_name())

__all__: Sequence[str] = (
    "ProcessorSingle",
    "LoaderSingleBatchableService",
)


ProcessorSingle = Processor[I, O, np.ndarray]


@dataclass(frozen=True, init=False)
class LoaderSingleBatchableService(
    LoaderInferenceServiceImpl[I, O, np.ndarray]
):
    def __init__(
        self,
        Processor: LoaderSpec[ProcessorSingle[I, O]],
        model: LoaderSpec[ModelSingle],
    ) -> None:
        super().__init__(Processor, model, logger)
