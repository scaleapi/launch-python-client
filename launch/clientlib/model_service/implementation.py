from dataclasses import dataclass
from logging import Logger
from typing import List, Sequence

import numpy as np
from launch_api.loader import Loader, LoaderSpec
from launch_api.model import B, Model
from launch_api.model_service.types import InferenceService, Processor
from launch_api.types import I, O

__all__: Sequence[str] = (
    "InferenceServiceImpl",
    "LoaderInferenceServiceImpl",
)


@dataclass
class InferenceServiceImpl(InferenceService[I, O, B]):

    processor: Processor[I, O, B]
    model: Model[B]
    logger: Logger

    def preprocess(self, request: I) -> B:
        return self.processor.preprocess(request)

    def postprocess(self, pred: B) -> List[O]:
        return self.processor.postprocess(pred)

    def __call__(self, batch: B) -> B:
        if isinstance(batch, np.ndarray):
            return self.model(batch)
        else:
            return self.model(**batch)

    def call(self, request: I) -> O:
        try:
            model_input = self.preprocess(request)
        except Exception:
            self.logger.exception(f"Failed to preprocess ({request=})")
            raise
        try:
            pred = self.__call__(model_input)
        except Exception:
            self.logger.exception(
                "Failed inference on request "
                f"({type(model_input)=}, {request=})"
            )
            raise
        try:
            response = self.postprocess(pred)
        except Exception:
            self.logger.exception(
                "Failed to postprocess prediction "
                f"({type(pred)=}, {type(model_input)=}, {request=})"
            )
            raise
        return response


@dataclass(frozen=True)
class LoaderInferenceServiceImpl(Loader[InferenceService[I, O, B]]):
    processor: LoaderSpec[Loader[Processor[I, O, B]]]
    model: LoaderSpec[Loader[Model[B]]]
    logger: Logger

    def load(self) -> InferenceService[I, O, B]:
        self.logger.info(f"Using processor loader:\n{self.processor}")
        self.logger.info(f"Using model loader:\n{self.model}")

        processor_loader = self.processor.construct()
        self.logger.info(f"Made processor loader: {processor_loader}")
        model_loader = self.model.construct()
        self.logger.info(f"Made model loader:     {model_loader}")

        try:
            batcher = processor_loader.load()
        except Exception:
            self.logger.exception(
                f"Could not create processor from {type(processor_loader)}"
            )
            raise
        else:
            self.logger.info(f"Created processor: {batcher}")
        try:
            model = model_loader.load()
        except Exception:
            self.logger.exception(
                f"Could not create model from {type(model_loader)}"
            )
            raise
        else:
            self.logger.info(f"Created model: {model}")
        return InferenceServiceImpl(batcher, model, self.logger)
