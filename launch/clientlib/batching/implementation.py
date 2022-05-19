from dataclasses import dataclass
from logging import Logger
from typing import List, Sequence

import numpy as np
from launch_api.batching.types import B, BatchableService, Batcher, Model
from launch_api.loader import Loader, LoaderSpec
from launch_api.types import I, O

__all__: Sequence[str] = (
    "BatchableServiceImpl",
    "LoaderBatchableServiceImpl",
)


@dataclass
class BatchableServiceImpl(BatchableService[I, O, B]):

    batcher: Batcher[I, O, B]
    model: Model[B]
    logger: Logger

    def batch(self, requests: List[I]) -> B:
        return self.batcher.batch(requests)

    def unbatch(self, preds: B) -> List[O]:
        return self.batcher.unbatch(preds)

    def __call__(self, batch: B) -> B:
        if isinstance(batch, np.ndarray):
            return self.model(batch)
        else:
            return self.model(**batch)

    def call(self, requests: List[I]) -> List[O]:
        try:
            batch = self.batch(requests)
        except Exception:
            self.logger.exception(
                f"Failed to preprocess & batch requests "
                f"({len(requests)=}, {requests=})"
            )
            raise
        try:
            preds = self.__call__(batch)
        except Exception:
            self.logger.exception(
                f"Failed inference on request batch "
                f"({type(batch)=}, {len(requests)=}, {requests=})"
            )
            raise
        try:
            responses = self.unbatch(preds)
        except Exception:
            self.logger.exception(
                f"Failed to postprocess prediction batch "
                f"({type(preds)=}, {type(batch)=}, {len(requests)=}, {requests=}))"
            )
            raise
        return responses


@dataclass(frozen=True)
class LoaderBatchableServiceImpl(Loader[BatchableService[I, O, B]]):
    batcher: LoaderSpec[Loader[Batcher[I, O, B]]]
    model: LoaderSpec[Loader[Model[B]]]
    logger: Logger

    def load(self) -> BatchableService[I, O, B]:
        self.logger.info(f"Using batcher loader:\n{self.batcher}")
        self.logger.info(f"Using model loader:\n{self.model}")

        batcher_loader = self.batcher.construct()
        self.logger.info(f"Made batcher loader: {batcher_loader}")
        model_loader = self.model.construct()
        self.logger.info(f"Made model loader:   {model_loader}")

        try:
            batcher = batcher_loader.load()
        except Exception:
            self.logger.exception(
                f"Could not create batcher from {type(batcher_loader)}"
            )
            raise
        else:
            self.logger.info(f"Created batcher: {batcher}")
        try:
            model = model_loader.load()
        except Exception:
            self.logger.exception(
                f"Could not create model from {type(model_loader)}"
            )
            raise
        else:
            self.logger.info(f"Created model: {model}")
        return BatchableServiceImpl(batcher, model, self.logger)
