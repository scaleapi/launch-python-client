from dataclasses import dataclass
from typing import Sequence

import numpy as np
from launch_api.batching.implementation import LoaderBatchableServiceImpl
from launch_api.batching.types import Batcher
from launch_api.loader import LoaderSpec
from launch_api.model import ModelSingle
from launch_api.types import I, O
from scaleml.utils.logging import logger_name, make_logger

logger = make_logger(logger_name())

__all__: Sequence[str] = (
    "BatcherSingle",
    "LoaderSingleBatchableService",
)


BatcherSingle = Batcher[I, O, np.ndarray]


@dataclass(frozen=True, init=False)
class LoaderSingleBatchableService(
    LoaderBatchableServiceImpl[I, O, np.ndarray]
):
    def __init__(
        self,
        batcher: LoaderSpec[BatcherSingle[I, O]],
        model: LoaderSpec[ModelSingle],
    ) -> None:
        super().__init__(batcher, model, logger)


# @dataclass
# class BatcherFromProcessor(BatcherSingle[I, O]):
#     processor: ProcessorSingle[I, O]
#     batch_index: int = 0
#
#     def batch(self, requests: List[I]) -> np.ndarray:
#         model_inputs = [self.processor.preprocess(r) for r in requests]
#         return np.stack(model_inputs, axis=self.batch_index)
#
#     def unbatch(self, preds: np.ndarray) -> List[O]:
#         s = preds.shape
#         n_batches = s[self.batch_index]
#
#
#
#         for i in range(n_batches):
#             p = preds.__getitem__()
#             self.processor.postprocess()
