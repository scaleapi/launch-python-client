# flake8: noqa
from typing import Sequence

from launch_api.model_service.named_arrays import *
from launch_api.model_service.single_array import *
from launch_api.model_service.types import *

__all__: Sequence[str] = (
    # single element inference service concepts
    "Processor",
    "InferenceService",
    # batch is a single array: B=np.ndarray
    "ProcessorSingle",
    "LoaderSingleBatchableService",
    # batch is a set of named arrays: B=Dict[str, np.ndarray]
    "ProcessorNamed",
    "LoaderNamedInferenceService",
)
