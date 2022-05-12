from abc import ABC, abstractmethod
from typing import Dict, Protocol, Tuple, TypeVar

import numpy as np
from launch_api.core import Service

NamedArrays = Dict[str, np.ndarray]

Shape = Tuple[int, ...]

NamedShapes = Dict[str, Shape]

B = TypeVar("B", np.ndarray, NamedArrays)


class Model(Protocol[B]):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> B:
        """Model inference: accepts an array or arrays as input and produces one or more arrays as output."""
        raise NotImplementedError


ModelSingle = Model[np.ndarray]

ModelNamed = Model[NamedArrays]


class TritonModel(Service[NamedArrays, NamedArrays], ABC):
    """A kind of service that only executes a model's forward pass for inference."""


class SpecsTritonModel(TritonModel, ABC):
    """A model that also describes the shapes of its input and output tensors."""

    input_shapes: NamedShapes
    output_shapes: NamedShapes
