from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

from launch_api.core import Service
from launch_api.object_conf_from_std_ml_serve import ObjectConf
from launch_api.types import I, O

__all__: Sequence[str] = (
    "Loader",
    "ServiceLoader",
    "S",
)

T = TypeVar("T")


class Loader(Generic[T], ABC):
    @abstractmethod
    def load(self) -> T:
        raise NotImplementedError


S = TypeVar("S", bound=Service)


class ServiceLoader(Generic[I, O, S], Loader[S[I, O]], ABC):
    pass


L = TypeVar("L", bound=Loader)


class LoaderSpec(ObjectConf[L]):
    """"""
