from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, Sequence, TypeVar

from scaleml.scaleml.utils import import_by_name

"""
COPIED FROM std-ml-srv: ml_serve.configuration.ObjectConf
"""

__all__: Sequence[str] = ("ObjectConf",)

T = TypeVar("T")


@dataclass(frozen=True)
class ObjectConf(Generic[T]):
    class_name: str
    """The fully-qualified name of a class within the Python environment.
    This value must be the name of the parameterized generic type `T`. 
    """
    args: Optional[Mapping[str, Any]] = None
    """The keyword arguments to apply to construct an instance of `class_name`.
    If `None`, then `class_name` is created with no parameters.
    """
    pass_args_whole: bool = False
    """Controls constructor parameter passing behavior.

    If true, then passes the `args` dictionary directly into the  `class_name`'s  `__init__`. 
    Otherwise, pass the key-value pairs as keyword arguments. 
    Defaults to false.
    """

    def construct(self) -> T:
        """Dynamically loads an instance of `class_name` using `args`.
        Raises an exception on loading failure.
        """
        class_ref = import_by_name(self.class_name, validate=True)

        if self.args is not None:
            args = dict(**self.args)
            if self.pass_args_whole:
                instance = class_ref(args)
            else:
                instance = class_ref(**args)
        else:
            instance = class_ref()
        return instance
