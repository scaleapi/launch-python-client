from typing import Dict, List, Sequence, TypeVar, Union

JsonObj = Dict[str, Union[str, int, float, dict, list]]
JsonList = List[Union[str, int, float, dict, list]]
JsonVal = Union[JsonObj, JsonList]

I = TypeVar("I")
O = TypeVar("O")

__all__: Sequence[str] = (
    # JSON
    "JsonObj",
    "JsonList",
    "JsonVal",
    # Re-used generic parameters that have same semantics
    "I",
    "O",
)
