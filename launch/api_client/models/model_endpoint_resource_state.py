# coding: utf-8

"""
    launch

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)

    The version of the OpenAPI document: 1.0.0
    Generated by OpenAPI Generator (https://openapi-generator.tech)

    Do not edit the class manually.
"""  # noqa: E501


from __future__ import annotations

import json
import pprint
import re  # noqa: F401
from typing import Any, ClassVar, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, StrictBool
from typing_extensions import Annotated, Self

from launch.api_client.models.cpus import Cpus
from launch.api_client.models.gpu_type import GpuType
from launch.api_client.models.memory import Memory
from launch.api_client.models.storage import Storage


class ModelEndpointResourceState(BaseModel):
    """
    This is the entity-layer class for the resource settings per worker of a Model Endpoint.
    """  # noqa: E501

    cpus: Cpus
    gpu_type: Optional[GpuType] = None
    gpus: Annotated[int, Field(strict=True, ge=0)]
    memory: Memory
    optimize_costs: Optional[StrictBool] = None
    storage: Optional[Storage] = None
    __properties: ClassVar[List[str]] = ["cpus", "gpu_type", "gpus", "memory", "optimize_costs", "storage"]

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        protected_namespaces=(),
    )

    def to_str(self) -> str:
        """Returns the string representation of the model using alias"""
        return pprint.pformat(self.model_dump(by_alias=True))

    def to_json(self) -> str:
        """Returns the JSON representation of the model using alias"""
        # TODO: pydantic v2: use .model_dump_json(by_alias=True, exclude_unset=True) instead
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> Optional[Self]:
        """Create an instance of ModelEndpointResourceState from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self) -> Dict[str, Any]:
        """Return the dictionary representation of the model using alias.

        This has the following differences from calling pydantic's
        `self.model_dump(by_alias=True)`:

        * `None` is only added to the output dict for nullable fields that
          were set at model initialization. Other fields with value `None`
          are ignored.
        """
        excluded_fields: Set[str] = set([])

        _dict = self.model_dump(
            by_alias=True,
            exclude=excluded_fields,
            exclude_none=True,
        )
        # override the default output from pydantic by calling `to_dict()` of cpus
        if self.cpus:
            _dict["cpus"] = self.cpus.to_dict()
        # override the default output from pydantic by calling `to_dict()` of memory
        if self.memory:
            _dict["memory"] = self.memory.to_dict()
        # override the default output from pydantic by calling `to_dict()` of storage
        if self.storage:
            _dict["storage"] = self.storage.to_dict()
        return _dict

    @classmethod
    def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional[Self]:
        """Create an instance of ModelEndpointResourceState from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return cls.model_validate(obj)

        _obj = cls.model_validate(
            {
                "cpus": Cpus.from_dict(obj["cpus"]) if obj.get("cpus") is not None else None,
                "gpu_type": obj.get("gpu_type"),
                "gpus": obj.get("gpus"),
                "memory": Memory.from_dict(obj["memory"]) if obj.get("memory") is not None else None,
                "optimize_costs": obj.get("optimize_costs"),
                "storage": Storage.from_dict(obj["storage"]) if obj.get("storage") is not None else None,
            }
        )
        return _obj
