# coding: utf-8

"""
    launch

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""

import decimal  # noqa: F401
import functools  # noqa: F401
import io  # noqa: F401
import re  # noqa: F401
import typing  # noqa: F401
import uuid  # noqa: F401
from datetime import date, datetime  # noqa: F401

import frozendict  # noqa: F401
import typing_extensions  # noqa: F401

from launch.api_client import schemas  # noqa: F401

class GetAsyncTaskV1Response(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "task_id",
            "status",
        }

        class properties:
            @staticmethod
            def status() -> typing.Type["TaskStatus"]:
                return TaskStatus
            task_id = schemas.StrSchema
            result = schemas.AnyTypeSchema
            traceback = schemas.StrSchema
            __annotations__ = {
                "status": status,
                "task_id": task_id,
                "result": result,
                "traceback": traceback,
            }
    task_id: MetaOapg.properties.task_id
    status: "TaskStatus"

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["status"]) -> "TaskStatus": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["task_id"]) -> MetaOapg.properties.task_id: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["result"]) -> MetaOapg.properties.result: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["traceback"]) -> MetaOapg.properties.traceback: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "status",
                "task_id",
                "result",
                "traceback",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["status"]) -> "TaskStatus": ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["task_id"]) -> MetaOapg.properties.task_id: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["result"]
    ) -> typing.Union[MetaOapg.properties.result, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["traceback"]
    ) -> typing.Union[MetaOapg.properties.traceback, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "status",
                "task_id",
                "result",
                "traceback",
            ],
            str,
        ],
    ):
        return super().get_item_oapg(name)
    def __new__(
        cls,
        *_args: typing.Union[
            dict,
            frozendict.frozendict,
        ],
        task_id: typing.Union[
            MetaOapg.properties.task_id,
            str,
        ],
        status: "TaskStatus",
        result: typing.Union[
            MetaOapg.properties.result,
            dict,
            frozendict.frozendict,
            str,
            date,
            datetime,
            uuid.UUID,
            int,
            float,
            decimal.Decimal,
            bool,
            None,
            list,
            tuple,
            bytes,
            io.FileIO,
            io.BufferedReader,
            schemas.Unset,
        ] = schemas.unset,
        traceback: typing.Union[MetaOapg.properties.traceback, str, schemas.Unset] = schemas.unset,
        _configuration: typing.Optional[schemas.Configuration] = None,
        **kwargs: typing.Union[
            schemas.AnyTypeSchema,
            dict,
            frozendict.frozendict,
            str,
            date,
            datetime,
            uuid.UUID,
            int,
            float,
            decimal.Decimal,
            None,
            list,
            tuple,
            bytes,
        ],
    ) -> "GetAsyncTaskV1Response":
        return super().__new__(
            cls,
            *_args,
            task_id=task_id,
            status=status,
            result=result,
            traceback=traceback,
            _configuration=_configuration,
            **kwargs,
        )

from launch.api_client.model.task_status import TaskStatus
