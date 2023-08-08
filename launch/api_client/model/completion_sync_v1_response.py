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


class CompletionSyncV1Response(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    Response object for a synchronous prompt completion task.
    """

    class MetaOapg:
        required = {
            "request_id",
        }

        class properties:
            request_id = schemas.StrSchema

            @staticmethod
            def output() -> typing.Type["CompletionOutput"]:
                return CompletionOutput

            __annotations__ = {
                "request_id": request_id,
                "output": output,
            }

    request_id: MetaOapg.properties.request_id

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["request_id"]) -> MetaOapg.properties.request_id:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["output"]) -> "CompletionOutput":
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "request_id",
                "output",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["request_id"]) -> MetaOapg.properties.request_id:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["output"]
    ) -> typing.Union["CompletionOutput", schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "request_id",
                "output",
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
        request_id: typing.Union[
            MetaOapg.properties.request_id,
            str,
        ],
        output: typing.Union["CompletionOutput", schemas.Unset] = schemas.unset,
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
    ) -> "CompletionSyncV1Response":
        return super().__new__(
            cls,
            *_args,
            request_id=request_id,
            output=output,
            _configuration=_configuration,
            **kwargs,
        )


from launch.api_client.model.completion_output import CompletionOutput
