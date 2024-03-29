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


class GetFineTuneResponse(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "fine_tune_id",
            "status",
        }

        class properties:
            fine_tune_id = schemas.StrSchema

            @staticmethod
            def status() -> typing.Type["BatchJobStatus"]:
                return BatchJobStatus

            __annotations__ = {
                "fine_tune_id": fine_tune_id,
                "status": status,
            }

    fine_tune_id: MetaOapg.properties.fine_tune_id
    status: "BatchJobStatus"

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["fine_tune_id"]) -> MetaOapg.properties.fine_tune_id:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["status"]) -> "BatchJobStatus":
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "fine_tune_id",
                "status",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["fine_tune_id"]) -> MetaOapg.properties.fine_tune_id:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["status"]) -> "BatchJobStatus":
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "fine_tune_id",
                "status",
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
        fine_tune_id: typing.Union[
            MetaOapg.properties.fine_tune_id,
            str,
        ],
        status: "BatchJobStatus",
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
    ) -> "GetFineTuneResponse":
        return super().__new__(
            cls,
            *_args,
            fine_tune_id=fine_tune_id,
            status=status,
            _configuration=_configuration,
            **kwargs,
        )


from launch.api_client.model.batch_job_status import BatchJobStatus
