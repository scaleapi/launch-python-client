# coding: utf-8

"""
    launch

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""

from datetime import date, datetime  # noqa: F401
import decimal  # noqa: F401
import functools  # noqa: F401
import io  # noqa: F401
import re  # noqa: F401
import typing  # noqa: F401
import typing_extensions  # noqa: F401
import uuid  # noqa: F401

import frozendict  # noqa: F401

from launch.api_client import schemas  # noqa: F401


class CreateBatchJobV1Response(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "job_id",
        }

        class properties:
            job_id = schemas.StrSchema
            __annotations__ = {
                "job_id": job_id,
            }

    job_id: MetaOapg.properties.job_id

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["job_id"]) -> MetaOapg.properties.job_id:
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(self, name: typing.Union[typing_extensions.Literal["job_id",], str]):
        # dict_instance[name] accessor
        return super().__getitem__(name)

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["job_id"]) -> MetaOapg.properties.job_id:
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(self, name: typing.Union[typing_extensions.Literal["job_id",], str]):
        return super().get_item_oapg(name)

    def __new__(
        cls,
        *_args: typing.Union[
            dict,
            frozendict.frozendict,
        ],
        job_id: typing.Union[
            MetaOapg.properties.job_id,
            str,
        ],
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
    ) -> "CreateBatchJobV1Response":
        return super().__new__(
            cls,
            *_args,
            job_id=job_id,
            _configuration=_configuration,
            **kwargs,
        )
