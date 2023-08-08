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


class ModelEndpointDeploymentState(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    This is the entity-layer class for the deployment settings related to a Model Endpoint.
    """

    class MetaOapg:
        required = {
            "max_workers",
            "min_workers",
            "per_worker",
        }

        class properties:
            class max_workers(schemas.IntSchema):
                class MetaOapg:
                    inclusive_minimum = 0

            class min_workers(schemas.IntSchema):
                class MetaOapg:
                    inclusive_minimum = 0

            per_worker = schemas.IntSchema

            class available_workers(schemas.IntSchema):
                class MetaOapg:
                    inclusive_minimum = 0

            class unavailable_workers(schemas.IntSchema):
                class MetaOapg:
                    inclusive_minimum = 0

            __annotations__ = {
                "max_workers": max_workers,
                "min_workers": min_workers,
                "per_worker": per_worker,
                "available_workers": available_workers,
                "unavailable_workers": unavailable_workers,
            }

    max_workers: MetaOapg.properties.max_workers
    min_workers: MetaOapg.properties.min_workers
    per_worker: MetaOapg.properties.per_worker

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["max_workers"]) -> MetaOapg.properties.max_workers:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["min_workers"]) -> MetaOapg.properties.min_workers:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["per_worker"]) -> MetaOapg.properties.per_worker:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["available_workers"]
    ) -> MetaOapg.properties.available_workers:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["unavailable_workers"]
    ) -> MetaOapg.properties.unavailable_workers:
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "max_workers",
                "min_workers",
                "per_worker",
                "available_workers",
                "unavailable_workers",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["max_workers"]) -> MetaOapg.properties.max_workers:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["min_workers"]) -> MetaOapg.properties.min_workers:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["per_worker"]) -> MetaOapg.properties.per_worker:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["available_workers"]
    ) -> typing.Union[MetaOapg.properties.available_workers, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["unavailable_workers"]
    ) -> typing.Union[MetaOapg.properties.unavailable_workers, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "max_workers",
                "min_workers",
                "per_worker",
                "available_workers",
                "unavailable_workers",
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
        max_workers: typing.Union[
            MetaOapg.properties.max_workers,
            decimal.Decimal,
            int,
        ],
        min_workers: typing.Union[
            MetaOapg.properties.min_workers,
            decimal.Decimal,
            int,
        ],
        per_worker: typing.Union[
            MetaOapg.properties.per_worker,
            decimal.Decimal,
            int,
        ],
        available_workers: typing.Union[
            MetaOapg.properties.available_workers, decimal.Decimal, int, schemas.Unset
        ] = schemas.unset,
        unavailable_workers: typing.Union[
            MetaOapg.properties.unavailable_workers, decimal.Decimal, int, schemas.Unset
        ] = schemas.unset,
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
    ) -> "ModelEndpointDeploymentState":
        return super().__new__(
            cls,
            *_args,
            max_workers=max_workers,
            min_workers=min_workers,
            per_worker=per_worker,
            available_workers=available_workers,
            unavailable_workers=unavailable_workers,
            _configuration=_configuration,
            **kwargs,
        )
