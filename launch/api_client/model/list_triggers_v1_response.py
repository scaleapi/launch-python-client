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


class ListTriggersV1Response(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "triggers",
        }

        class properties:
            class triggers(schemas.ListSchema):
                class MetaOapg:
                    @staticmethod
                    def items() -> typing.Type["GetTriggerV1Response"]:
                        return GetTriggerV1Response

                def __new__(
                    cls,
                    _arg: typing.Union[typing.Tuple["GetTriggerV1Response"], typing.List["GetTriggerV1Response"]],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> "triggers":
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )

                def __getitem__(self, i: int) -> "GetTriggerV1Response":
                    return super().__getitem__(i)

            __annotations__ = {
                "triggers": triggers,
            }

    triggers: MetaOapg.properties.triggers

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["triggers"]) -> MetaOapg.properties.triggers:
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal["triggers",],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["triggers"]) -> MetaOapg.properties.triggers:
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal["triggers",],
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
        triggers: typing.Union[
            MetaOapg.properties.triggers,
            list,
            tuple,
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
    ) -> "ListTriggersV1Response":
        return super().__new__(
            cls,
            *_args,
            triggers=triggers,
            _configuration=_configuration,
            **kwargs,
        )


from launch.api_client.model.get_trigger_v1_response import GetTriggerV1Response
