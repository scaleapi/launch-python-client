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
from launch_client import schemas  # noqa: F401

class GetFineTuneEventsResponse(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "events",
        }

        class properties:
            class events(schemas.ListSchema):
                class MetaOapg:
                    @staticmethod
                    def items() -> typing.Type["LLMFineTuneEvent"]:
                        return LLMFineTuneEvent
                def __new__(
                    cls,
                    _arg: typing.Union[
                        typing.Tuple["LLMFineTuneEvent"], typing.List["LLMFineTuneEvent"]
                    ],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> "events":
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
                def __getitem__(self, i: int) -> "LLMFineTuneEvent":
                    return super().__getitem__(i)
            __annotations__ = {
                "events": events,
            }
    events: MetaOapg.properties.events

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["events"]
    ) -> MetaOapg.properties.events: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "events",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["events"]
    ) -> MetaOapg.properties.events: ...
    @typing.overload
    def get_item_oapg(
        self, name: str
    ) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "events",
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
        events: typing.Union[
            MetaOapg.properties.events,
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
    ) -> "GetFineTuneEventsResponse":
        return super().__new__(
            cls,
            *_args,
            events=events,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.llm_fine_tune_event import LLMFineTuneEvent
