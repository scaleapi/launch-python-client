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

class CompletionSyncV1Request(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    Request object for a synchronous prompt completion task.
    """

    class MetaOapg:
        required = {
            "max_new_tokens",
            "temperature",
            "prompt",
        }

        class properties:
            max_new_tokens = schemas.IntSchema
            prompt = schemas.StrSchema

            class temperature(schemas.NumberSchema):
                pass
            return_token_log_probs = schemas.BoolSchema

            class stop_sequences(schemas.ListSchema):
                class MetaOapg:
                    items = schemas.StrSchema
                def __new__(
                    cls,
                    _arg: typing.Union[
                        typing.Tuple[
                            typing.Union[
                                MetaOapg.items,
                                str,
                            ]
                        ],
                        typing.List[
                            typing.Union[
                                MetaOapg.items,
                                str,
                            ]
                        ],
                    ],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> "stop_sequences":
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
                def __getitem__(self, i: int) -> MetaOapg.items:
                    return super().__getitem__(i)
            __annotations__ = {
                "max_new_tokens": max_new_tokens,
                "prompt": prompt,
                "temperature": temperature,
                "return_token_log_probs": return_token_log_probs,
                "stop_sequences": stop_sequences,
            }
    max_new_tokens: MetaOapg.properties.max_new_tokens
    temperature: MetaOapg.properties.temperature
    prompt: MetaOapg.properties.prompt

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["max_new_tokens"]) -> MetaOapg.properties.max_new_tokens: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["prompt"]) -> MetaOapg.properties.prompt: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["temperature"]) -> MetaOapg.properties.temperature: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["return_token_log_probs"]
    ) -> MetaOapg.properties.return_token_log_probs: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["stop_sequences"]) -> MetaOapg.properties.stop_sequences: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "max_new_tokens",
                "prompt",
                "temperature",
                "return_token_log_probs",
                "stop_sequences",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["max_new_tokens"]
    ) -> MetaOapg.properties.max_new_tokens: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["prompt"]) -> MetaOapg.properties.prompt: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["temperature"]) -> MetaOapg.properties.temperature: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["return_token_log_probs"]
    ) -> typing.Union[MetaOapg.properties.return_token_log_probs, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["stop_sequences"]
    ) -> typing.Union[MetaOapg.properties.stop_sequences, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "max_new_tokens",
                "prompt",
                "temperature",
                "return_token_log_probs",
                "stop_sequences",
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
        max_new_tokens: typing.Union[
            MetaOapg.properties.max_new_tokens,
            decimal.Decimal,
            int,
        ],
        temperature: typing.Union[
            MetaOapg.properties.temperature,
            decimal.Decimal,
            int,
            float,
        ],
        prompt: typing.Union[
            MetaOapg.properties.prompt,
            str,
        ],
        return_token_log_probs: typing.Union[
            MetaOapg.properties.return_token_log_probs, bool, schemas.Unset
        ] = schemas.unset,
        stop_sequences: typing.Union[MetaOapg.properties.stop_sequences, list, tuple, schemas.Unset] = schemas.unset,
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
    ) -> "CompletionSyncV1Request":
        return super().__new__(
            cls,
            *_args,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prompt=prompt,
            return_token_log_probs=return_token_log_probs,
            stop_sequences=stop_sequences,
            _configuration=_configuration,
            **kwargs,
        )
