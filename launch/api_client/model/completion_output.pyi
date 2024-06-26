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

class CompletionOutput(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "num_prompt_tokens",
            "num_completion_tokens",
            "text",
        }

        class properties:
            num_completion_tokens = schemas.IntSchema
            num_prompt_tokens = schemas.IntSchema
            text = schemas.StrSchema

            class tokens(schemas.ListSchema):
                class MetaOapg:
                    @staticmethod
                    def items() -> typing.Type["TokenOutput"]:
                        return TokenOutput
                def __new__(
                    cls,
                    _arg: typing.Union[typing.Tuple["TokenOutput"], typing.List["TokenOutput"]],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> "tokens":
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
                def __getitem__(self, i: int) -> "TokenOutput":
                    return super().__getitem__(i)
            __annotations__ = {
                "num_completion_tokens": num_completion_tokens,
                "num_prompt_tokens": num_prompt_tokens,
                "text": text,
                "tokens": tokens,
            }
    num_prompt_tokens: MetaOapg.properties.num_prompt_tokens
    num_completion_tokens: MetaOapg.properties.num_completion_tokens
    text: MetaOapg.properties.text

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["num_completion_tokens"]
    ) -> MetaOapg.properties.num_completion_tokens: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["num_prompt_tokens"]
    ) -> MetaOapg.properties.num_prompt_tokens: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["text"]) -> MetaOapg.properties.text: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["tokens"]) -> MetaOapg.properties.tokens: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "num_completion_tokens",
                "num_prompt_tokens",
                "text",
                "tokens",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["num_completion_tokens"]
    ) -> MetaOapg.properties.num_completion_tokens: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["num_prompt_tokens"]
    ) -> MetaOapg.properties.num_prompt_tokens: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["text"]) -> MetaOapg.properties.text: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["tokens"]
    ) -> typing.Union[MetaOapg.properties.tokens, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "num_completion_tokens",
                "num_prompt_tokens",
                "text",
                "tokens",
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
        num_prompt_tokens: typing.Union[
            MetaOapg.properties.num_prompt_tokens,
            decimal.Decimal,
            int,
        ],
        num_completion_tokens: typing.Union[
            MetaOapg.properties.num_completion_tokens,
            decimal.Decimal,
            int,
        ],
        text: typing.Union[
            MetaOapg.properties.text,
            str,
        ],
        tokens: typing.Union[MetaOapg.properties.tokens, list, tuple, schemas.Unset] = schemas.unset,
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
    ) -> "CompletionOutput":
        return super().__new__(
            cls,
            *_args,
            num_prompt_tokens=num_prompt_tokens,
            num_completion_tokens=num_completion_tokens,
            text=text,
            tokens=tokens,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.token_output import TokenOutput
