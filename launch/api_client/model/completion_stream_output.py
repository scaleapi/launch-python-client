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


class CompletionStreamOutput(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "finished",
            "text",
        }

        class properties:
            finished = schemas.BoolSchema
            text = schemas.StrSchema
            num_completion_tokens = schemas.IntSchema

            @staticmethod
            def token() -> typing.Type["TokenOutput"]:
                return TokenOutput

            __annotations__ = {
                "finished": finished,
                "text": text,
                "num_completion_tokens": num_completion_tokens,
                "token": token,
            }

    finished: MetaOapg.properties.finished
    text: MetaOapg.properties.text

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["finished"]
    ) -> MetaOapg.properties.finished:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["text"]) -> MetaOapg.properties.text:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["num_completion_tokens"]
    ) -> MetaOapg.properties.num_completion_tokens:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["token"]) -> "TokenOutput":
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "finished",
                "text",
                "num_completion_tokens",
                "token",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["finished"]
    ) -> MetaOapg.properties.finished:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["text"]) -> MetaOapg.properties.text:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["num_completion_tokens"]
    ) -> typing.Union[MetaOapg.properties.num_completion_tokens, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["token"]
    ) -> typing.Union["TokenOutput", schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "finished",
                "text",
                "num_completion_tokens",
                "token",
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
        finished: typing.Union[
            MetaOapg.properties.finished,
            bool,
        ],
        text: typing.Union[
            MetaOapg.properties.text,
            str,
        ],
        num_completion_tokens: typing.Union[
            MetaOapg.properties.num_completion_tokens, decimal.Decimal, int, schemas.Unset
        ] = schemas.unset,
        token: typing.Union["TokenOutput", schemas.Unset] = schemas.unset,
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
    ) -> "CompletionStreamOutput":
        return super().__new__(
            cls,
            *_args,
            finished=finished,
            text=text,
            num_completion_tokens=num_completion_tokens,
            token=token,
            _configuration=_configuration,
            **kwargs,
        )


from launch.api_client.model.token_output import TokenOutput
