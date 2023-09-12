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


class CustomFramework(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    This is the entity-layer class for a custom framework specification.
    """

    class MetaOapg:
        required = {
            "image_repository",
            "framework_type",
            "image_tag",
        }

        class properties:
            class framework_type(schemas.EnumBase, schemas.StrSchema):
                class MetaOapg:
                    enum_value_to_name = {
                        "custom_base_image": "CUSTOM_BASE_IMAGE",
                    }

                @schemas.classproperty
                def CUSTOM_BASE_IMAGE(cls):
                    return cls("custom_base_image")

            image_repository = schemas.StrSchema
            image_tag = schemas.StrSchema
            __annotations__ = {
                "framework_type": framework_type,
                "image_repository": image_repository,
                "image_tag": image_tag,
            }

    image_repository: MetaOapg.properties.image_repository
    framework_type: MetaOapg.properties.framework_type
    image_tag: MetaOapg.properties.image_tag

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["framework_type"]
    ) -> MetaOapg.properties.framework_type:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["image_repository"]
    ) -> MetaOapg.properties.image_repository:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["image_tag"]
    ) -> MetaOapg.properties.image_tag:
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "framework_type",
                "image_repository",
                "image_tag",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["framework_type"]
    ) -> MetaOapg.properties.framework_type:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["image_repository"]
    ) -> MetaOapg.properties.image_repository:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["image_tag"]
    ) -> MetaOapg.properties.image_tag:
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "framework_type",
                "image_repository",
                "image_tag",
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
        image_repository: typing.Union[
            MetaOapg.properties.image_repository,
            str,
        ],
        framework_type: typing.Union[
            MetaOapg.properties.framework_type,
            str,
        ],
        image_tag: typing.Union[
            MetaOapg.properties.image_tag,
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
    ) -> "CustomFramework":
        return super().__new__(
            cls,
            *_args,
            image_repository=image_repository,
            framework_type=framework_type,
            image_tag=image_tag,
            _configuration=_configuration,
            **kwargs,
        )
