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


class RunnableImageFlavor(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    This is the entity-layer class for the Model Bundle flavor of a runnable image.
    """

    class MetaOapg:
        required = {
            "flavor",
            "protocol",
            "tag",
            "repository",
            "command",
        }

        class properties:
            class command(schemas.ListSchema):
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
                ) -> "command":
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )

                def __getitem__(self, i: int) -> MetaOapg.items:
                    return super().__getitem__(i)

            class flavor(schemas.EnumBase, schemas.StrSchema):
                class MetaOapg:
                    enum_value_to_name = {
                        "runnable_image": "RUNNABLE_IMAGE",
                    }

                @schemas.classproperty
                def RUNNABLE_IMAGE(cls):
                    return cls("runnable_image")

            class protocol(schemas.EnumBase, schemas.StrSchema):
                class MetaOapg:
                    enum_value_to_name = {
                        "http": "HTTP",
                    }

                @schemas.classproperty
                def HTTP(cls):
                    return cls("http")

            repository = schemas.StrSchema
            tag = schemas.StrSchema

            class env(schemas.DictSchema):
                class MetaOapg:
                    additional_properties = schemas.StrSchema

                def __getitem__(
                    self,
                    name: typing.Union[str,],
                ) -> MetaOapg.additional_properties:
                    # dict_instance[name] accessor
                    return super().__getitem__(name)

                def get_item_oapg(
                    self,
                    name: typing.Union[str,],
                ) -> MetaOapg.additional_properties:
                    return super().get_item_oapg(name)

                def __new__(
                    cls,
                    *_args: typing.Union[
                        dict,
                        frozendict.frozendict,
                    ],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                    **kwargs: typing.Union[
                        MetaOapg.additional_properties,
                        str,
                    ],
                ) -> "env":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )

            healthcheck_route = schemas.StrSchema
            predict_route = schemas.StrSchema
            readiness_initial_delay_seconds = schemas.IntSchema
            __annotations__ = {
                "command": command,
                "flavor": flavor,
                "protocol": protocol,
                "repository": repository,
                "tag": tag,
                "env": env,
                "healthcheck_route": healthcheck_route,
                "predict_route": predict_route,
                "readiness_initial_delay_seconds": readiness_initial_delay_seconds,
            }

    flavor: MetaOapg.properties.flavor
    protocol: MetaOapg.properties.protocol
    tag: MetaOapg.properties.tag
    repository: MetaOapg.properties.repository
    command: MetaOapg.properties.command

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["command"]) -> MetaOapg.properties.command:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["flavor"]) -> MetaOapg.properties.flavor:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["protocol"]) -> MetaOapg.properties.protocol:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["repository"]) -> MetaOapg.properties.repository:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["tag"]) -> MetaOapg.properties.tag:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["env"]) -> MetaOapg.properties.env:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["healthcheck_route"]
    ) -> MetaOapg.properties.healthcheck_route:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["predict_route"]) -> MetaOapg.properties.predict_route:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["readiness_initial_delay_seconds"]
    ) -> MetaOapg.properties.readiness_initial_delay_seconds:
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "command",
                "flavor",
                "protocol",
                "repository",
                "tag",
                "env",
                "healthcheck_route",
                "predict_route",
                "readiness_initial_delay_seconds",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["command"]) -> MetaOapg.properties.command:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["flavor"]) -> MetaOapg.properties.flavor:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["protocol"]) -> MetaOapg.properties.protocol:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["repository"]) -> MetaOapg.properties.repository:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["tag"]) -> MetaOapg.properties.tag:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["env"]
    ) -> typing.Union[MetaOapg.properties.env, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["healthcheck_route"]
    ) -> typing.Union[MetaOapg.properties.healthcheck_route, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["predict_route"]
    ) -> typing.Union[MetaOapg.properties.predict_route, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["readiness_initial_delay_seconds"]
    ) -> typing.Union[MetaOapg.properties.readiness_initial_delay_seconds, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "command",
                "flavor",
                "protocol",
                "repository",
                "tag",
                "env",
                "healthcheck_route",
                "predict_route",
                "readiness_initial_delay_seconds",
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
        flavor: typing.Union[
            MetaOapg.properties.flavor,
            str,
        ],
        protocol: typing.Union[
            MetaOapg.properties.protocol,
            str,
        ],
        tag: typing.Union[
            MetaOapg.properties.tag,
            str,
        ],
        repository: typing.Union[
            MetaOapg.properties.repository,
            str,
        ],
        command: typing.Union[
            MetaOapg.properties.command,
            list,
            tuple,
        ],
        env: typing.Union[MetaOapg.properties.env, dict, frozendict.frozendict, schemas.Unset] = schemas.unset,
        healthcheck_route: typing.Union[MetaOapg.properties.healthcheck_route, str, schemas.Unset] = schemas.unset,
        predict_route: typing.Union[MetaOapg.properties.predict_route, str, schemas.Unset] = schemas.unset,
        readiness_initial_delay_seconds: typing.Union[
            MetaOapg.properties.readiness_initial_delay_seconds, decimal.Decimal, int, schemas.Unset
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
    ) -> "RunnableImageFlavor":
        return super().__new__(
            cls,
            *_args,
            flavor=flavor,
            protocol=protocol,
            tag=tag,
            repository=repository,
            command=command,
            env=env,
            healthcheck_route=healthcheck_route,
            predict_route=predict_route,
            readiness_initial_delay_seconds=readiness_initial_delay_seconds,
            _configuration=_configuration,
            **kwargs,
        )
