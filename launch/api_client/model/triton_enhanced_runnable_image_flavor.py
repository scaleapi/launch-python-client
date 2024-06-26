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


class TritonEnhancedRunnableImageFlavor(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    For deployments that require tritonserver running in a container.
    """

    class MetaOapg:
        required = {
            "flavor",
            "protocol",
            "tag",
            "repository",
            "triton_commit_tag",
            "triton_model_repository",
            "command",
            "triton_num_cpu",
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
                        "triton_enhanced_runnable_image": "TRITON_ENHANCED_RUNNABLE_IMAGE",
                    }

                @schemas.classproperty
                def TRITON_ENHANCED_RUNNABLE_IMAGE(cls):
                    return cls("triton_enhanced_runnable_image")

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
            triton_commit_tag = schemas.StrSchema
            triton_model_repository = schemas.StrSchema
            triton_num_cpu = schemas.NumberSchema

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
            triton_memory = schemas.StrSchema

            class triton_model_replicas(schemas.DictSchema):
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
                ) -> "triton_model_replicas":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )

            triton_readiness_initial_delay_seconds = schemas.IntSchema
            triton_storage = schemas.StrSchema
            __annotations__ = {
                "command": command,
                "flavor": flavor,
                "protocol": protocol,
                "repository": repository,
                "tag": tag,
                "triton_commit_tag": triton_commit_tag,
                "triton_model_repository": triton_model_repository,
                "triton_num_cpu": triton_num_cpu,
                "env": env,
                "healthcheck_route": healthcheck_route,
                "predict_route": predict_route,
                "readiness_initial_delay_seconds": readiness_initial_delay_seconds,
                "triton_memory": triton_memory,
                "triton_model_replicas": triton_model_replicas,
                "triton_readiness_initial_delay_seconds": triton_readiness_initial_delay_seconds,
                "triton_storage": triton_storage,
            }

    flavor: MetaOapg.properties.flavor
    protocol: MetaOapg.properties.protocol
    tag: MetaOapg.properties.tag
    repository: MetaOapg.properties.repository
    triton_commit_tag: MetaOapg.properties.triton_commit_tag
    triton_model_repository: MetaOapg.properties.triton_model_repository
    command: MetaOapg.properties.command
    triton_num_cpu: MetaOapg.properties.triton_num_cpu

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
    def __getitem__(
        self, name: typing_extensions.Literal["triton_commit_tag"]
    ) -> MetaOapg.properties.triton_commit_tag:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["triton_model_repository"]
    ) -> MetaOapg.properties.triton_model_repository:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["triton_num_cpu"]) -> MetaOapg.properties.triton_num_cpu:
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
    def __getitem__(self, name: typing_extensions.Literal["triton_memory"]) -> MetaOapg.properties.triton_memory:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["triton_model_replicas"]
    ) -> MetaOapg.properties.triton_model_replicas:
        ...

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["triton_readiness_initial_delay_seconds"]
    ) -> MetaOapg.properties.triton_readiness_initial_delay_seconds:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["triton_storage"]) -> MetaOapg.properties.triton_storage:
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
                "triton_commit_tag",
                "triton_model_repository",
                "triton_num_cpu",
                "env",
                "healthcheck_route",
                "predict_route",
                "readiness_initial_delay_seconds",
                "triton_memory",
                "triton_model_replicas",
                "triton_readiness_initial_delay_seconds",
                "triton_storage",
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
        self, name: typing_extensions.Literal["triton_commit_tag"]
    ) -> MetaOapg.properties.triton_commit_tag:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["triton_model_repository"]
    ) -> MetaOapg.properties.triton_model_repository:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["triton_num_cpu"]) -> MetaOapg.properties.triton_num_cpu:
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
    def get_item_oapg(
        self, name: typing_extensions.Literal["triton_memory"]
    ) -> typing.Union[MetaOapg.properties.triton_memory, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["triton_model_replicas"]
    ) -> typing.Union[MetaOapg.properties.triton_model_replicas, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["triton_readiness_initial_delay_seconds"]
    ) -> typing.Union[MetaOapg.properties.triton_readiness_initial_delay_seconds, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["triton_storage"]
    ) -> typing.Union[MetaOapg.properties.triton_storage, schemas.Unset]:
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
                "triton_commit_tag",
                "triton_model_repository",
                "triton_num_cpu",
                "env",
                "healthcheck_route",
                "predict_route",
                "readiness_initial_delay_seconds",
                "triton_memory",
                "triton_model_replicas",
                "triton_readiness_initial_delay_seconds",
                "triton_storage",
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
        triton_commit_tag: typing.Union[
            MetaOapg.properties.triton_commit_tag,
            str,
        ],
        triton_model_repository: typing.Union[
            MetaOapg.properties.triton_model_repository,
            str,
        ],
        command: typing.Union[
            MetaOapg.properties.command,
            list,
            tuple,
        ],
        triton_num_cpu: typing.Union[
            MetaOapg.properties.triton_num_cpu,
            decimal.Decimal,
            int,
            float,
        ],
        env: typing.Union[MetaOapg.properties.env, dict, frozendict.frozendict, schemas.Unset] = schemas.unset,
        healthcheck_route: typing.Union[MetaOapg.properties.healthcheck_route, str, schemas.Unset] = schemas.unset,
        predict_route: typing.Union[MetaOapg.properties.predict_route, str, schemas.Unset] = schemas.unset,
        readiness_initial_delay_seconds: typing.Union[
            MetaOapg.properties.readiness_initial_delay_seconds, decimal.Decimal, int, schemas.Unset
        ] = schemas.unset,
        triton_memory: typing.Union[MetaOapg.properties.triton_memory, str, schemas.Unset] = schemas.unset,
        triton_model_replicas: typing.Union[
            MetaOapg.properties.triton_model_replicas, dict, frozendict.frozendict, schemas.Unset
        ] = schemas.unset,
        triton_readiness_initial_delay_seconds: typing.Union[
            MetaOapg.properties.triton_readiness_initial_delay_seconds, decimal.Decimal, int, schemas.Unset
        ] = schemas.unset,
        triton_storage: typing.Union[MetaOapg.properties.triton_storage, str, schemas.Unset] = schemas.unset,
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
    ) -> "TritonEnhancedRunnableImageFlavor":
        return super().__new__(
            cls,
            *_args,
            flavor=flavor,
            protocol=protocol,
            tag=tag,
            repository=repository,
            triton_commit_tag=triton_commit_tag,
            triton_model_repository=triton_model_repository,
            command=command,
            triton_num_cpu=triton_num_cpu,
            env=env,
            healthcheck_route=healthcheck_route,
            predict_route=predict_route,
            readiness_initial_delay_seconds=readiness_initial_delay_seconds,
            triton_memory=triton_memory,
            triton_model_replicas=triton_model_replicas,
            triton_readiness_initial_delay_seconds=triton_readiness_initial_delay_seconds,
            triton_storage=triton_storage,
            _configuration=_configuration,
            **kwargs,
        )
