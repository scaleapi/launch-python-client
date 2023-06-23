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

class CreateDockerImageBatchJobBundleV1Request(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "image_repository",
            "name",
            "image_tag",
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
            image_repository = schemas.StrSchema
            image_tag = schemas.StrSchema
            name = schemas.StrSchema

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
            mount_location = schemas.StrSchema

            class resource_requests(
                schemas.ComposedSchema,
            ):
                class MetaOapg:
                    @classmethod
                    @functools.lru_cache()
                    def all_of(cls):
                        # we need this here to make our import statements work
                        # we must store _composed_schemas in here so the code is only run
                        # when we invoke this method. If we kept this at the class
                        # level we would get an error because the class level
                        # code would be run when this module is imported, and these composed
                        # classes don't exist yet because their module has not finished
                        # loading
                        return [
                            CreateDockerImageBatchJobResourceRequests,
                        ]
                def __new__(
                    cls,
                    *_args: typing.Union[
                        dict,
                        frozendict.frozendict,
                        str,
                        date,
                        datetime,
                        uuid.UUID,
                        int,
                        float,
                        decimal.Decimal,
                        bool,
                        None,
                        list,
                        tuple,
                        bytes,
                        io.FileIO,
                        io.BufferedReader,
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
                ) -> "resource_requests":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            __annotations__ = {
                "command": command,
                "image_repository": image_repository,
                "image_tag": image_tag,
                "name": name,
                "env": env,
                "mount_location": mount_location,
                "resource_requests": resource_requests,
            }
    image_repository: MetaOapg.properties.image_repository
    name: MetaOapg.properties.name
    image_tag: MetaOapg.properties.image_tag
    command: MetaOapg.properties.command

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["command"]) -> MetaOapg.properties.command: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["image_repository"]
    ) -> MetaOapg.properties.image_repository: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["image_tag"]) -> MetaOapg.properties.image_tag: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["env"]) -> MetaOapg.properties.env: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["mount_location"]) -> MetaOapg.properties.mount_location: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["resource_requests"]
    ) -> MetaOapg.properties.resource_requests: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "command",
                "image_repository",
                "image_tag",
                "name",
                "env",
                "mount_location",
                "resource_requests",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["command"]) -> MetaOapg.properties.command: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["image_repository"]
    ) -> MetaOapg.properties.image_repository: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["image_tag"]) -> MetaOapg.properties.image_tag: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["env"]
    ) -> typing.Union[MetaOapg.properties.env, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["mount_location"]
    ) -> typing.Union[MetaOapg.properties.mount_location, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["resource_requests"]
    ) -> typing.Union[MetaOapg.properties.resource_requests, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "command",
                "image_repository",
                "image_tag",
                "name",
                "env",
                "mount_location",
                "resource_requests",
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
        name: typing.Union[
            MetaOapg.properties.name,
            str,
        ],
        image_tag: typing.Union[
            MetaOapg.properties.image_tag,
            str,
        ],
        command: typing.Union[
            MetaOapg.properties.command,
            list,
            tuple,
        ],
        env: typing.Union[MetaOapg.properties.env, dict, frozendict.frozendict, schemas.Unset] = schemas.unset,
        mount_location: typing.Union[MetaOapg.properties.mount_location, str, schemas.Unset] = schemas.unset,
        resource_requests: typing.Union[
            MetaOapg.properties.resource_requests,
            dict,
            frozendict.frozendict,
            str,
            date,
            datetime,
            uuid.UUID,
            int,
            float,
            decimal.Decimal,
            bool,
            None,
            list,
            tuple,
            bytes,
            io.FileIO,
            io.BufferedReader,
            schemas.Unset,
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
    ) -> "CreateDockerImageBatchJobBundleV1Request":
        return super().__new__(
            cls,
            *_args,
            image_repository=image_repository,
            name=name,
            image_tag=image_tag,
            command=command,
            env=env,
            mount_location=mount_location,
            resource_requests=resource_requests,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.create_docker_image_batch_job_resource_requests import (
    CreateDockerImageBatchJobResourceRequests,
)
