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


class DockerImageBatchJobBundleV1Response(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "image_repository",
            "name",
            "created_at",
            "id",
            "image_tag",
            "env",
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

            created_at = schemas.DateTimeSchema

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

            id = schemas.StrSchema
            image_repository = schemas.StrSchema
            image_tag = schemas.StrSchema
            name = schemas.StrSchema
            cpus = schemas.StrSchema
            gpu_type = schemas.StrSchema
            gpus = schemas.IntSchema
            memory = schemas.StrSchema
            mount_location = schemas.StrSchema
            public = schemas.BoolSchema
            storage = schemas.StrSchema
            __annotations__ = {
                "command": command,
                "created_at": created_at,
                "env": env,
                "id": id,
                "image_repository": image_repository,
                "image_tag": image_tag,
                "name": name,
                "cpus": cpus,
                "gpu_type": gpu_type,
                "gpus": gpus,
                "memory": memory,
                "mount_location": mount_location,
                "public": public,
                "storage": storage,
            }

    image_repository: MetaOapg.properties.image_repository
    name: MetaOapg.properties.name
    created_at: MetaOapg.properties.created_at
    id: MetaOapg.properties.id
    image_tag: MetaOapg.properties.image_tag
    env: MetaOapg.properties.env
    command: MetaOapg.properties.command

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["command"]) -> MetaOapg.properties.command:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["created_at"]) -> MetaOapg.properties.created_at:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["env"]) -> MetaOapg.properties.env:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["image_repository"]) -> MetaOapg.properties.image_repository:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["image_tag"]) -> MetaOapg.properties.image_tag:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["cpus"]) -> MetaOapg.properties.cpus:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["gpu_type"]) -> MetaOapg.properties.gpu_type:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["gpus"]) -> MetaOapg.properties.gpus:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["memory"]) -> MetaOapg.properties.memory:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["mount_location"]) -> MetaOapg.properties.mount_location:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["public"]) -> MetaOapg.properties.public:
        ...

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["storage"]) -> MetaOapg.properties.storage:
        ...

    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema:
        ...

    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "command",
                "created_at",
                "env",
                "id",
                "image_repository",
                "image_tag",
                "name",
                "cpus",
                "gpu_type",
                "gpus",
                "memory",
                "mount_location",
                "public",
                "storage",
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
    def get_item_oapg(self, name: typing_extensions.Literal["created_at"]) -> MetaOapg.properties.created_at:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["env"]) -> MetaOapg.properties.env:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["image_repository"]
    ) -> MetaOapg.properties.image_repository:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["image_tag"]) -> MetaOapg.properties.image_tag:
        ...

    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["cpus"]
    ) -> typing.Union[MetaOapg.properties.cpus, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["gpu_type"]
    ) -> typing.Union[MetaOapg.properties.gpu_type, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["gpus"]
    ) -> typing.Union[MetaOapg.properties.gpus, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["memory"]
    ) -> typing.Union[MetaOapg.properties.memory, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["mount_location"]
    ) -> typing.Union[MetaOapg.properties.mount_location, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["public"]
    ) -> typing.Union[MetaOapg.properties.public, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["storage"]
    ) -> typing.Union[MetaOapg.properties.storage, schemas.Unset]:
        ...

    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]:
        ...

    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "command",
                "created_at",
                "env",
                "id",
                "image_repository",
                "image_tag",
                "name",
                "cpus",
                "gpu_type",
                "gpus",
                "memory",
                "mount_location",
                "public",
                "storage",
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
        created_at: typing.Union[
            MetaOapg.properties.created_at,
            str,
            datetime,
        ],
        id: typing.Union[
            MetaOapg.properties.id,
            str,
        ],
        image_tag: typing.Union[
            MetaOapg.properties.image_tag,
            str,
        ],
        env: typing.Union[
            MetaOapg.properties.env,
            dict,
            frozendict.frozendict,
        ],
        command: typing.Union[
            MetaOapg.properties.command,
            list,
            tuple,
        ],
        cpus: typing.Union[MetaOapg.properties.cpus, str, schemas.Unset] = schemas.unset,
        gpu_type: typing.Union[MetaOapg.properties.gpu_type, str, schemas.Unset] = schemas.unset,
        gpus: typing.Union[MetaOapg.properties.gpus, decimal.Decimal, int, schemas.Unset] = schemas.unset,
        memory: typing.Union[MetaOapg.properties.memory, str, schemas.Unset] = schemas.unset,
        mount_location: typing.Union[MetaOapg.properties.mount_location, str, schemas.Unset] = schemas.unset,
        public: typing.Union[MetaOapg.properties.public, bool, schemas.Unset] = schemas.unset,
        storage: typing.Union[MetaOapg.properties.storage, str, schemas.Unset] = schemas.unset,
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
    ) -> "DockerImageBatchJobBundleV1Response":
        return super().__new__(
            cls,
            *_args,
            image_repository=image_repository,
            name=name,
            created_at=created_at,
            id=id,
            image_tag=image_tag,
            env=env,
            command=command,
            cpus=cpus,
            gpu_type=gpu_type,
            gpus=gpus,
            memory=memory,
            mount_location=mount_location,
            public=public,
            storage=storage,
            _configuration=_configuration,
            **kwargs,
        )
