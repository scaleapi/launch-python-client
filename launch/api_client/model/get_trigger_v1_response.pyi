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

class GetTriggerV1Response(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "owner",
            "cron_schedule",
            "docker_image_batch_job_bundle_id",
            "name",
            "created_at",
            "id",
            "created_by",
        }

        class properties:
            created_at = schemas.DateTimeSchema
            created_by = schemas.StrSchema
            cron_schedule = schemas.StrSchema
            docker_image_batch_job_bundle_id = schemas.StrSchema
            id = schemas.StrSchema
            name = schemas.StrSchema
            owner = schemas.StrSchema
            default_job_config = schemas.DictSchema

            class default_job_metadata(schemas.DictSchema):
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
                ) -> "default_job_metadata":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            __annotations__ = {
                "created_at": created_at,
                "created_by": created_by,
                "cron_schedule": cron_schedule,
                "docker_image_batch_job_bundle_id": docker_image_batch_job_bundle_id,
                "id": id,
                "name": name,
                "owner": owner,
                "default_job_config": default_job_config,
                "default_job_metadata": default_job_metadata,
            }
    owner: MetaOapg.properties.owner
    cron_schedule: MetaOapg.properties.cron_schedule
    docker_image_batch_job_bundle_id: MetaOapg.properties.docker_image_batch_job_bundle_id
    name: MetaOapg.properties.name
    created_at: MetaOapg.properties.created_at
    id: MetaOapg.properties.id
    created_by: MetaOapg.properties.created_by

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["created_at"]) -> MetaOapg.properties.created_at: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["created_by"]) -> MetaOapg.properties.created_by: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["cron_schedule"]) -> MetaOapg.properties.cron_schedule: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["docker_image_batch_job_bundle_id"]
    ) -> MetaOapg.properties.docker_image_batch_job_bundle_id: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["owner"]) -> MetaOapg.properties.owner: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["default_job_config"]
    ) -> MetaOapg.properties.default_job_config: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["default_job_metadata"]
    ) -> MetaOapg.properties.default_job_metadata: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "created_at",
                "created_by",
                "cron_schedule",
                "docker_image_batch_job_bundle_id",
                "id",
                "name",
                "owner",
                "default_job_config",
                "default_job_metadata",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["created_at"]) -> MetaOapg.properties.created_at: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["created_by"]) -> MetaOapg.properties.created_by: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["cron_schedule"]) -> MetaOapg.properties.cron_schedule: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["docker_image_batch_job_bundle_id"]
    ) -> MetaOapg.properties.docker_image_batch_job_bundle_id: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["owner"]) -> MetaOapg.properties.owner: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["default_job_config"]
    ) -> typing.Union[MetaOapg.properties.default_job_config, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["default_job_metadata"]
    ) -> typing.Union[MetaOapg.properties.default_job_metadata, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "created_at",
                "created_by",
                "cron_schedule",
                "docker_image_batch_job_bundle_id",
                "id",
                "name",
                "owner",
                "default_job_config",
                "default_job_metadata",
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
        owner: typing.Union[
            MetaOapg.properties.owner,
            str,
        ],
        cron_schedule: typing.Union[
            MetaOapg.properties.cron_schedule,
            str,
        ],
        docker_image_batch_job_bundle_id: typing.Union[
            MetaOapg.properties.docker_image_batch_job_bundle_id,
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
        created_by: typing.Union[
            MetaOapg.properties.created_by,
            str,
        ],
        default_job_config: typing.Union[
            MetaOapg.properties.default_job_config, dict, frozendict.frozendict, schemas.Unset
        ] = schemas.unset,
        default_job_metadata: typing.Union[
            MetaOapg.properties.default_job_metadata, dict, frozendict.frozendict, schemas.Unset
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
    ) -> "GetTriggerV1Response":
        return super().__new__(
            cls,
            *_args,
            owner=owner,
            cron_schedule=cron_schedule,
            docker_image_batch_job_bundle_id=docker_image_batch_job_bundle_id,
            name=name,
            created_at=created_at,
            id=id,
            created_by=created_by,
            default_job_config=default_job_config,
            default_job_metadata=default_job_metadata,
            _configuration=_configuration,
            **kwargs,
        )
