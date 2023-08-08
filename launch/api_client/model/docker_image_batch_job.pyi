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

class DockerImageBatchJob(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
        Ref: https://openapi-generator.tech

        Do not edit the class manually.

        This is the entity-layer class for a Docker Image Batch Job, i.e. a batch job
    created via the "supply a docker image for a k8s job" API.
    """

    class MetaOapg:
        required = {
            "owner",
            "created_at",
            "id",
            "created_by",
            "status",
        }

        class properties:
            created_at = schemas.DateTimeSchema
            created_by = schemas.StrSchema
            id = schemas.StrSchema
            owner = schemas.StrSchema

            @staticmethod
            def status() -> typing.Type["BatchJobStatus"]:
                return BatchJobStatus

            class annotations(schemas.DictSchema):
                class MetaOapg:
                    additional_properties = schemas.StrSchema
                def __getitem__(
                    self,
                    name: typing.Union[
                        str,
                    ],
                ) -> MetaOapg.additional_properties:
                    # dict_instance[name] accessor
                    return super().__getitem__(name)
                def get_item_oapg(
                    self,
                    name: typing.Union[
                        str,
                    ],
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
                ) -> "annotations":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            completed_at = schemas.DateTimeSchema
            override_job_max_runtime_s = schemas.IntSchema
            __annotations__ = {
                "created_at": created_at,
                "created_by": created_by,
                "id": id,
                "owner": owner,
                "status": status,
                "annotations": annotations,
                "completed_at": completed_at,
                "override_job_max_runtime_s": override_job_max_runtime_s,
            }
    owner: MetaOapg.properties.owner
    created_at: MetaOapg.properties.created_at
    id: MetaOapg.properties.id
    created_by: MetaOapg.properties.created_by
    status: "BatchJobStatus"

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["created_at"]
    ) -> MetaOapg.properties.created_at: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["created_by"]
    ) -> MetaOapg.properties.created_by: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["owner"]
    ) -> MetaOapg.properties.owner: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["status"]) -> "BatchJobStatus": ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["annotations"]
    ) -> MetaOapg.properties.annotations: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["completed_at"]
    ) -> MetaOapg.properties.completed_at: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["override_job_max_runtime_s"]
    ) -> MetaOapg.properties.override_job_max_runtime_s: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "created_at",
                "created_by",
                "id",
                "owner",
                "status",
                "annotations",
                "completed_at",
                "override_job_max_runtime_s",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["created_at"]
    ) -> MetaOapg.properties.created_at: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["created_by"]
    ) -> MetaOapg.properties.created_by: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["owner"]
    ) -> MetaOapg.properties.owner: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["status"]) -> "BatchJobStatus": ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["annotations"]
    ) -> typing.Union[MetaOapg.properties.annotations, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["completed_at"]
    ) -> typing.Union[MetaOapg.properties.completed_at, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["override_job_max_runtime_s"]
    ) -> typing.Union[MetaOapg.properties.override_job_max_runtime_s, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: str
    ) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "created_at",
                "created_by",
                "id",
                "owner",
                "status",
                "annotations",
                "completed_at",
                "override_job_max_runtime_s",
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
        status: "BatchJobStatus",
        annotations: typing.Union[
            MetaOapg.properties.annotations, dict, frozendict.frozendict, schemas.Unset
        ] = schemas.unset,
        completed_at: typing.Union[
            MetaOapg.properties.completed_at, str, datetime, schemas.Unset
        ] = schemas.unset,
        override_job_max_runtime_s: typing.Union[
            MetaOapg.properties.override_job_max_runtime_s, decimal.Decimal, int, schemas.Unset
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
    ) -> "DockerImageBatchJob":
        return super().__new__(
            cls,
            *_args,
            owner=owner,
            created_at=created_at,
            id=id,
            created_by=created_by,
            status=status,
            annotations=annotations,
            completed_at=completed_at,
            override_job_max_runtime_s=override_job_max_runtime_s,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.batch_job_status import BatchJobStatus
