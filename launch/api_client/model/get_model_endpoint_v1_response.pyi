# coding: utf-8

"""
    launch

    No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Generated by: https://openapi-generator.tech
"""

from datetime import date, datetime  # noqa: F401
import decimal  # noqa: F401
import functools  # noqa: F401
import io  # noqa: F401
import re  # noqa: F401
import typing  # noqa: F401
import typing_extensions  # noqa: F401
import uuid  # noqa: F401

import frozendict  # noqa: F401

from launch_client import schemas  # noqa: F401

class GetModelEndpointV1Response(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "endpoint_type",
            "last_updated_at",
            "destination",
            "name",
            "created_at",
            "bundle_name",
            "id",
            "created_by",
            "status",
        }

        class properties:
            bundle_name = schemas.StrSchema
            created_at = schemas.DateTimeSchema
            created_by = schemas.StrSchema
            destination = schemas.StrSchema

            @staticmethod
            def endpoint_type() -> typing.Type["ModelEndpointType"]:
                return ModelEndpointType
            id = schemas.StrSchema
            last_updated_at = schemas.DateTimeSchema
            name = schemas.StrSchema

            @staticmethod
            def status() -> typing.Type["ModelEndpointStatus"]:
                return ModelEndpointStatus
            aws_role = schemas.StrSchema

            @staticmethod
            def default_callback_auth() -> typing.Type["CallbackAuth"]:
                return CallbackAuth

            class default_callback_url(schemas.StrSchema):
                pass
            deployment_name = schemas.StrSchema

            @staticmethod
            def deployment_state() -> typing.Type["ModelEndpointDeploymentState"]:
                return ModelEndpointDeploymentState

            class labels(schemas.DictSchema):
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
                ) -> "labels":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            metadata = schemas.DictSchema
            num_queued_items = schemas.IntSchema

            class post_inference_hooks(schemas.ListSchema):
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
                ) -> "post_inference_hooks":
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
                def __getitem__(self, i: int) -> MetaOapg.items:
                    return super().__getitem__(i)
            @staticmethod
            def resource_state() -> typing.Type["ModelEndpointResourceState"]:
                return ModelEndpointResourceState
            results_s3_bucket = schemas.StrSchema
            __annotations__ = {
                "bundle_name": bundle_name,
                "created_at": created_at,
                "created_by": created_by,
                "destination": destination,
                "endpoint_type": endpoint_type,
                "id": id,
                "last_updated_at": last_updated_at,
                "name": name,
                "status": status,
                "aws_role": aws_role,
                "default_callback_auth": default_callback_auth,
                "default_callback_url": default_callback_url,
                "deployment_name": deployment_name,
                "deployment_state": deployment_state,
                "labels": labels,
                "metadata": metadata,
                "num_queued_items": num_queued_items,
                "post_inference_hooks": post_inference_hooks,
                "resource_state": resource_state,
                "results_s3_bucket": results_s3_bucket,
            }
    endpoint_type: "ModelEndpointType"
    last_updated_at: MetaOapg.properties.last_updated_at
    destination: MetaOapg.properties.destination
    name: MetaOapg.properties.name
    created_at: MetaOapg.properties.created_at
    bundle_name: MetaOapg.properties.bundle_name
    id: MetaOapg.properties.id
    created_by: MetaOapg.properties.created_by
    status: "ModelEndpointStatus"

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["bundle_name"]) -> MetaOapg.properties.bundle_name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["created_at"]) -> MetaOapg.properties.created_at: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["created_by"]) -> MetaOapg.properties.created_by: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["destination"]) -> MetaOapg.properties.destination: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["endpoint_type"]) -> "ModelEndpointType": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["last_updated_at"]
    ) -> MetaOapg.properties.last_updated_at: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["status"]) -> "ModelEndpointStatus": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["aws_role"]) -> MetaOapg.properties.aws_role: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["default_callback_auth"]) -> "CallbackAuth": ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["default_callback_url"]
    ) -> MetaOapg.properties.default_callback_url: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["deployment_name"]
    ) -> MetaOapg.properties.deployment_name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["deployment_state"]) -> "ModelEndpointDeploymentState": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["labels"]) -> MetaOapg.properties.labels: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["metadata"]) -> MetaOapg.properties.metadata: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["num_queued_items"]
    ) -> MetaOapg.properties.num_queued_items: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["post_inference_hooks"]
    ) -> MetaOapg.properties.post_inference_hooks: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["resource_state"]) -> "ModelEndpointResourceState": ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["results_s3_bucket"]
    ) -> MetaOapg.properties.results_s3_bucket: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "bundle_name",
                "created_at",
                "created_by",
                "destination",
                "endpoint_type",
                "id",
                "last_updated_at",
                "name",
                "status",
                "aws_role",
                "default_callback_auth",
                "default_callback_url",
                "deployment_name",
                "deployment_state",
                "labels",
                "metadata",
                "num_queued_items",
                "post_inference_hooks",
                "resource_state",
                "results_s3_bucket",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["bundle_name"]) -> MetaOapg.properties.bundle_name: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["created_at"]) -> MetaOapg.properties.created_at: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["created_by"]) -> MetaOapg.properties.created_by: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["destination"]) -> MetaOapg.properties.destination: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["endpoint_type"]) -> "ModelEndpointType": ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["id"]) -> MetaOapg.properties.id: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["last_updated_at"]
    ) -> MetaOapg.properties.last_updated_at: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["status"]) -> "ModelEndpointStatus": ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["aws_role"]
    ) -> typing.Union[MetaOapg.properties.aws_role, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["default_callback_auth"]
    ) -> typing.Union["CallbackAuth", schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["default_callback_url"]
    ) -> typing.Union[MetaOapg.properties.default_callback_url, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["deployment_name"]
    ) -> typing.Union[MetaOapg.properties.deployment_name, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["deployment_state"]
    ) -> typing.Union["ModelEndpointDeploymentState", schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["labels"]
    ) -> typing.Union[MetaOapg.properties.labels, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["metadata"]
    ) -> typing.Union[MetaOapg.properties.metadata, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["num_queued_items"]
    ) -> typing.Union[MetaOapg.properties.num_queued_items, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["post_inference_hooks"]
    ) -> typing.Union[MetaOapg.properties.post_inference_hooks, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["resource_state"]
    ) -> typing.Union["ModelEndpointResourceState", schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["results_s3_bucket"]
    ) -> typing.Union[MetaOapg.properties.results_s3_bucket, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "bundle_name",
                "created_at",
                "created_by",
                "destination",
                "endpoint_type",
                "id",
                "last_updated_at",
                "name",
                "status",
                "aws_role",
                "default_callback_auth",
                "default_callback_url",
                "deployment_name",
                "deployment_state",
                "labels",
                "metadata",
                "num_queued_items",
                "post_inference_hooks",
                "resource_state",
                "results_s3_bucket",
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
        endpoint_type: "ModelEndpointType",
        last_updated_at: typing.Union[
            MetaOapg.properties.last_updated_at,
            str,
            datetime,
        ],
        destination: typing.Union[
            MetaOapg.properties.destination,
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
        bundle_name: typing.Union[
            MetaOapg.properties.bundle_name,
            str,
        ],
        id: typing.Union[
            MetaOapg.properties.id,
            str,
        ],
        created_by: typing.Union[
            MetaOapg.properties.created_by,
            str,
        ],
        status: "ModelEndpointStatus",
        aws_role: typing.Union[MetaOapg.properties.aws_role, str, schemas.Unset] = schemas.unset,
        default_callback_auth: typing.Union["CallbackAuth", schemas.Unset] = schemas.unset,
        default_callback_url: typing.Union[
            MetaOapg.properties.default_callback_url, str, schemas.Unset
        ] = schemas.unset,
        deployment_name: typing.Union[MetaOapg.properties.deployment_name, str, schemas.Unset] = schemas.unset,
        deployment_state: typing.Union["ModelEndpointDeploymentState", schemas.Unset] = schemas.unset,
        labels: typing.Union[MetaOapg.properties.labels, dict, frozendict.frozendict, schemas.Unset] = schemas.unset,
        metadata: typing.Union[
            MetaOapg.properties.metadata, dict, frozendict.frozendict, schemas.Unset
        ] = schemas.unset,
        num_queued_items: typing.Union[
            MetaOapg.properties.num_queued_items, decimal.Decimal, int, schemas.Unset
        ] = schemas.unset,
        post_inference_hooks: typing.Union[
            MetaOapg.properties.post_inference_hooks, list, tuple, schemas.Unset
        ] = schemas.unset,
        resource_state: typing.Union["ModelEndpointResourceState", schemas.Unset] = schemas.unset,
        results_s3_bucket: typing.Union[MetaOapg.properties.results_s3_bucket, str, schemas.Unset] = schemas.unset,
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
    ) -> "GetModelEndpointV1Response":
        return super().__new__(
            cls,
            *_args,
            endpoint_type=endpoint_type,
            last_updated_at=last_updated_at,
            destination=destination,
            name=name,
            created_at=created_at,
            bundle_name=bundle_name,
            id=id,
            created_by=created_by,
            status=status,
            aws_role=aws_role,
            default_callback_auth=default_callback_auth,
            default_callback_url=default_callback_url,
            deployment_name=deployment_name,
            deployment_state=deployment_state,
            labels=labels,
            metadata=metadata,
            num_queued_items=num_queued_items,
            post_inference_hooks=post_inference_hooks,
            resource_state=resource_state,
            results_s3_bucket=results_s3_bucket,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.callback_auth import CallbackAuth
from launch_client.model.model_endpoint_deployment_state import ModelEndpointDeploymentState
from launch_client.model.model_endpoint_resource_state import ModelEndpointResourceState
from launch_client.model.model_endpoint_status import ModelEndpointStatus
from launch_client.model.model_endpoint_type import ModelEndpointType
