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

class ListDockerImageBatchJobBundleV1Response(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "docker_image_batch_job_bundles",
        }

        class properties:
            class docker_image_batch_job_bundles(schemas.ListSchema):
                class MetaOapg:
                    @staticmethod
                    def items() -> typing.Type["DockerImageBatchJobBundleV1Response"]:
                        return DockerImageBatchJobBundleV1Response
                def __new__(
                    cls,
                    _arg: typing.Union[
                        typing.Tuple["DockerImageBatchJobBundleV1Response"],
                        typing.List["DockerImageBatchJobBundleV1Response"],
                    ],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> "docker_image_batch_job_bundles":
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
                def __getitem__(self, i: int) -> "DockerImageBatchJobBundleV1Response":
                    return super().__getitem__(i)
            __annotations__ = {
                "docker_image_batch_job_bundles": docker_image_batch_job_bundles,
            }
    docker_image_batch_job_bundles: MetaOapg.properties.docker_image_batch_job_bundles

    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["docker_image_batch_job_bundles"]
    ) -> MetaOapg.properties.docker_image_batch_job_bundles: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "docker_image_batch_job_bundles",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["docker_image_batch_job_bundles"]
    ) -> MetaOapg.properties.docker_image_batch_job_bundles: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "docker_image_batch_job_bundles",
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
        docker_image_batch_job_bundles: typing.Union[
            MetaOapg.properties.docker_image_batch_job_bundles,
            list,
            tuple,
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
    ) -> "ListDockerImageBatchJobBundleV1Response":
        return super().__new__(
            cls,
            *_args,
            docker_image_batch_job_bundles=docker_image_batch_job_bundles,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.docker_image_batch_job_bundle_v1_response import (
    DockerImageBatchJobBundleV1Response,
)
