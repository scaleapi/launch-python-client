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

class CreateModelBundleRequest(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    Request object for creating a Model Bundle.
    """

    class MetaOapg:
        required = {
            "requirements",
            "name",
            "location",
            "env_params",
        }

        class properties:
            @staticmethod
            def env_params() -> typing.Type["ModelBundleEnvironmentParams"]:
                return ModelBundleEnvironmentParams
            location = schemas.StrSchema
            name = schemas.StrSchema

            class requirements(schemas.ListSchema):
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
                ) -> "requirements":
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
                def __getitem__(self, i: int) -> MetaOapg.items:
                    return super().__getitem__(i)
            app_config = schemas.DictSchema
            metadata = schemas.DictSchema

            @staticmethod
            def packaging_type() -> typing.Type["ModelBundlePackagingType"]:
                return ModelBundlePackagingType
            schema_location = schemas.StrSchema
            __annotations__ = {
                "env_params": env_params,
                "location": location,
                "name": name,
                "requirements": requirements,
                "app_config": app_config,
                "metadata": metadata,
                "packaging_type": packaging_type,
                "schema_location": schema_location,
            }
    requirements: MetaOapg.properties.requirements
    name: MetaOapg.properties.name
    location: MetaOapg.properties.location
    env_params: "ModelBundleEnvironmentParams"

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["env_params"]) -> "ModelBundleEnvironmentParams": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["location"]) -> MetaOapg.properties.location: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["requirements"]) -> MetaOapg.properties.requirements: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["app_config"]) -> MetaOapg.properties.app_config: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["metadata"]) -> MetaOapg.properties.metadata: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["packaging_type"]) -> "ModelBundlePackagingType": ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["schema_location"]
    ) -> MetaOapg.properties.schema_location: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "env_params",
                "location",
                "name",
                "requirements",
                "app_config",
                "metadata",
                "packaging_type",
                "schema_location",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["env_params"]) -> "ModelBundleEnvironmentParams": ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["location"]) -> MetaOapg.properties.location: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["requirements"]) -> MetaOapg.properties.requirements: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["app_config"]
    ) -> typing.Union[MetaOapg.properties.app_config, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["metadata"]
    ) -> typing.Union[MetaOapg.properties.metadata, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["packaging_type"]
    ) -> typing.Union["ModelBundlePackagingType", schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["schema_location"]
    ) -> typing.Union[MetaOapg.properties.schema_location, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "env_params",
                "location",
                "name",
                "requirements",
                "app_config",
                "metadata",
                "packaging_type",
                "schema_location",
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
        requirements: typing.Union[
            MetaOapg.properties.requirements,
            list,
            tuple,
        ],
        name: typing.Union[
            MetaOapg.properties.name,
            str,
        ],
        location: typing.Union[
            MetaOapg.properties.location,
            str,
        ],
        env_params: "ModelBundleEnvironmentParams",
        app_config: typing.Union[
            MetaOapg.properties.app_config, dict, frozendict.frozendict, schemas.Unset
        ] = schemas.unset,
        metadata: typing.Union[
            MetaOapg.properties.metadata, dict, frozendict.frozendict, schemas.Unset
        ] = schemas.unset,
        packaging_type: typing.Union["ModelBundlePackagingType", schemas.Unset] = schemas.unset,
        schema_location: typing.Union[MetaOapg.properties.schema_location, str, schemas.Unset] = schemas.unset,
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
    ) -> "CreateModelBundleRequest":
        return super().__new__(
            cls,
            *_args,
            requirements=requirements,
            name=name,
            location=location,
            env_params=env_params,
            app_config=app_config,
            metadata=metadata,
            packaging_type=packaging_type,
            schema_location=schema_location,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.model_bundle_environment_params import (
    ModelBundleEnvironmentParams,
)
from launch_client.model.model_bundle_packaging_type import (
    ModelBundlePackagingType,
)
