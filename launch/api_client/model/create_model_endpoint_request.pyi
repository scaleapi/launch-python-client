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

class CreateModelEndpointRequest(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "endpoint_type",
            "metadata",
            "memory",
            "cpus",
            "max_workers",
            "model_bundle_id",
            "min_workers",
            "gpus",
            "name",
            "per_worker",
            "labels",
        }

        class properties:
            class cpus(
                schemas.ComposedSchema,
            ):
                class MetaOapg:
                    any_of_0 = schemas.StrSchema
                    any_of_1 = schemas.IntSchema
                    any_of_2 = schemas.NumberSchema

                    @classmethod
                    @functools.lru_cache()
                    def any_of(cls):
                        # we need this here to make our import statements work
                        # we must store _composed_schemas in here so the code is only run
                        # when we invoke this method. If we kept this at the class
                        # level we would get an error because the class level
                        # code would be run when this module is imported, and these composed
                        # classes don't exist yet because their module has not finished
                        # loading
                        return [
                            cls.any_of_0,
                            cls.any_of_1,
                            cls.any_of_2,
                        ]
                def __new__(
                    cls,
                    *args: typing.Union[
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
                ) -> "cpus":
                    return super().__new__(
                        cls,
                        *args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            @staticmethod
            def endpoint_type() -> typing.Type["ModelEndpointType"]:
                return ModelEndpointType

            class gpus(schemas.IntSchema):
                pass

            class labels(schemas.DictSchema):
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
                    *args: typing.Union[
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
                        *args,
                        _configuration=_configuration,
                        **kwargs,
                    )

            class max_workers(schemas.IntSchema):
                pass

            class memory(
                schemas.ComposedSchema,
            ):
                class MetaOapg:
                    any_of_0 = schemas.StrSchema
                    any_of_1 = schemas.IntSchema
                    any_of_2 = schemas.NumberSchema

                    @classmethod
                    @functools.lru_cache()
                    def any_of(cls):
                        # we need this here to make our import statements work
                        # we must store _composed_schemas in here so the code is only run
                        # when we invoke this method. If we kept this at the class
                        # level we would get an error because the class level
                        # code would be run when this module is imported, and these composed
                        # classes don't exist yet because their module has not finished
                        # loading
                        return [
                            cls.any_of_0,
                            cls.any_of_1,
                            cls.any_of_2,
                        ]
                def __new__(
                    cls,
                    *args: typing.Union[
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
                ) -> "memory":
                    return super().__new__(
                        cls,
                        *args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            metadata = schemas.DictSchema

            class min_workers(schemas.IntSchema):
                pass
            model_bundle_id = schemas.StrSchema
            name = schemas.StrSchema
            per_worker = schemas.IntSchema
            billing_tags = schemas.DictSchema

            class default_callback_url(schemas.StrSchema):
                pass
            @staticmethod
            def gpu_type() -> typing.Type["GpuType"]:
                return GpuType
            optimize_costs = schemas.BoolSchema

            class post_inference_hooks(schemas.ListSchema):
                class MetaOapg:
                    items = schemas.StrSchema
                def __new__(
                    cls,
                    arg: typing.Union[
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
                        arg,
                        _configuration=_configuration,
                    )
                def __getitem__(self, i: int) -> MetaOapg.items:
                    return super().__getitem__(i)
            prewarm = schemas.BoolSchema

            class storage(
                schemas.ComposedSchema,
            ):
                class MetaOapg:
                    any_of_0 = schemas.StrSchema
                    any_of_1 = schemas.IntSchema
                    any_of_2 = schemas.NumberSchema

                    @classmethod
                    @functools.lru_cache()
                    def any_of(cls):
                        # we need this here to make our import statements work
                        # we must store _composed_schemas in here so the code is only run
                        # when we invoke this method. If we kept this at the class
                        # level we would get an error because the class level
                        # code would be run when this module is imported, and these composed
                        # classes don't exist yet because their module has not finished
                        # loading
                        return [
                            cls.any_of_0,
                            cls.any_of_1,
                            cls.any_of_2,
                        ]
                def __new__(
                    cls,
                    *args: typing.Union[
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
                ) -> "storage":
                    return super().__new__(
                        cls,
                        *args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            __annotations__ = {
                "cpus": cpus,
                "endpoint_type": endpoint_type,
                "gpus": gpus,
                "labels": labels,
                "max_workers": max_workers,
                "memory": memory,
                "metadata": metadata,
                "min_workers": min_workers,
                "model_bundle_id": model_bundle_id,
                "name": name,
                "per_worker": per_worker,
                "billing_tags": billing_tags,
                "default_callback_url": default_callback_url,
                "gpu_type": gpu_type,
                "optimize_costs": optimize_costs,
                "post_inference_hooks": post_inference_hooks,
                "prewarm": prewarm,
                "storage": storage,
            }
    endpoint_type: "ModelEndpointType"
    metadata: MetaOapg.properties.metadata
    memory: MetaOapg.properties.memory
    cpus: MetaOapg.properties.cpus
    max_workers: MetaOapg.properties.max_workers
    model_bundle_id: MetaOapg.properties.model_bundle_id
    min_workers: MetaOapg.properties.min_workers
    gpus: MetaOapg.properties.gpus
    name: MetaOapg.properties.name
    per_worker: MetaOapg.properties.per_worker
    labels: MetaOapg.properties.labels

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["cpus"]) -> MetaOapg.properties.cpus: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["endpoint_type"]) -> "ModelEndpointType": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["gpus"]) -> MetaOapg.properties.gpus: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["labels"]) -> MetaOapg.properties.labels: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["max_workers"]) -> MetaOapg.properties.max_workers: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["memory"]) -> MetaOapg.properties.memory: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["metadata"]) -> MetaOapg.properties.metadata: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["min_workers"]) -> MetaOapg.properties.min_workers: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["model_bundle_id"]
    ) -> MetaOapg.properties.model_bundle_id: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["per_worker"]) -> MetaOapg.properties.per_worker: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["billing_tags"]) -> MetaOapg.properties.billing_tags: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["default_callback_url"]
    ) -> MetaOapg.properties.default_callback_url: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["gpu_type"]) -> "GpuType": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["optimize_costs"]) -> MetaOapg.properties.optimize_costs: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["post_inference_hooks"]
    ) -> MetaOapg.properties.post_inference_hooks: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["prewarm"]) -> MetaOapg.properties.prewarm: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["storage"]) -> MetaOapg.properties.storage: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "cpus",
                "endpoint_type",
                "gpus",
                "labels",
                "max_workers",
                "memory",
                "metadata",
                "min_workers",
                "model_bundle_id",
                "name",
                "per_worker",
                "billing_tags",
                "default_callback_url",
                "gpu_type",
                "optimize_costs",
                "post_inference_hooks",
                "prewarm",
                "storage",
            ],
            str,
        ],
    ):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["cpus"]) -> MetaOapg.properties.cpus: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["endpoint_type"]) -> "ModelEndpointType": ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["gpus"]) -> MetaOapg.properties.gpus: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["labels"]) -> MetaOapg.properties.labels: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["max_workers"]) -> MetaOapg.properties.max_workers: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["memory"]) -> MetaOapg.properties.memory: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["metadata"]) -> MetaOapg.properties.metadata: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["min_workers"]) -> MetaOapg.properties.min_workers: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["model_bundle_id"]
    ) -> MetaOapg.properties.model_bundle_id: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["per_worker"]) -> MetaOapg.properties.per_worker: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["billing_tags"]
    ) -> typing.Union[MetaOapg.properties.billing_tags, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["default_callback_url"]
    ) -> typing.Union[MetaOapg.properties.default_callback_url, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["gpu_type"]) -> typing.Union["GpuType", schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["optimize_costs"]
    ) -> typing.Union[MetaOapg.properties.optimize_costs, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["post_inference_hooks"]
    ) -> typing.Union[MetaOapg.properties.post_inference_hooks, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["prewarm"]
    ) -> typing.Union[MetaOapg.properties.prewarm, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["storage"]
    ) -> typing.Union[MetaOapg.properties.storage, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    def get_item_oapg(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "cpus",
                "endpoint_type",
                "gpus",
                "labels",
                "max_workers",
                "memory",
                "metadata",
                "min_workers",
                "model_bundle_id",
                "name",
                "per_worker",
                "billing_tags",
                "default_callback_url",
                "gpu_type",
                "optimize_costs",
                "post_inference_hooks",
                "prewarm",
                "storage",
            ],
            str,
        ],
    ):
        return super().get_item_oapg(name)
    def __new__(
        cls,
        *args: typing.Union[
            dict,
            frozendict.frozendict,
        ],
        endpoint_type: "ModelEndpointType",
        metadata: typing.Union[
            MetaOapg.properties.metadata,
            dict,
            frozendict.frozendict,
        ],
        memory: typing.Union[
            MetaOapg.properties.memory,
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
        cpus: typing.Union[
            MetaOapg.properties.cpus,
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
        max_workers: typing.Union[
            MetaOapg.properties.max_workers,
            decimal.Decimal,
            int,
        ],
        model_bundle_id: typing.Union[
            MetaOapg.properties.model_bundle_id,
            str,
        ],
        min_workers: typing.Union[
            MetaOapg.properties.min_workers,
            decimal.Decimal,
            int,
        ],
        gpus: typing.Union[
            MetaOapg.properties.gpus,
            decimal.Decimal,
            int,
        ],
        name: typing.Union[
            MetaOapg.properties.name,
            str,
        ],
        per_worker: typing.Union[
            MetaOapg.properties.per_worker,
            decimal.Decimal,
            int,
        ],
        labels: typing.Union[
            MetaOapg.properties.labels,
            dict,
            frozendict.frozendict,
        ],
        billing_tags: typing.Union[
            MetaOapg.properties.billing_tags,
            dict,
            frozendict.frozendict,
            schemas.Unset,
        ] = schemas.unset,
        default_callback_url: typing.Union[
            MetaOapg.properties.default_callback_url, str, schemas.Unset
        ] = schemas.unset,
        gpu_type: typing.Union["GpuType", schemas.Unset] = schemas.unset,
        optimize_costs: typing.Union[MetaOapg.properties.optimize_costs, bool, schemas.Unset] = schemas.unset,
        post_inference_hooks: typing.Union[
            MetaOapg.properties.post_inference_hooks,
            list,
            tuple,
            schemas.Unset,
        ] = schemas.unset,
        prewarm: typing.Union[MetaOapg.properties.prewarm, bool, schemas.Unset] = schemas.unset,
        storage: typing.Union[
            MetaOapg.properties.storage,
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
    ) -> "CreateModelEndpointRequest":
        return super().__new__(
            cls,
            *args,
            endpoint_type=endpoint_type,
            metadata=metadata,
            memory=memory,
            cpus=cpus,
            max_workers=max_workers,
            model_bundle_id=model_bundle_id,
            min_workers=min_workers,
            gpus=gpus,
            name=name,
            per_worker=per_worker,
            labels=labels,
            billing_tags=billing_tags,
            default_callback_url=default_callback_url,
            gpu_type=gpu_type,
            optimize_costs=optimize_costs,
            post_inference_hooks=post_inference_hooks,
            prewarm=prewarm,
            storage=storage,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.gpu_type import GpuType
from launch_client.model.model_endpoint_type import ModelEndpointType
