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

class CreateLLMModelEndpointV1Request(schemas.DictSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.
    """

    class MetaOapg:
        required = {
            "metadata",
            "memory",
            "model_name",
            "cpus",
            "inference_framework_image_tag",
            "max_workers",
            "gpu_type",
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
                ) -> "cpus":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            @staticmethod
            def gpu_type() -> typing.Type["GpuType"]:
                return GpuType
            gpus = schemas.IntSchema
            inference_framework_image_tag = schemas.StrSchema

            class labels(schemas.DictSchema):
                class MetaOapg:
                    additional_properties = schemas.StrSchema
                def __getitem__(self, name: typing.Union[str,]) -> MetaOapg.additional_properties:
                    # dict_instance[name] accessor
                    return super().__getitem__(name)
                def get_item_oapg(self, name: typing.Union[str,]) -> MetaOapg.additional_properties:
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
            max_workers = schemas.IntSchema

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
                ) -> "memory":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            metadata = schemas.DictSchema
            min_workers = schemas.IntSchema
            model_name = schemas.StrSchema
            name = schemas.StrSchema
            per_worker = schemas.IntSchema
            billing_tags = schemas.DictSchema
            checkpoint_path = schemas.StrSchema

            @staticmethod
            def default_callback_auth() -> typing.Type["CallbackAuth"]:
                return CallbackAuth

            class default_callback_url(schemas.StrSchema):
                pass

            class endpoint_type(
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
                            ModelEndpointType,
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
                ) -> "endpoint_type":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            high_priority = schemas.BoolSchema

            class inference_framework(
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
                            LLMInferenceFramework,
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
                ) -> "inference_framework":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            num_shards = schemas.IntSchema
            optimize_costs = schemas.BoolSchema

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
            prewarm = schemas.BoolSchema
            public_inference = schemas.BoolSchema

            @staticmethod
            def quantize() -> typing.Type["Quantization"]:
                return Quantization

            class source(
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
                            LLMSource,
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
                ) -> "source":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )

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
                ) -> "storage":
                    return super().__new__(
                        cls,
                        *_args,
                        _configuration=_configuration,
                        **kwargs,
                    )
            __annotations__ = {
                "cpus": cpus,
                "gpu_type": gpu_type,
                "gpus": gpus,
                "inference_framework_image_tag": inference_framework_image_tag,
                "labels": labels,
                "max_workers": max_workers,
                "memory": memory,
                "metadata": metadata,
                "min_workers": min_workers,
                "model_name": model_name,
                "name": name,
                "per_worker": per_worker,
                "billing_tags": billing_tags,
                "checkpoint_path": checkpoint_path,
                "default_callback_auth": default_callback_auth,
                "default_callback_url": default_callback_url,
                "endpoint_type": endpoint_type,
                "high_priority": high_priority,
                "inference_framework": inference_framework,
                "num_shards": num_shards,
                "optimize_costs": optimize_costs,
                "post_inference_hooks": post_inference_hooks,
                "prewarm": prewarm,
                "public_inference": public_inference,
                "quantize": quantize,
                "source": source,
                "storage": storage,
            }
    metadata: MetaOapg.properties.metadata
    memory: MetaOapg.properties.memory
    model_name: MetaOapg.properties.model_name
    cpus: MetaOapg.properties.cpus
    inference_framework_image_tag: MetaOapg.properties.inference_framework_image_tag
    max_workers: MetaOapg.properties.max_workers
    gpu_type: "GpuType"
    min_workers: MetaOapg.properties.min_workers
    gpus: MetaOapg.properties.gpus
    name: MetaOapg.properties.name
    per_worker: MetaOapg.properties.per_worker
    labels: MetaOapg.properties.labels

    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["cpus"]) -> MetaOapg.properties.cpus: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["gpu_type"]) -> "GpuType": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["gpus"]) -> MetaOapg.properties.gpus: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["inference_framework_image_tag"]
    ) -> MetaOapg.properties.inference_framework_image_tag: ...
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
    def __getitem__(self, name: typing_extensions.Literal["model_name"]) -> MetaOapg.properties.model_name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["name"]) -> MetaOapg.properties.name: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["per_worker"]) -> MetaOapg.properties.per_worker: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["billing_tags"]) -> MetaOapg.properties.billing_tags: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["checkpoint_path"]
    ) -> MetaOapg.properties.checkpoint_path: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["default_callback_auth"]) -> "CallbackAuth": ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["default_callback_url"]
    ) -> MetaOapg.properties.default_callback_url: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["endpoint_type"]) -> MetaOapg.properties.endpoint_type: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["high_priority"]) -> MetaOapg.properties.high_priority: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["inference_framework"]
    ) -> MetaOapg.properties.inference_framework: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["num_shards"]) -> MetaOapg.properties.num_shards: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["optimize_costs"]) -> MetaOapg.properties.optimize_costs: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["post_inference_hooks"]
    ) -> MetaOapg.properties.post_inference_hooks: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["prewarm"]) -> MetaOapg.properties.prewarm: ...
    @typing.overload
    def __getitem__(
        self, name: typing_extensions.Literal["public_inference"]
    ) -> MetaOapg.properties.public_inference: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["quantize"]) -> "Quantization": ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["source"]) -> MetaOapg.properties.source: ...
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["storage"]) -> MetaOapg.properties.storage: ...
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    def __getitem__(
        self,
        name: typing.Union[
            typing_extensions.Literal[
                "cpus",
                "gpu_type",
                "gpus",
                "inference_framework_image_tag",
                "labels",
                "max_workers",
                "memory",
                "metadata",
                "min_workers",
                "model_name",
                "name",
                "per_worker",
                "billing_tags",
                "checkpoint_path",
                "default_callback_auth",
                "default_callback_url",
                "endpoint_type",
                "high_priority",
                "inference_framework",
                "num_shards",
                "optimize_costs",
                "post_inference_hooks",
                "prewarm",
                "public_inference",
                "quantize",
                "source",
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
    def get_item_oapg(self, name: typing_extensions.Literal["gpu_type"]) -> "GpuType": ...
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["gpus"]) -> MetaOapg.properties.gpus: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["inference_framework_image_tag"]
    ) -> MetaOapg.properties.inference_framework_image_tag: ...
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
    def get_item_oapg(self, name: typing_extensions.Literal["model_name"]) -> MetaOapg.properties.model_name: ...
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
        self, name: typing_extensions.Literal["checkpoint_path"]
    ) -> typing.Union[MetaOapg.properties.checkpoint_path, schemas.Unset]: ...
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
        self, name: typing_extensions.Literal["endpoint_type"]
    ) -> typing.Union[MetaOapg.properties.endpoint_type, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["high_priority"]
    ) -> typing.Union[MetaOapg.properties.high_priority, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["inference_framework"]
    ) -> typing.Union[MetaOapg.properties.inference_framework, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["num_shards"]
    ) -> typing.Union[MetaOapg.properties.num_shards, schemas.Unset]: ...
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
        self, name: typing_extensions.Literal["public_inference"]
    ) -> typing.Union[MetaOapg.properties.public_inference, schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["quantize"]
    ) -> typing.Union["Quantization", schemas.Unset]: ...
    @typing.overload
    def get_item_oapg(
        self, name: typing_extensions.Literal["source"]
    ) -> typing.Union[MetaOapg.properties.source, schemas.Unset]: ...
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
                "gpu_type",
                "gpus",
                "inference_framework_image_tag",
                "labels",
                "max_workers",
                "memory",
                "metadata",
                "min_workers",
                "model_name",
                "name",
                "per_worker",
                "billing_tags",
                "checkpoint_path",
                "default_callback_auth",
                "default_callback_url",
                "endpoint_type",
                "high_priority",
                "inference_framework",
                "num_shards",
                "optimize_costs",
                "post_inference_hooks",
                "prewarm",
                "public_inference",
                "quantize",
                "source",
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
        model_name: typing.Union[
            MetaOapg.properties.model_name,
            str,
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
        inference_framework_image_tag: typing.Union[
            MetaOapg.properties.inference_framework_image_tag,
            str,
        ],
        max_workers: typing.Union[
            MetaOapg.properties.max_workers,
            decimal.Decimal,
            int,
        ],
        gpu_type: "GpuType",
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
            MetaOapg.properties.billing_tags, dict, frozendict.frozendict, schemas.Unset
        ] = schemas.unset,
        checkpoint_path: typing.Union[MetaOapg.properties.checkpoint_path, str, schemas.Unset] = schemas.unset,
        default_callback_auth: typing.Union["CallbackAuth", schemas.Unset] = schemas.unset,
        default_callback_url: typing.Union[
            MetaOapg.properties.default_callback_url, str, schemas.Unset
        ] = schemas.unset,
        endpoint_type: typing.Union[
            MetaOapg.properties.endpoint_type,
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
        high_priority: typing.Union[MetaOapg.properties.high_priority, bool, schemas.Unset] = schemas.unset,
        inference_framework: typing.Union[
            MetaOapg.properties.inference_framework,
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
        num_shards: typing.Union[MetaOapg.properties.num_shards, decimal.Decimal, int, schemas.Unset] = schemas.unset,
        optimize_costs: typing.Union[MetaOapg.properties.optimize_costs, bool, schemas.Unset] = schemas.unset,
        post_inference_hooks: typing.Union[
            MetaOapg.properties.post_inference_hooks, list, tuple, schemas.Unset
        ] = schemas.unset,
        prewarm: typing.Union[MetaOapg.properties.prewarm, bool, schemas.Unset] = schemas.unset,
        public_inference: typing.Union[MetaOapg.properties.public_inference, bool, schemas.Unset] = schemas.unset,
        quantize: typing.Union["Quantization", schemas.Unset] = schemas.unset,
        source: typing.Union[
            MetaOapg.properties.source,
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
    ) -> "CreateLLMModelEndpointV1Request":
        return super().__new__(
            cls,
            *_args,
            metadata=metadata,
            memory=memory,
            model_name=model_name,
            cpus=cpus,
            inference_framework_image_tag=inference_framework_image_tag,
            max_workers=max_workers,
            gpu_type=gpu_type,
            min_workers=min_workers,
            gpus=gpus,
            name=name,
            per_worker=per_worker,
            labels=labels,
            billing_tags=billing_tags,
            checkpoint_path=checkpoint_path,
            default_callback_auth=default_callback_auth,
            default_callback_url=default_callback_url,
            endpoint_type=endpoint_type,
            high_priority=high_priority,
            inference_framework=inference_framework,
            num_shards=num_shards,
            optimize_costs=optimize_costs,
            post_inference_hooks=post_inference_hooks,
            prewarm=prewarm,
            public_inference=public_inference,
            quantize=quantize,
            source=source,
            storage=storage,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.callback_auth import CallbackAuth
from launch_client.model.gpu_type import GpuType
from launch_client.model.llm_inference_framework import LLMInferenceFramework
from launch_client.model.llm_source import LLMSource
from launch_client.model.model_endpoint_type import ModelEndpointType
from launch_client.model.quantization import Quantization
