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


class ModelBundlePackagingType(schemas.EnumBase, schemas.StrSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    The canonical list of possible packaging types for Model Bundles.
    """

    class MetaOapg:
        enum_value_to_name = {
            "cloudpickle": "CLOUDPICKLE",
            "zip": "ZIP",
        }

    @schemas.classproperty
    def CLOUDPICKLE(cls):
        return cls("cloudpickle")

    @schemas.classproperty
    def ZIP(cls):
        return cls("zip")
