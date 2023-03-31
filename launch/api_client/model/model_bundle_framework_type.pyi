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

class ModelBundleFrameworkType(schemas.EnumBase, schemas.StrSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    The canonical list of possible machine learning frameworks of Model Bundles.
    """

    @schemas.classproperty
    def PYTORCH(cls):
        return cls("pytorch")
    @schemas.classproperty
    def TENSORFLOW(cls):
        return cls("tensorflow")
    @schemas.classproperty
    def CUSTOM_BASE_IMAGE(cls):
        return cls("custom_base_image")
