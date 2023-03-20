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

class ModelEndpointStatus(schemas.EnumBase, schemas.StrSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    An enumeration.
    """

    @schemas.classproperty
    def READY(cls):
        return cls("READY")
    @schemas.classproperty
    def UPDATE_PENDING(cls):
        return cls("UPDATE_PENDING")
    @schemas.classproperty
    def UPDATE_IN_PROGRESS(cls):
        return cls("UPDATE_IN_PROGRESS")
    @schemas.classproperty
    def UPDATE_FAILED(cls):
        return cls("UPDATE_FAILED")
    @schemas.classproperty
    def DELETE_IN_PROGRESS(cls):
        return cls("DELETE_IN_PROGRESS")
