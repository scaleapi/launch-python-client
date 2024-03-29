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

class TaskStatus(schemas.EnumBase, schemas.StrSchema):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    An enumeration.
    """

    @schemas.classproperty
    def PENDING(cls):
        return cls("PENDING")
    @schemas.classproperty
    def STARTED(cls):
        return cls("STARTED")
    @schemas.classproperty
    def SUCCESS(cls):
        return cls("SUCCESS")
    @schemas.classproperty
    def FAILURE(cls):
        return cls("FAILURE")
    @schemas.classproperty
    def UNDEFINED(cls):
        return cls("UNDEFINED")
