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


class ListFilesResponse(
    schemas.DictSchema
):
    """NOTE: This class is auto generated by OpenAPI Generator.
    Ref: https://openapi-generator.tech

    Do not edit the class manually.

    Response object for listing files.
    """


    class MetaOapg:
        required = {
            "files",
        }
        
        class properties:
            
            
            class files(
                schemas.ListSchema
            ):
            
            
                class MetaOapg:
                    
                    @staticmethod
                    def items() -> typing.Type['GetFileResponse']:
                        return GetFileResponse
            
                def __new__(
                    cls,
                    _arg: typing.Union[typing.Tuple['GetFileResponse'], typing.List['GetFileResponse']],
                    _configuration: typing.Optional[schemas.Configuration] = None,
                ) -> 'files':
                    return super().__new__(
                        cls,
                        _arg,
                        _configuration=_configuration,
                    )
            
                def __getitem__(self, i: int) -> 'GetFileResponse':
                    return super().__getitem__(i)
            __annotations__ = {
                "files": files,
            }
    
    files: MetaOapg.properties.files
    
    @typing.overload
    def __getitem__(self, name: typing_extensions.Literal["files"]) -> MetaOapg.properties.files: ...
    
    @typing.overload
    def __getitem__(self, name: str) -> schemas.UnsetAnyTypeSchema: ...
    
    def __getitem__(self, name: typing.Union[typing_extensions.Literal["files", ], str]):
        # dict_instance[name] accessor
        return super().__getitem__(name)
    
    
    @typing.overload
    def get_item_oapg(self, name: typing_extensions.Literal["files"]) -> MetaOapg.properties.files: ...
    
    @typing.overload
    def get_item_oapg(self, name: str) -> typing.Union[schemas.UnsetAnyTypeSchema, schemas.Unset]: ...
    
    def get_item_oapg(self, name: typing.Union[typing_extensions.Literal["files", ], str]):
        return super().get_item_oapg(name)
    

    def __new__(
        cls,
        *_args: typing.Union[dict, frozendict.frozendict, ],
        files: typing.Union[MetaOapg.properties.files, list, tuple, ],
        _configuration: typing.Optional[schemas.Configuration] = None,
        **kwargs: typing.Union[schemas.AnyTypeSchema, dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, None, list, tuple, bytes],
    ) -> 'ListFilesResponse':
        return super().__new__(
            cls,
            *_args,
            files=files,
            _configuration=_configuration,
            **kwargs,
        )

from launch_client.model.get_file_response import GetFileResponse
