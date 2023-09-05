import typing_extensions

from launch.api_client.apis.tags import TagValues
from launch.api_client.apis.tags.default_api import DefaultApi

TagToApi = typing_extensions.TypedDict(
    "TagToApi",
    {
        TagValues.DEFAULT: DefaultApi,
    },
)

tag_to_api = TagToApi(
    {
        TagValues.DEFAULT: DefaultApi,
    }
)
