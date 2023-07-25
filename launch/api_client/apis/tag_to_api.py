import typing_extensions
from launch_client.apis.tags import TagValues
from launch_client.apis.tags.default_api import DefaultApi

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
