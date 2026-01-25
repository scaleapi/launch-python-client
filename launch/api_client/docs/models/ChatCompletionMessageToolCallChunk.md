# launch.api_client.model.chat_completion_message_tool_call_chunk.ChatCompletionMessageToolCallChunk

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**index** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**id** | None, str,  | NoneClass, str,  | The ID of the tool call. | [optional] 
**type** | None, str,  | NoneClass, str,  | The type of the tool. Currently, only &#x60;function&#x60; is supported. | [optional] must be one of ["function", ] 
**function** | [**Function2**](Function2.md) | [**Function2**](Function2.md) |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

