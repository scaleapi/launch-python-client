# launch.api_client.model.chat_completion_request_assistant_message.ChatCompletionRequestAssistantMessage

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**role** | str,  | str,  | The role of the messages author, in this case &#x60;assistant&#x60;. | must be one of ["assistant", ] 
**[content](#content)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | The contents of the assistant message. Required unless &#x60;tool_calls&#x60; or &#x60;function_call&#x60; is specified.  | [optional] 
**refusal** | None, str,  | NoneClass, str,  | The refusal message by the assistant. | [optional] 
**name** | None, str,  | NoneClass, str,  | An optional name for the participant. Provides the model information to differentiate between participants of the same role. | [optional] 
**audio** | [**Audio**](Audio.md) | [**Audio**](Audio.md) |  | [optional] 
**tool_calls** | [**ChatCompletionMessageToolCallsInput**](ChatCompletionMessageToolCallsInput.md) | [**ChatCompletionMessageToolCallsInput**](ChatCompletionMessageToolCallsInput.md) |  | [optional] 
**function_call** | [**FunctionCall**](FunctionCall.md) | [**FunctionCall**](FunctionCall.md) |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# content

The contents of the assistant message. Required unless `tool_calls` or `function_call` is specified. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | The contents of the assistant message. Required unless &#x60;tool_calls&#x60; or &#x60;function_call&#x60; is specified.  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | 
[Content](Content.md) | [**Content**](Content.md) | [**Content**](Content.md) |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

