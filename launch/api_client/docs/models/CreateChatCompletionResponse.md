# launch.api_client.model.create_chat_completion_response.CreateChatCompletionResponse

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**created** | decimal.Decimal, int,  | decimal.Decimal,  | The Unix timestamp (in seconds) of when the chat completion was created. | 
**model** | str,  | str,  | The model used for the chat completion. | 
**id** | str,  | str,  | A unique identifier for the chat completion. | 
**[choices](#choices)** | list, tuple,  | tuple,  | A list of chat completion choices. Can be more than one if &#x60;n&#x60; is greater than 1. | 
**object** | str,  | str,  | The object type, which is always &#x60;chat.completion&#x60;. | must be one of ["chat.completion", ] 
**service_tier** | [**ServiceTier**](ServiceTier.md) | [**ServiceTier**](ServiceTier.md) |  | [optional] 
**system_fingerprint** | None, str,  | NoneClass, str,  | This fingerprint represents the backend configuration that the model runs with.  Can be used in conjunction with the &#x60;seed&#x60; request parameter to understand when backend changes have been made that might impact determinism.  | [optional] 
**usage** | [**CompletionUsage**](CompletionUsage.md) | [**CompletionUsage**](CompletionUsage.md) |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# choices

A list of chat completion choices. Can be more than one if `n` is greater than 1.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  | A list of chat completion choices. Can be more than one if &#x60;n&#x60; is greater than 1. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**Choice**](Choice.md) | [**Choice**](Choice.md) | [**Choice**](Choice.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

