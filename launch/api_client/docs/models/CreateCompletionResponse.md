# launch.api_client.model.create_completion_response.CreateCompletionResponse

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**created** | decimal.Decimal, int,  | decimal.Decimal,  | The Unix timestamp (in seconds) of when the completion was created. | 
**model** | str,  | str,  | The model used for completion. | 
**id** | str,  | str,  | A unique identifier for the completion. | 
**[choices](#choices)** | list, tuple,  | tuple,  | The list of completion choices the model generated for the input prompt. | 
**object** | str,  | str,  | The object type, which is always \&quot;text_completion\&quot; | must be one of ["text_completion", ] 
**system_fingerprint** | None, str,  | NoneClass, str,  | This fingerprint represents the backend configuration that the model runs with.  Can be used in conjunction with the &#x60;seed&#x60; request parameter to understand when backend changes have been made that might impact determinism.  | [optional] 
**usage** | [**CompletionUsage**](CompletionUsage.md) | [**CompletionUsage**](CompletionUsage.md) |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# choices

The list of completion choices the model generated for the input prompt.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  | The list of completion choices the model generated for the input prompt. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**Choice2**](Choice2.md) | [**Choice2**](Choice2.md) | [**Choice2**](Choice2.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

