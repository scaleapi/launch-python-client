# launch.api_client.model.completion_output.CompletionOutput

Represents the output of a completion request to a model.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Represents the output of a completion request to a model. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**num_completion_tokens** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**text** | str,  | str,  |  | 
**num_prompt_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**[tokens](#tokens)** | list, tuple, None,  | tuple, NoneClass,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# tokens

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**TokenOutput**](TokenOutput.md) | [**TokenOutput**](TokenOutput.md) | [**TokenOutput**](TokenOutput.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

