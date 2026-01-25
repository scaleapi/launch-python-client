# launch.api_client.model.completion_stream_output.CompletionStreamOutput

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**finished** | bool,  | BoolClass,  |  | 
**text** | str,  | str,  |  | 
**num_prompt_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**num_completion_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**token** | [**TokenOutput**](TokenOutput.md) | [**TokenOutput**](TokenOutput.md) |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

