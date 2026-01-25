# launch.api_client.model.create_batch_completions_v1_request_content.CreateBatchCompletionsV1RequestContent

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**max_new_tokens** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**temperature** | decimal.Decimal, int, float,  | decimal.Decimal,  |  | 
**[prompts](#prompts)** | list, tuple,  | tuple,  |  | 
**[stop_sequences](#stop_sequences)** | list, tuple, None,  | tuple, NoneClass,  |  | [optional] 
**return_token_log_probs** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of False
**presence_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  |  | [optional] 
**frequency_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  |  | [optional] 
**top_k** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**top_p** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  |  | [optional] 
**skip_special_tokens** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of True
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# prompts

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# stop_sequences

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

