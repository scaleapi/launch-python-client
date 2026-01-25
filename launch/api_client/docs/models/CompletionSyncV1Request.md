# launch.api_client.model.completion_sync_v1_request.CompletionSyncV1Request

Request object for a synchronous prompt completion task.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Request object for a synchronous prompt completion task. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**max_new_tokens** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**temperature** | decimal.Decimal, int, float,  | decimal.Decimal,  |  | 
**prompt** | str,  | str,  |  | 
**[stop_sequences](#stop_sequences)** | list, tuple, None,  | tuple, NoneClass,  |  | [optional] 
**return_token_log_probs** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of False
**presence_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  |  | [optional] 
**frequency_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  |  | [optional] 
**top_k** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**top_p** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  |  | [optional] 
**include_stop_str_in_output** | None, bool,  | NoneClass, BoolClass,  |  | [optional] 
**[guided_json](#guided_json)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**guided_regex** | None, str,  | NoneClass, str,  |  | [optional] 
**[guided_choice](#guided_choice)** | list, tuple, None,  | tuple, NoneClass,  |  | [optional] 
**guided_grammar** | None, str,  | NoneClass, str,  |  | [optional] 
**skip_special_tokens** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of True
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# stop_sequences

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# guided_json

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# guided_choice

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

