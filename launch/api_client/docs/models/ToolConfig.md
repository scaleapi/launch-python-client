# launch.api_client.model.tool_config.ToolConfig

Configuration for tool use. NOTE: this config is highly experimental and signature will change significantly in future iterations.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Configuration for tool use. NOTE: this config is highly experimental and signature will change significantly in future iterations. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**name** | str,  | str,  |  | 
**max_iterations** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] if omitted the server will use the default value of 10
**execution_timeout_seconds** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] if omitted the server will use the default value of 60
**should_retry_on_error** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of True
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

