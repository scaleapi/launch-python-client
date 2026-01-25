# launch.api_client.model.json_schema.JsonSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**name** | str,  | str,  | The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.  | 
**description** | None, str,  | NoneClass, str,  | A description of what the response format is for, used by the model to determine how to respond in the format.  | [optional] 
**schema** | [**ResponseFormatJsonSchemaSchema**](ResponseFormatJsonSchemaSchema.md) | [**ResponseFormatJsonSchemaSchema**](ResponseFormatJsonSchemaSchema.md) |  | [optional] 
**strict** | None, bool,  | NoneClass, BoolClass,  | Whether to enable strict schema adherence when generating the output. If set to true, the model will always follow the exact schema defined in the &#x60;schema&#x60; field. Only a subset of JSON Schema is supported when &#x60;strict&#x60; is &#x60;true&#x60;. To learn more, read the [Structured Outputs guide](/docs/guides/structured-outputs).  | [optional] if omitted the server will use the default value of False
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

