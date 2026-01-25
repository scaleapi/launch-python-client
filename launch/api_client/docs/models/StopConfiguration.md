# launch.api_client.model.stop_configuration.StopConfiguration

Not supported with latest reasoning models `o3` and `o4-mini`.  Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Not supported with latest reasoning models &#x60;o3&#x60; and &#x60;o4-mini&#x60;.  Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | 
[StopConfiguration1](StopConfiguration1.md) | [**StopConfiguration1**](StopConfiguration1.md) | [**StopConfiguration1**](StopConfiguration1.md) |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

