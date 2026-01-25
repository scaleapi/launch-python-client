# launch.api_client.model.sync_endpoint_predict_v1_request.SyncEndpointPredictV1Request

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**url** | None, str,  | NoneClass, str,  |  | [optional] 
**args** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] 
**cloudpickle** | None, str,  | NoneClass, str,  |  | [optional] 
**callback_url** | None, str,  | NoneClass, str,  |  | [optional] 
**callback_auth** | [**CallbackAuth**](CallbackAuth.md) | [**CallbackAuth**](CallbackAuth.md) |  | [optional] 
**return_pickled** | bool,  | BoolClass,  |  | [optional] if omitted the server will use the default value of False
**destination_path** | None, str,  | NoneClass, str,  |  | [optional] 
**timeout_seconds** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  |  | [optional] 
**num_retries** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

