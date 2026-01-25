# launch.api_client.model.model_endpoint_deployment_state.ModelEndpointDeploymentState

This is the entity-layer class for the deployment settings related to a Model Endpoint.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | This is the entity-layer class for the deployment settings related to a Model Endpoint. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**max_workers** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**min_workers** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**concurrent_requests_per_worker** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**per_worker** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**available_workers** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**unavailable_workers** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

