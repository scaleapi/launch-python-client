# launch.api_client.model.get_model_endpoint_v1_response.GetModelEndpointV1Response

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**endpoint_type** | [**ModelEndpointType**](ModelEndpointType.md) | [**ModelEndpointType**](ModelEndpointType.md) |  | 
**last_updated_at** | str, datetime,  | str,  |  | value must conform to RFC-3339 date-time
**destination** | str,  | str,  |  | 
**name** | str,  | str,  |  | 
**created_at** | str, datetime,  | str,  |  | value must conform to RFC-3339 date-time
**bundle_name** | str,  | str,  |  | 
**id** | str,  | str,  |  | 
**created_by** | str,  | str,  |  | 
**status** | [**ModelEndpointStatus**](ModelEndpointStatus.md) | [**ModelEndpointStatus**](ModelEndpointStatus.md) |  | 
**deployment_name** | None, str,  | NoneClass, str,  |  | [optional] 
**[metadata](#metadata)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**[post_inference_hooks](#post_inference_hooks)** | list, tuple, None,  | tuple, NoneClass,  |  | [optional] 
**default_callback_url** | None, str,  | NoneClass, str,  |  | [optional] 
**default_callback_auth** | [**CallbackAuth**](CallbackAuth.md) | [**CallbackAuth**](CallbackAuth.md) |  | [optional] 
**[labels](#labels)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**aws_role** | None, str,  | NoneClass, str,  |  | [optional] 
**results_s3_bucket** | None, str,  | NoneClass, str,  |  | [optional] 
**deployment_state** | [**ModelEndpointDeploymentState**](ModelEndpointDeploymentState.md) | [**ModelEndpointDeploymentState**](ModelEndpointDeploymentState.md) |  | [optional] 
**resource_state** | [**ModelEndpointResourceState**](ModelEndpointResourceState.md) | [**ModelEndpointResourceState**](ModelEndpointResourceState.md) |  | [optional] 
**num_queued_items** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**public_inference** | None, bool,  | NoneClass, BoolClass,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# metadata

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# post_inference_hooks

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# labels

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

