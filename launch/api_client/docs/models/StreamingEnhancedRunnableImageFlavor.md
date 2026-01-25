# launch.api_client.model.streaming_enhanced_runnable_image_flavor.StreamingEnhancedRunnableImageFlavor

For deployments that expose a streaming route in a container.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | For deployments that expose a streaming route in a container. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**flavor** | str,  | str,  |  | must be one of ["streaming_enhanced_runnable_image", ] 
**protocol** | str,  | str,  |  | must be one of ["http", ] 
**tag** | str,  | str,  |  | 
**repository** | str,  | str,  |  | 
**[streaming_command](#streaming_command)** | list, tuple,  | tuple,  |  | 
**[command](#command)** | list, tuple,  | tuple,  |  | [optional] 
**predict_route** | str,  | str,  |  | [optional] if omitted the server will use the default value of "/predict"
**healthcheck_route** | str,  | str,  |  | [optional] if omitted the server will use the default value of "/readyz"
**[env](#env)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**readiness_initial_delay_seconds** | decimal.Decimal, int,  | decimal.Decimal,  |  | [optional] if omitted the server will use the default value of 120
**[extra_routes](#extra_routes)** | list, tuple,  | tuple,  |  | [optional] 
**[routes](#routes)** | list, tuple,  | tuple,  |  | [optional] 
**forwarder_type** | None, str,  | NoneClass, str,  |  | [optional] if omitted the server will use the default value of "default"
**[worker_command](#worker_command)** | list, tuple, None,  | tuple, NoneClass,  |  | [optional] 
**[worker_env](#worker_env)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**streaming_predict_route** | str,  | str,  |  | [optional] if omitted the server will use the default value of "/stream"
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# streaming_command

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# command

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# env

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

# extra_routes

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# routes

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# worker_command

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# worker_env

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

