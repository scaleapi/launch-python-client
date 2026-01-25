# launch.api_client.model.create_docker_image_batch_job_bundle_v1_request.CreateDockerImageBatchJobBundleV1Request

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**image_repository** | str,  | str,  |  | 
**name** | str,  | str,  |  | 
**image_tag** | str,  | str,  |  | 
**[command](#command)** | list, tuple,  | tuple,  |  | 
**[env](#env)** | dict, frozendict.frozendict,  | frozendict.frozendict,  |  | [optional] if omitted the server will use the default value of {}
**mount_location** | None, str,  | NoneClass, str,  |  | [optional] 
**resource_requests** | [**CreateDockerImageBatchJobResourceRequests**](CreateDockerImageBatchJobResourceRequests.md) | [**CreateDockerImageBatchJobResourceRequests**](CreateDockerImageBatchJobResourceRequests.md) |  | [optional] 
**public** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of False
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

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
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | if omitted the server will use the default value of {}

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

