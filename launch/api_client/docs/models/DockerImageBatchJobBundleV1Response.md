# launch.api_client.model.docker_image_batch_job_bundle_v1_response.DockerImageBatchJobBundleV1Response

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**image_repository** | str,  | str,  |  | 
**name** | str,  | str,  |  | 
**created_at** | str, datetime,  | str,  |  | value must conform to RFC-3339 date-time
**id** | str,  | str,  |  | 
**image_tag** | str,  | str,  |  | 
**[env](#env)** | dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 
**[command](#command)** | list, tuple,  | tuple,  |  | 
**mount_location** | None, str,  | NoneClass, str,  |  | [optional] 
**cpus** | None, str,  | NoneClass, str,  |  | [optional] 
**memory** | None, str,  | NoneClass, str,  |  | [optional] 
**storage** | None, str,  | NoneClass, str,  |  | [optional] 
**gpus** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**gpu_type** | None, str,  | NoneClass, str,  |  | [optional] 
**public** | None, bool,  | NoneClass, BoolClass,  |  | [optional] 
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
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

