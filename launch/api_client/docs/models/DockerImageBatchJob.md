# launch.api_client.model.docker_image_batch_job.DockerImageBatchJob

This is the entity-layer class for a Docker Image Batch Job, i.e. a batch job created via the \"supply a docker image for a k8s job\" API.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | This is the entity-layer class for a Docker Image Batch Job, i.e. a batch job created via the \&quot;supply a docker image for a k8s job\&quot; API. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**owner** | str,  | str,  |  | 
**created_at** | str, datetime,  | str,  |  | value must conform to RFC-3339 date-time
**id** | str,  | str,  |  | 
**created_by** | str,  | str,  |  | 
**status** | [**BatchJobStatus**](BatchJobStatus.md) | [**BatchJobStatus**](BatchJobStatus.md) |  | 
**completed_at** | None, str, datetime,  | NoneClass, str,  |  | [optional] value must conform to RFC-3339 date-time
**[annotations](#annotations)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**override_job_max_runtime_s** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**num_workers** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] if omitted the server will use the default value of 1
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# annotations

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

