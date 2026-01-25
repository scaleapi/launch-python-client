# launch.api_client.model.create_batch_job_v1_request.CreateBatchJobV1Request

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**model_bundle_id** | str,  | str,  |  | 
**resource_requests** | [**CreateBatchJobResourceRequests**](CreateBatchJobResourceRequests.md) | [**CreateBatchJobResourceRequests**](CreateBatchJobResourceRequests.md) |  | 
**serialization_format** | [**BatchJobSerializationFormat**](BatchJobSerializationFormat.md) | [**BatchJobSerializationFormat**](BatchJobSerializationFormat.md) |  | 
**input_path** | str,  | str,  |  | 
**[labels](#labels)** | dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 
**timeout_seconds** | decimal.Decimal, int, float,  | decimal.Decimal,  |  | [optional] if omitted the server will use the default value of 43200.0
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# labels

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

