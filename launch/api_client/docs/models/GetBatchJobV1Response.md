# launch.api_client.model.get_batch_job_v1_response.GetBatchJobV1Response

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**duration** | str,  | str,  |  | 
**status** | [**BatchJobStatus**](BatchJobStatus.md) | [**BatchJobStatus**](BatchJobStatus.md) |  | 
**result** | None, str,  | NoneClass, str,  |  | [optional] 
**num_tasks_pending** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**num_tasks_completed** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

