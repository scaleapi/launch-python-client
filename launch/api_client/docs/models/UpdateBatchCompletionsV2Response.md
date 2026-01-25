# launch.api_client.model.update_batch_completions_v2_response.UpdateBatchCompletionsV2Response

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**completed_at** | None, str,  | NoneClass, str,  |  | 
**[metadata](#metadata)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 
**expires_at** | str,  | str,  |  | 
**model_config** | [**BatchCompletionsModelConfig**](BatchCompletionsModelConfig.md) | [**BatchCompletionsModelConfig**](BatchCompletionsModelConfig.md) |  | 
**job_id** | str,  | str,  |  | 
**success** | bool,  | BoolClass,  | Whether the update was successful | 
**created_at** | str,  | str,  |  | 
**output_data_path** | str,  | str,  | Path to the output file. The output file will be a JSON file of type List[CompletionOutput]. | 
**status** | [**BatchCompletionsJobStatus**](BatchCompletionsJobStatus.md) | [**BatchCompletionsJobStatus**](BatchCompletionsJobStatus.md) |  | 
**input_data_path** | None, str,  | NoneClass, str,  | Path to the input file. The input file should be a JSON file of type List[CreateBatchCompletionsRequestContent]. | [optional] 
**priority** | None, str,  | NoneClass, str,  | Priority of the batch inference job. Default to None. | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# metadata

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

