# launch.api_client.model.get_fine_tune_response.GetFineTuneResponse

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**id** | str,  | str,  | Unique ID of the fine tune | 
**status** | [**BatchJobStatus**](BatchJobStatus.md) | [**BatchJobStatus**](BatchJobStatus.md) |  | 
**fine_tuned_model** | None, str,  | NoneClass, str,  | Name of the resulting fine-tuned model. This can be plugged into the Completion API ones the fine-tune is complete | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

