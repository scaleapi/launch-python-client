# launch.api_client.model.completion_usage.CompletionUsage

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**completion_tokens** | decimal.Decimal, int,  | decimal.Decimal,  | Number of tokens in the generated completion. | 
**prompt_tokens** | decimal.Decimal, int,  | decimal.Decimal,  | Number of tokens in the prompt. | 
**total_tokens** | decimal.Decimal, int,  | decimal.Decimal,  | Total number of tokens used in the request (prompt + completion). | 
**completion_tokens_details** | [**CompletionTokensDetails**](CompletionTokensDetails.md) | [**CompletionTokensDetails**](CompletionTokensDetails.md) |  | [optional] 
**prompt_tokens_details** | [**PromptTokensDetails**](PromptTokensDetails.md) | [**PromptTokensDetails**](PromptTokensDetails.md) |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

