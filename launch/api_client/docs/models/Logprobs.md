# launch.api_client.model.logprobs.Logprobs

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**[refusal](#refusal)** | list, tuple, None,  | tuple, NoneClass,  | A list of message refusal tokens with log probability information. | 
**[content](#content)** | list, tuple, None,  | tuple, NoneClass,  | A list of message content tokens with log probability information. | 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# content

A list of message content tokens with log probability information.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | A list of message content tokens with log probability information. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ChatCompletionTokenLogprob**](ChatCompletionTokenLogprob.md) | [**ChatCompletionTokenLogprob**](ChatCompletionTokenLogprob.md) | [**ChatCompletionTokenLogprob**](ChatCompletionTokenLogprob.md) |  | 

# refusal

A list of message refusal tokens with log probability information.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | A list of message refusal tokens with log probability information. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ChatCompletionTokenLogprob**](ChatCompletionTokenLogprob.md) | [**ChatCompletionTokenLogprob**](ChatCompletionTokenLogprob.md) | [**ChatCompletionTokenLogprob**](ChatCompletionTokenLogprob.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

