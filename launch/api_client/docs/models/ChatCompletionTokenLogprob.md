# launch.api_client.model.chat_completion_token_logprob.ChatCompletionTokenLogprob

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**[top_logprobs](#top_logprobs)** | list, tuple,  | tuple,  | List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested &#x60;top_logprobs&#x60; returned. | 
**logprob** | decimal.Decimal, int, float,  | decimal.Decimal,  | The log probability of this token, if it is within the top 20 most likely tokens. Otherwise, the value &#x60;-9999.0&#x60; is used to signify that the token is very unlikely. | 
**[bytes](#bytes)** | list, tuple, None,  | tuple, NoneClass,  | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be &#x60;null&#x60; if there is no bytes representation for the token. | 
**token** | str,  | str,  | The token. | 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# bytes

A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be `null` if there is no bytes representation for the token.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | A list of integers representing the UTF-8 bytes representation of the token. Useful in instances where characters are represented by multiple tokens and their byte representations must be combined to generate the correct text representation. Can be &#x60;null&#x60; if there is no bytes representation for the token. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | decimal.Decimal, int,  | decimal.Decimal,  |  | 

# top_logprobs

List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested `top_logprobs` returned.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  | List of the most likely tokens and their log probability, at this token position. In rare cases, there may be fewer than the number of requested &#x60;top_logprobs&#x60; returned. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**TopLogprob**](TopLogprob.md) | [**TopLogprob**](TopLogprob.md) | [**TopLogprob**](TopLogprob.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

