# launch.api_client.model.chat_completion_stream_options.ChatCompletionStreamOptions

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**include_usage** | None, bool,  | NoneClass, BoolClass,  | If set, an additional chunk will be streamed before the &#x60;data: [DONE]&#x60; message. The &#x60;usage&#x60; field on this chunk shows the token usage statistics for the entire request, and the &#x60;choices&#x60; field will always be an empty array.   All other chunks will also include a &#x60;usage&#x60; field, but with a null value. **NOTE:** If the stream is interrupted, you may not receive the final usage chunk which contains the total token usage for the request.  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

