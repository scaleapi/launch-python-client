# launch.api_client.model.chat_completion_tool_choice_option.ChatCompletionToolChoiceOption

Controls which (if any) tool is called by the model. `none` means the model will not call any tool and instead generates a message. `auto` means the model can pick between generating a message or calling one or more tools. `required` means the model must call one or more tools. Specifying a particular tool via `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}` forces the model to call that tool.  `none` is the default when no tools are present. `auto` is the default if tools are present. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Controls which (if any) tool is called by the model. &#x60;none&#x60; means the model will not call any tool and instead generates a message. &#x60;auto&#x60; means the model can pick between generating a message or calling one or more tools. &#x60;required&#x60; means the model must call one or more tools. Specifying a particular tool via &#x60;{\&quot;type\&quot;: \&quot;function\&quot;, \&quot;function\&quot;: {\&quot;name\&quot;: \&quot;my_function\&quot;}}&#x60; forces the model to call that tool.  &#x60;none&#x60; is the default when no tools are present. &#x60;auto&#x60; is the default if tools are present.  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | must be one of ["none", "auto", "required", ] 
[ChatCompletionNamedToolChoice](ChatCompletionNamedToolChoice.md) | [**ChatCompletionNamedToolChoice**](ChatCompletionNamedToolChoice.md) | [**ChatCompletionNamedToolChoice**](ChatCompletionNamedToolChoice.md) |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | must be one of ["none", "auto", "required", ] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

