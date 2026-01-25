# launch.api_client.model.filtered_chat_completion_v2_request.FilteredChatCompletionV2Request

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**[messages](#messages)** | list, tuple,  | tuple,  | A list of messages comprising the conversation so far. Depending on the [model](/docs/models) you use, different message types (modalities) are supported, like [text](/docs/guides/text-generation), [images](/docs/guides/vision), and [audio](/docs/guides/audio).  | 
**best_of** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Number of output sequences that are generated from the prompt.             From these &#x60;best_of&#x60; sequences, the top &#x60;n&#x60; sequences are returned.             &#x60;best_of&#x60; must be greater than or equal to &#x60;n&#x60;. This is treated as             the beam width when &#x60;use_beam_search&#x60; is True. By default, &#x60;best_of&#x60;             is set to &#x60;n&#x60;. | [optional] 
**top_k** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Controls the number of top tokens to consider. -1 means consider all tokens. | [optional] 
**min_p** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Float that represents the minimum probability for a token to be             considered, relative to the probability of the most likely token.             Must be in [0, 1]. Set to 0 to disable this. | [optional] 
**use_beam_search** | None, bool,  | NoneClass, BoolClass,  | Whether to use beam search for sampling. | [optional] 
**length_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Float that penalizes sequences based on their length.             Used in beam search. | [optional] 
**repetition_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Float that penalizes new tokens based on whether             they appear in the prompt and the generated text so far. Values &gt; 1             encourage the model to use new tokens, while values &lt; 1 encourage             the model to repeat tokens. | [optional] 
**early_stopping** | None, bool,  | NoneClass, BoolClass,  | Controls the stopping condition for beam search. It             accepts the following values: &#x60;True&#x60;, where the generation stops as             soon as there are &#x60;best_of&#x60; complete candidates; &#x60;False&#x60;, where an             heuristic is applied and the generation stops when is it very             unlikely to find better candidates; &#x60;\&quot;never\&quot;&#x60;, where the beam search             procedure only stops when there cannot be better candidates             (canonical beam search algorithm). | [optional] 
**[stop_token_ids](#stop_token_ids)** | list, tuple, None,  | tuple, NoneClass,  | List of tokens that stop the generation when they are             generated. The returned output will contain the stop tokens unless             the stop tokens are special tokens. | [optional] 
**include_stop_str_in_output** | None, bool,  | NoneClass, BoolClass,  | Whether to include the stop strings in             output text. Defaults to False. | [optional] 
**ignore_eos** | None, bool,  | NoneClass, BoolClass,  | Whether to ignore the EOS token and continue generating             tokens after the EOS token is generated. | [optional] 
**min_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Minimum number of tokens to generate per output sequence             before EOS or stop_token_ids can be generated | [optional] 
**skip_special_tokens** | None, bool,  | NoneClass, BoolClass,  | Whether to skip special tokens in the output. Only supported in vllm. | [optional] if omitted the server will use the default value of True
**spaces_between_special_tokens** | None, bool,  | NoneClass, BoolClass,  | Whether to add spaces between special tokens in the output. Only supported in vllm. | [optional] if omitted the server will use the default value of True
**echo** | None, bool,  | NoneClass, BoolClass,  | If true, the new message will be prepended with the last message if they belong to the same role. | [optional] 
**add_generation_prompt** | None, bool,  | NoneClass, BoolClass,  | If true, the generation prompt will be added to the chat template. This is a parameter used by chat template in tokenizer config of the model. | [optional] 
**continue_final_message** | None, bool,  | NoneClass, BoolClass,  | If this is set, the chat will be formatted so that the final message in the chat is open-ended, without any EOS tokens. The model will continue this message rather than starting a new one. This allows you to \&quot;prefill\&quot; part of the model&#x27;s response for it. Cannot be used at the same time as &#x60;add_generation_prompt&#x60;. | [optional] 
**add_special_tokens** | None, bool,  | NoneClass, BoolClass,  | If true, special tokens (e.g. BOS) will be added to the prompt on top of what is added by the chat template. For most models, the chat template takes care of adding the special tokens so this should be set to false (as is the default). | [optional] 
**[documents](#documents)** | list, tuple, None,  | tuple, NoneClass,  | A list of dicts representing documents that will be accessible to the model if it is performing RAG (retrieval-augmented generation). If the template does not support RAG, this argument will have no effect. We recommend that each document should be a dict containing \&quot;title\&quot; and \&quot;text\&quot; keys. | [optional] 
**chat_template** | None, str,  | NoneClass, str,  | A Jinja template to use for this conversion. As of transformers v4.44, default chat template is no longer allowed, so you must provide a chat template if the model&#x27;s tokenizer does not define one and no override template is given | [optional] 
**[chat_template_kwargs](#chat_template_kwargs)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Additional kwargs to pass to the template renderer. Will be accessible by the chat template. | [optional] 
**[guided_json](#guided_json)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | JSON schema for guided decoding. Only supported in vllm. | [optional] 
**guided_regex** | None, str,  | NoneClass, str,  | Regex for guided decoding. Only supported in vllm. | [optional] 
**[guided_choice](#guided_choice)** | list, tuple, None,  | tuple, NoneClass,  | Choices for guided decoding. Only supported in vllm. | [optional] 
**guided_grammar** | None, str,  | NoneClass, str,  | Context-free grammar for guided decoding. Only supported in vllm. | [optional] 
**guided_decoding_backend** | None, str,  | NoneClass, str,  | If specified, will override the default guided decoding backend of the server for this specific request. If set, must be either &#x27;outlines&#x27; / &#x27;lm-format-enforcer&#x27; | [optional] 
**guided_whitespace_pattern** | None, str,  | NoneClass, str,  | If specified, will override the default whitespace pattern for guided json decoding. | [optional] 
**priority** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The priority of the request (lower means earlier handling; default: 0). Any priority other than 0 will raise an error if the served model does not use priority scheduling. | [optional] 
**metadata** | [**Metadata**](Metadata.md) | [**Metadata**](Metadata.md) |  | [optional] 
**temperature** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or &#x60;top_p&#x60; but not both.  | [optional] if omitted the server will use the default value of 1
**top_p** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or &#x60;temperature&#x60; but not both.  | [optional] if omitted the server will use the default value of 1
**user** | None, str,  | NoneClass, str,  | A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices#end-user-ids).  | [optional] 
**service_tier** | [**ServiceTier**](ServiceTier.md) | [**ServiceTier**](ServiceTier.md) |  | [optional] 
**model** | None, str,  | NoneClass, str,  |  | [optional] 
**modalities** | [**ResponseModalities**](ResponseModalities.md) | [**ResponseModalities**](ResponseModalities.md) |  | [optional] 
**reasoning_effort** | [**ReasoningEffort**](ReasoningEffort.md) | [**ReasoningEffort**](ReasoningEffort.md) |  | [optional] 
**max_completion_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and [reasoning tokens](/docs/guides/reasoning).  | [optional] 
**frequency_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model&#x27;s likelihood to repeat the same line verbatim.  | [optional] if omitted the server will use the default value of 0
**presence_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model&#x27;s likelihood to talk about new topics.  | [optional] if omitted the server will use the default value of 0
**web_search_options** | [**WebSearchOptions**](WebSearchOptions.md) | [**WebSearchOptions**](WebSearchOptions.md) |  | [optional] 
**top_logprobs** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability. &#x60;logprobs&#x60; must be set to &#x60;true&#x60; if this parameter is used.  | [optional] 
**[response_format](#response_format)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | An object specifying the format that the model must output.  Setting to &#x60;{ \&quot;type\&quot;: \&quot;json_schema\&quot;, \&quot;json_schema\&quot;: {...} }&#x60; enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the [Structured Outputs guide](/docs/guides/structured-outputs).  Setting to &#x60;{ \&quot;type\&quot;: \&quot;json_object\&quot; }&#x60; enables the older JSON mode, which ensures the message the model generates is valid JSON. Using &#x60;json_schema&#x60; is preferred for models that support it.  | [optional] 
**audio** | [**Audio2**](Audio2.md) | [**Audio2**](Audio2.md) |  | [optional] 
**store** | None, bool,  | NoneClass, BoolClass,  | Whether or not to store the output of this chat completion request for  use in our [model distillation](/docs/guides/distillation) or [evals](/docs/guides/evals) products.  | [optional] if omitted the server will use the default value of False
**stream** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of False
**stop** | [**StopConfiguration**](StopConfiguration.md) | [**StopConfiguration**](StopConfiguration.md) |  | [optional] 
**[logit_bias](#logit_bias)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  | [optional] 
**logprobs** | None, bool,  | NoneClass, BoolClass,  | Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the &#x60;content&#x60; of &#x60;message&#x60;.  | [optional] if omitted the server will use the default value of False
**max_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The maximum number of [tokens](/tokenizer) that can be generated in the chat completion. This value can be used to control [costs](https://openai.com/api/pricing/) for text generated via API.  This value is now deprecated in favor of &#x60;max_completion_tokens&#x60;, and is not compatible with [o-series models](/docs/guides/reasoning).  | [optional] 
**n** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | How many chat completion choices to generate for each input message. Note that you will be charged based on the number of generated tokens across all of the choices. Keep &#x60;n&#x60; as &#x60;1&#x60; to minimize costs. | [optional] if omitted the server will use the default value of 1
**prediction** | [**PredictionContent**](PredictionContent.md) | [**PredictionContent**](PredictionContent.md) |  | [optional] 
**seed** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same &#x60;seed&#x60; and parameters should return the same result. Determinism is not guaranteed, and you should refer to the &#x60;system_fingerprint&#x60; response parameter to monitor changes in the backend.  | [optional] 
**stream_options** | [**ChatCompletionStreamOptions**](ChatCompletionStreamOptions.md) | [**ChatCompletionStreamOptions**](ChatCompletionStreamOptions.md) |  | [optional] 
**[tools](#tools)** | list, tuple, None,  | tuple, NoneClass,  | A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.  | [optional] 
**tool_choice** | [**ChatCompletionToolChoiceOption**](ChatCompletionToolChoiceOption.md) | [**ChatCompletionToolChoiceOption**](ChatCompletionToolChoiceOption.md) |  | [optional] 
**parallel_tool_calls** | bool,  | BoolClass,  | Whether to enable [parallel function calling](/docs/guides/function-calling#configuring-parallel-function-calling) during tool use. | [optional] 
**[function_call](#function_call)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Deprecated in favor of &#x60;tool_choice&#x60;.  Controls which (if any) function is called by the model.  &#x60;none&#x60; means the model will not call a function and instead generates a message.  &#x60;auto&#x60; means the model can pick between generating a message or calling a function.  Specifying a particular function via &#x60;{\&quot;name\&quot;: \&quot;my_function\&quot;}&#x60; forces the model to call that function.  &#x60;none&#x60; is the default when no functions are present. &#x60;auto&#x60; is the default if functions are present.  | [optional] 
**[functions](#functions)** | list, tuple, None,  | tuple, NoneClass,  | Deprecated in favor of &#x60;tools&#x60;.  A list of functions the model may generate JSON inputs for.  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# messages

A list of messages comprising the conversation so far. Depending on the [model](/docs/models) you use, different message types (modalities) are supported, like [text](/docs/guides/text-generation), [images](/docs/guides/vision), and [audio](/docs/guides/audio). 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  | A list of messages comprising the conversation so far. Depending on the [model](/docs/models) you use, different message types (modalities) are supported, like [text](/docs/guides/text-generation), [images](/docs/guides/vision), and [audio](/docs/guides/audio).  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ChatCompletionRequestMessage**](ChatCompletionRequestMessage.md) | [**ChatCompletionRequestMessage**](ChatCompletionRequestMessage.md) | [**ChatCompletionRequestMessage**](ChatCompletionRequestMessage.md) |  | 

# stop_token_ids

List of tokens that stop the generation when they are             generated. The returned output will contain the stop tokens unless             the stop tokens are special tokens.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | List of tokens that stop the generation when they are             generated. The returned output will contain the stop tokens unless             the stop tokens are special tokens. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | decimal.Decimal, int,  | decimal.Decimal,  |  | 

# documents

A list of dicts representing documents that will be accessible to the model if it is performing RAG (retrieval-augmented generation). If the template does not support RAG, this argument will have no effect. We recommend that each document should be a dict containing \"title\" and \"text\" keys.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | A list of dicts representing documents that will be accessible to the model if it is performing RAG (retrieval-augmented generation). If the template does not support RAG, this argument will have no effect. We recommend that each document should be a dict containing \&quot;title\&quot; and \&quot;text\&quot; keys. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[items](#items) | dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

# items

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

# chat_template_kwargs

Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Additional kwargs to pass to the template renderer. Will be accessible by the chat template. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# guided_json

JSON schema for guided decoding. Only supported in vllm.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | JSON schema for guided decoding. Only supported in vllm. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# guided_choice

Choices for guided decoding. Only supported in vllm.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | Choices for guided decoding. Only supported in vllm. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# response_format

An object specifying the format that the model must output.  Setting to `{ \"type\": \"json_schema\", \"json_schema\": {...} }` enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the [Structured Outputs guide](/docs/guides/structured-outputs).  Setting to `{ \"type\": \"json_object\" }` enables the older JSON mode, which ensures the message the model generates is valid JSON. Using `json_schema` is preferred for models that support it. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | An object specifying the format that the model must output.  Setting to &#x60;{ \&quot;type\&quot;: \&quot;json_schema\&quot;, \&quot;json_schema\&quot;: {...} }&#x60; enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the [Structured Outputs guide](/docs/guides/structured-outputs).  Setting to &#x60;{ \&quot;type\&quot;: \&quot;json_object\&quot; }&#x60; enables the older JSON mode, which ensures the message the model generates is valid JSON. Using &#x60;json_schema&#x60; is preferred for models that support it.  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[ResponseFormatText](ResponseFormatText.md) | [**ResponseFormatText**](ResponseFormatText.md) | [**ResponseFormatText**](ResponseFormatText.md) |  | 
[ResponseFormatJsonSchema](ResponseFormatJsonSchema.md) | [**ResponseFormatJsonSchema**](ResponseFormatJsonSchema.md) | [**ResponseFormatJsonSchema**](ResponseFormatJsonSchema.md) |  | 
[ResponseFormatJsonObject](ResponseFormatJsonObject.md) | [**ResponseFormatJsonObject**](ResponseFormatJsonObject.md) | [**ResponseFormatJsonObject**](ResponseFormatJsonObject.md) |  | 

# logit_bias

Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | decimal.Decimal, int,  | decimal.Decimal,  | any string name can be used but the value must be the correct type | [optional] 

# tools

A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for. A max of 128 functions are supported.  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ChatCompletionTool**](ChatCompletionTool.md) | [**ChatCompletionTool**](ChatCompletionTool.md) | [**ChatCompletionTool**](ChatCompletionTool.md) |  | 

# function_call

Deprecated in favor of `tool_choice`.  Controls which (if any) function is called by the model.  `none` means the model will not call a function and instead generates a message.  `auto` means the model can pick between generating a message or calling a function.  Specifying a particular function via `{\"name\": \"my_function\"}` forces the model to call that function.  `none` is the default when no functions are present. `auto` is the default if functions are present. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Deprecated in favor of &#x60;tool_choice&#x60;.  Controls which (if any) function is called by the model.  &#x60;none&#x60; means the model will not call a function and instead generates a message.  &#x60;auto&#x60; means the model can pick between generating a message or calling a function.  Specifying a particular function via &#x60;{\&quot;name\&quot;: \&quot;my_function\&quot;}&#x60; forces the model to call that function.  &#x60;none&#x60; is the default when no functions are present. &#x60;auto&#x60; is the default if functions are present.  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | must be one of ["none", "auto", ] 
[ChatCompletionFunctionCallOption](ChatCompletionFunctionCallOption.md) | [**ChatCompletionFunctionCallOption**](ChatCompletionFunctionCallOption.md) | [**ChatCompletionFunctionCallOption**](ChatCompletionFunctionCallOption.md) |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | must be one of ["none", "auto", ] 

# functions

Deprecated in favor of `tools`.  A list of functions the model may generate JSON inputs for. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | Deprecated in favor of &#x60;tools&#x60;.  A list of functions the model may generate JSON inputs for.  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[**ChatCompletionFunctions**](ChatCompletionFunctions.md) | [**ChatCompletionFunctions**](ChatCompletionFunctions.md) | [**ChatCompletionFunctions**](ChatCompletionFunctions.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

