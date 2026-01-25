# launch.api_client.model.filtered_completion_v2_request.FilteredCompletionV2Request

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**[prompt](#prompt)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.  Note that &lt;|endoftext|&gt; is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document.  | 
**best_of** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Generates &#x60;best_of&#x60; completions server-side and returns the \&quot;best\&quot; (the one with the highest log probability per token). Results cannot be streamed.  When used with &#x60;n&#x60;, &#x60;best_of&#x60; controls the number of candidate completions and &#x60;n&#x60; specifies how many to return – &#x60;best_of&#x60; must be greater than &#x60;n&#x60;.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for &#x60;max_tokens&#x60; and &#x60;stop&#x60;.  | [optional] if omitted the server will use the default value of 1
**top_k** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Controls the number of top tokens to consider. -1 means consider all tokens. | [optional] 
**min_p** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Float that represents the minimum probability for a token to be             considered, relative to the probability of the most likely token.             Must be in [0, 1]. Set to 0 to disable this. | [optional] 
**use_beam_search** | None, bool,  | NoneClass, BoolClass,  | Whether to use beam search for sampling. | [optional] 
**length_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Float that penalizes sequences based on their length.             Used in beam search. | [optional] 
**repetition_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Float that penalizes new tokens based on whether             they appear in the prompt and the generated text so far. Values &gt; 1             encourage the model to use new tokens, while values &lt; 1 encourage             the model to repeat tokens. | [optional] 
**early_stopping** | None, bool,  | NoneClass, BoolClass,  | Controls the stopping condition for beam search. It             accepts the following values: &#x60;True&#x60;, where the generation stops as             soon as there are &#x60;best_of&#x60; complete candidates; &#x60;False&#x60;, where an             heuristic is applied and the generation stops when is it very             unlikely to find better candidates; &#x60;\&quot;never\&quot;&#x60;, where the beam search             procedure only stops when there cannot be better candidates             (canonical beam search algorithm). | [optional] 
**[stop_token_ids](#stop_token_ids)** | list, tuple, None,  | tuple, NoneClass,  | List of tokens that stop the generation when they are             generated. The returned output will contain the stop tokens unless             the stop tokens are special tokens. | [optional] 
**include_stop_str_in_output** | None, bool,  | NoneClass, BoolClass,  | Whether to include the stop strings in output text. | [optional] 
**ignore_eos** | None, bool,  | NoneClass, BoolClass,  | Whether to ignore the EOS token and continue generating             tokens after the EOS token is generated. | [optional] 
**min_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Minimum number of tokens to generate per output sequence             before EOS or stop_token_ids can be generated | [optional] 
**skip_special_tokens** | None, bool,  | NoneClass, BoolClass,  | Whether to skip special tokens in the output. Only supported in vllm. | [optional] if omitted the server will use the default value of True
**spaces_between_special_tokens** | None, bool,  | NoneClass, BoolClass,  | Whether to add spaces between special tokens in the output. Only supported in vllm. | [optional] if omitted the server will use the default value of True
**add_special_tokens** | None, bool,  | NoneClass, BoolClass,  | If true (the default), special tokens (e.g. BOS) will be added to the prompt. | [optional] 
**[response_format](#response_format)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Similar to chat completion, this parameter specifies the format of output. Only {&#x27;type&#x27;: &#x27;json_object&#x27;} or {&#x27;type&#x27;: &#x27;text&#x27; } is supported. | [optional] 
**[guided_json](#guided_json)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | JSON schema for guided decoding. Only supported in vllm. | [optional] 
**guided_regex** | None, str,  | NoneClass, str,  | Regex for guided decoding. Only supported in vllm. | [optional] 
**[guided_choice](#guided_choice)** | list, tuple, None,  | tuple, NoneClass,  | Choices for guided decoding. Only supported in vllm. | [optional] 
**guided_grammar** | None, str,  | NoneClass, str,  | Context-free grammar for guided decoding. Only supported in vllm. | [optional] 
**guided_decoding_backend** | None, str,  | NoneClass, str,  | If specified, will override the default guided decoding backend of the server for this specific request. If set, must be either &#x27;outlines&#x27; / &#x27;lm-format-enforcer&#x27; | [optional] 
**guided_whitespace_pattern** | None, str,  | NoneClass, str,  | If specified, will override the default whitespace pattern for guided json decoding. | [optional] 
**model** | None, str,  | NoneClass, str,  |  | [optional] 
**echo** | None, bool,  | NoneClass, BoolClass,  | Echo back the prompt in addition to the completion  | [optional] if omitted the server will use the default value of False
**frequency_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model&#x27;s likelihood to repeat the same line verbatim.  [See more information about frequency and presence penalties.](/docs/guides/text-generation)  | [optional] if omitted the server will use the default value of 0
**[logit_bias](#logit_bias)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](/tokenizer?view&#x3D;bpe) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  As an example, you can pass &#x60;{\&quot;50256\&quot;: -100}&#x60; to prevent the &lt;|endoftext|&gt; token from being generated.  | [optional] 
**logprobs** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Include the log probabilities on the &#x60;logprobs&#x60; most likely output tokens, as well the chosen tokens. For example, if &#x60;logprobs&#x60; is 5, the API will return a list of the 5 most likely tokens. The API will always return the &#x60;logprob&#x60; of the sampled token, so there may be up to &#x60;logprobs+1&#x60; elements in the response.  The maximum value for &#x60;logprobs&#x60; is 5.  | [optional] 
**max_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The maximum number of [tokens](/tokenizer) that can be generated in the completion.  The token count of your prompt plus &#x60;max_tokens&#x60; cannot exceed the model&#x27;s context length. [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) for counting tokens.  | [optional] if omitted the server will use the default value of 16
**n** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | How many completions to generate for each prompt.  **Note:** Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for &#x60;max_tokens&#x60; and &#x60;stop&#x60;.  | [optional] if omitted the server will use the default value of 1
**presence_penalty** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model&#x27;s likelihood to talk about new topics.  [See more information about frequency and presence penalties.](/docs/guides/text-generation)  | [optional] if omitted the server will use the default value of 0
**seed** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same &#x60;seed&#x60; and parameters should return the same result.  Determinism is not guaranteed, and you should refer to the &#x60;system_fingerprint&#x60; response parameter to monitor changes in the backend.  | [optional] 
**stop** | [**StopConfiguration**](StopConfiguration.md) | [**StopConfiguration**](StopConfiguration.md) |  | [optional] 
**stream** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of False
**stream_options** | [**ChatCompletionStreamOptions**](ChatCompletionStreamOptions.md) | [**ChatCompletionStreamOptions**](ChatCompletionStreamOptions.md) |  | [optional] 
**suffix** | None, str,  | NoneClass, str,  | The suffix that comes after a completion of inserted text.  This parameter is only supported for &#x60;gpt-3.5-turbo-instruct&#x60;.  | [optional] 
**temperature** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.  We generally recommend altering this or &#x60;top_p&#x60; but not both.  | [optional] if omitted the server will use the default value of 1
**top_p** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.  We generally recommend altering this or &#x60;temperature&#x60; but not both.  | [optional] if omitted the server will use the default value of 1
**user** | None, str,  | NoneClass, str,  | A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. [Learn more](/docs/guides/safety-best-practices#end-user-ids).  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# prompt

The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.  Note that <|endoftext|> is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays.  Note that &lt;|endoftext|&gt; is the document separator that the model sees during training, so if a prompt is not specified the model will generate as if from the beginning of a new document.  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | 
[any_of_1](#any_of_1) | list, tuple,  | tuple,  |  | 
[Prompt](Prompt.md) | [**Prompt**](Prompt.md) | [**Prompt**](Prompt.md) |  | 
[Prompt1](Prompt1.md) | [**Prompt1**](Prompt1.md) | [**Prompt1**](Prompt1.md) |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

# any_of_1

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

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

# response_format

Similar to chat completion, this parameter specifies the format of output. Only {'type': 'json_object'} or {'type': 'text' } is supported.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Similar to chat completion, this parameter specifies the format of output. Only {&#x27;type&#x27;: &#x27;json_object&#x27;} or {&#x27;type&#x27;: &#x27;text&#x27; } is supported. | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[ResponseFormatText](ResponseFormatText.md) | [**ResponseFormatText**](ResponseFormatText.md) | [**ResponseFormatText**](ResponseFormatText.md) |  | 
[ResponseFormatJsonSchema](ResponseFormatJsonSchema.md) | [**ResponseFormatJsonSchema**](ResponseFormatJsonSchema.md) | [**ResponseFormatJsonSchema**](ResponseFormatJsonSchema.md) |  | 
[ResponseFormatJsonObject](ResponseFormatJsonObject.md) | [**ResponseFormatJsonObject**](ResponseFormatJsonObject.md) | [**ResponseFormatJsonObject**](ResponseFormatJsonObject.md) |  | 

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

# logit_bias

Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](/tokenizer?view=bpe) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  As an example, you can pass `{\"50256\": -100}` to prevent the <|endoftext|> token from being generated. 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Modify the likelihood of specified tokens appearing in the completion.  Accepts a JSON object that maps tokens (specified by their token ID in the GPT tokenizer) to an associated bias value from -100 to 100. You can use this [tokenizer tool](/tokenizer?view&#x3D;bpe) to convert text to token IDs. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.  As an example, you can pass &#x60;{\&quot;50256\&quot;: -100}&#x60; to prevent the &lt;|endoftext|&gt; token from being generated.  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | decimal.Decimal, int,  | decimal.Decimal,  | any string name can be used but the value must be the correct type | [optional] 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

