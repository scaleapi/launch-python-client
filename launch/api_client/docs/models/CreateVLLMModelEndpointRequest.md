# launch.api_client.model.create_vllm_model_endpoint_request.CreateVLLMModelEndpointRequest

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**[metadata](#metadata)** | dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 
**model_name** | str,  | str,  |  | 
**max_workers** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**min_workers** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**name** | str,  | str,  |  | 
**per_worker** | decimal.Decimal, int,  | decimal.Decimal,  |  | 
**[labels](#labels)** | dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 
**quantize** | [**Quantization**](Quantization.md) | [**Quantization**](Quantization.md) |  | [optional] 
**checkpoint_path** | None, str,  | NoneClass, str,  |  | [optional] 
**[post_inference_hooks](#post_inference_hooks)** | list, tuple, None,  | tuple, NoneClass,  |  | [optional] 
**[cpus](#cpus)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] 
**gpus** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**[memory](#memory)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] 
**gpu_type** | [**GpuType**](GpuType.md) | [**GpuType**](GpuType.md) |  | [optional] 
**[storage](#storage)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | [optional] 
**nodes_per_worker** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**optimize_costs** | None, bool,  | NoneClass, BoolClass,  |  | [optional] 
**prewarm** | None, bool,  | NoneClass, BoolClass,  |  | [optional] 
**high_priority** | None, bool,  | NoneClass, BoolClass,  |  | [optional] 
**[billing_tags](#billing_tags)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**default_callback_url** | None, str,  | NoneClass, str,  |  | [optional] 
**default_callback_auth** | [**CallbackAuth**](CallbackAuth.md) | [**CallbackAuth**](CallbackAuth.md) |  | [optional] 
**public_inference** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of True
**chat_template_override** | None, str,  | NoneClass, str,  | A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint | [optional] 
**enable_startup_metrics** | None, bool,  | NoneClass, BoolClass,  | Enable startup metrics collection via OpenTelemetry. When enabled, emits traces and metrics for download, Python init, and vLLM init phases. | [optional] if omitted the server will use the default value of False
**source** | [**LLMSource**](LLMSource.md) | [**LLMSource**](LLMSource.md) |  | [optional] 
**inference_framework_image_tag** | str,  | str,  |  | [optional] if omitted the server will use the default value of "latest"
**num_shards** | decimal.Decimal, int,  | decimal.Decimal,  |  | [optional] if omitted the server will use the default value of 1
**endpoint_type** | [**ModelEndpointType**](ModelEndpointType.md) | [**ModelEndpointType**](ModelEndpointType.md) |  | [optional] 
**max_gpu_memory_utilization** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Maximum GPU memory utilization for the batch inference. Default to 90%. Deprecated in favor of specifying this in VLLMModelConfig | [optional] 
**attention_backend** | None, str,  | NoneClass, str,  | Attention backend to use for vLLM. Default to None. | [optional] 
**max_model_len** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Model context length, If unspecified, will be automatically derived from the model config | [optional] 
**max_num_seqs** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Maximum number of sequences per iteration | [optional] 
**enforce_eager** | None, bool,  | NoneClass, BoolClass,  | Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal perforamnce and flexibility | [optional] 
**trust_remote_code** | None, bool,  | NoneClass, BoolClass,  | Whether to trust remote code from Hugging face hub. This is only applicable to models whose code is not supported natively by the transformers library (e.g. deepseek). Default to False. | [optional] if omitted the server will use the default value of False
**pipeline_parallel_size** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Number of pipeline stages. Default to None. | [optional] 
**tensor_parallel_size** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Number of tensor parallel replicas. Default to None. | [optional] 
**quantization** | None, str,  | NoneClass, str,  | Method used to quantize the weights. If None, we first check the &#x60;quantization_config&#x60; attribute in the model config file. If that is None, we assume the model weights are not quantized and use &#x60;dtype&#x60; to determine the data type of the weights. | [optional] 
**disable_log_requests** | None, bool,  | NoneClass, BoolClass,  | Disable logging requests. Default to None. | [optional] 
**chat_template** | None, str,  | NoneClass, str,  | A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint | [optional] 
**tool_call_parser** | None, str,  | NoneClass, str,  | Tool call parser | [optional] 
**enable_auto_tool_choice** | None, bool,  | NoneClass, BoolClass,  | Enable auto tool choice | [optional] 
**load_format** | None, str,  | NoneClass, str,  | The format of the model weights to load.  * \&quot;auto\&quot; will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available. * \&quot;pt\&quot; will load the weights in the pytorch bin format. * \&quot;safetensors\&quot; will load the weights in the safetensors format. * \&quot;npcache\&quot; will load the weights in pytorch format and store a numpy cache to speed up the loading. * \&quot;dummy\&quot; will initialize the weights with random values, which is mainly for profiling. * \&quot;tensorizer\&quot; will load the weights using tensorizer from CoreWeave. See the Tensorize vLLM Model script in the Examples section for more information. * \&quot;bitsandbytes\&quot; will load the weights using bitsandbytes quantization.  | [optional] 
**config_format** | None, str,  | NoneClass, str,  | The config format which shall be loaded.  Defaults to &#x27;auto&#x27; which defaults to &#x27;hf&#x27;. | [optional] 
**tokenizer_mode** | None, str,  | NoneClass, str,  | Tokenizer mode. &#x27;auto&#x27; will use the fast tokenizer ifavailable, &#x27;slow&#x27; will always use the slow tokenizer, and&#x27;mistral&#x27; will always use the tokenizer from &#x60;mistral_common&#x60;. | [optional] 
**limit_mm_per_prompt** | None, str,  | NoneClass, str,  | Maximum number of data instances per modality per prompt. Only applicable for multimodal models. | [optional] 
**max_num_batched_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Maximum number of batched tokens per iteration | [optional] 
**tokenizer** | None, str,  | NoneClass, str,  | Name or path of the huggingface tokenizer to use. | [optional] 
**dtype** | None, str,  | NoneClass, str,  | Data type for model weights and activations. The &#x27;auto&#x27; option will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models. | [optional] 
**seed** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Random seed for reproducibility. | [optional] 
**revision** | None, str,  | NoneClass, str,  | The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | [optional] 
**code_revision** | None, str,  | NoneClass, str,  | The specific revision to use for the model code on Hugging Face Hub. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | [optional] 
**[rope_scaling](#rope_scaling)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Dictionary containing the scaling configuration for the RoPE embeddings. When using this flag, don&#x27;t update &#x60;max_position_embeddings&#x60; to the expected new maximum. | [optional] 
**tokenizer_revision** | None, str,  | NoneClass, str,  | The specific tokenizer version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version. | [optional] 
**quantization_param_path** | None, str,  | NoneClass, str,  | Path to JSON file containing scaling factors. Used to load KV cache scaling factors into the model when KV cache type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also be used to load activation and weight scaling factors when the model dtype is FP8_E4M3 on ROCm. | [optional] 
**max_seq_len_to_capture** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Maximum sequence len covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode. Additionally for encoder-decoder models, if the sequence length of the encoder input is larger than this, we fall back to the eager mode. | [optional] 
**disable_sliding_window** | None, bool,  | NoneClass, BoolClass,  | Whether to disable sliding window. If True, we will disable the sliding window functionality of the model. If the model does not support sliding window, this argument is ignored. | [optional] 
**skip_tokenizer_init** | None, bool,  | NoneClass, BoolClass,  | If true, skip initialization of tokenizer and detokenizer. | [optional] 
**served_model_name** | None, str,  | NoneClass, str,  | The model name used in metrics tag &#x60;model_name&#x60;, matches the model name exposed via the APIs. If multiple model names provided, the first name will be used. If not specified, the model name will be the same as &#x60;model&#x60;. | [optional] 
**[override_neuron_config](#override_neuron_config)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Initialize non default neuron config or override default neuron config that are specific to Neuron devices, this argument will be used to configure the neuron config that can not be gathered from the vllm arguments. | [optional] 
**[mm_processor_kwargs](#mm_processor_kwargs)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Arguments to be forwarded to the model&#x27;s processor for multi-modal data, e.g., image processor. | [optional] 
**block_size** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Size of a cache block in number of tokens. | [optional] 
**gpu_memory_utilization** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Fraction of GPU memory to use for the vLLM execution. | [optional] 
**swap_space** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Size of the CPU swap space per GPU (in GiB). | [optional] 
**cache_dtype** | None, str,  | NoneClass, str,  | Data type for kv cache storage. | [optional] 
**num_gpu_blocks_override** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Number of GPU blocks to use. This overrides the profiled num_gpu_blocks if specified. Does nothing if None. | [optional] 
**enable_prefix_caching** | None, bool,  | NoneClass, BoolClass,  | Enables automatic prefix caching. | [optional] 
**inference_framework** | str,  | str,  |  | [optional] must be one of ["vllm", ] if omitted the server will use the default value of "vllm"
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# metadata

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# labels

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

# post_inference_hooks

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# cpus

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | 
[any_of_1](#any_of_1) | decimal.Decimal, int,  | decimal.Decimal,  |  | 
[any_of_2](#any_of_2) | decimal.Decimal, int, float,  | decimal.Decimal,  |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

# any_of_1

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | 

# any_of_2

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int, float,  | decimal.Decimal,  |  | 

# memory

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | 
[any_of_1](#any_of_1) | decimal.Decimal, int,  | decimal.Decimal,  |  | 
[any_of_2](#any_of_2) | decimal.Decimal, int, float,  | decimal.Decimal,  |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

# any_of_1

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | 

# any_of_2

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int, float,  | decimal.Decimal,  |  | 

# storage

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[any_of_0](#any_of_0) | str,  | str,  |  | 
[any_of_1](#any_of_1) | decimal.Decimal, int,  | decimal.Decimal,  |  | 
[any_of_2](#any_of_2) | decimal.Decimal, int, float,  | decimal.Decimal,  |  | 

# any_of_0

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

# any_of_1

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int,  | decimal.Decimal,  |  | 

# any_of_2

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
decimal.Decimal, int, float,  | decimal.Decimal,  |  | 

# billing_tags

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# rope_scaling

Dictionary containing the scaling configuration for the RoPE embeddings. When using this flag, don't update `max_position_embeddings` to the expected new maximum.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Dictionary containing the scaling configuration for the RoPE embeddings. When using this flag, don&#x27;t update &#x60;max_position_embeddings&#x60; to the expected new maximum. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# override_neuron_config

Initialize non default neuron config or override default neuron config that are specific to Neuron devices, this argument will be used to configure the neuron config that can not be gathered from the vllm arguments.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Initialize non default neuron config or override default neuron config that are specific to Neuron devices, this argument will be used to configure the neuron config that can not be gathered from the vllm arguments. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# mm_processor_kwargs

Arguments to be forwarded to the model's processor for multi-modal data, e.g., image processor.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  | Arguments to be forwarded to the model&#x27;s processor for multi-modal data, e.g., image processor. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

