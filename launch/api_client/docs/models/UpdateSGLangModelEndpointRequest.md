# launch.api_client.model.update_sg_lang_model_endpoint_request.UpdateSGLangModelEndpointRequest

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
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
**model_name** | None, str,  | NoneClass, str,  |  | [optional] 
**source** | [**LLMSource**](LLMSource.md) | [**LLMSource**](LLMSource.md) |  | [optional] 
**inference_framework** | str,  | str,  |  | [optional] must be one of ["sglang", ] if omitted the server will use the default value of "sglang"
**inference_framework_image_tag** | None, str,  | NoneClass, str,  |  | [optional] 
**num_shards** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**[metadata](#metadata)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**force_bundle_recreation** | None, bool,  | NoneClass, BoolClass,  |  | [optional] if omitted the server will use the default value of False
**min_workers** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**max_workers** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**per_worker** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**[labels](#labels)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**trust_remote_code** | None, bool,  | NoneClass, BoolClass,  | Whether to trust remote code from Hugging face hub. This is only applicable to models whose code is not supported natively by the transformers library (e.g. deepseek). Default to False. | [optional] if omitted the server will use the default value of False
**tp_size** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The tensor parallel size. | [optional] 
**skip_tokenizer_init** | None, bool,  | NoneClass, BoolClass,  | If set, skip init tokenizer and pass input_ids in generate request | [optional] 
**load_format** | None, str,  | NoneClass, str,  | The format of the model weights to load. | [optional] 
**dtype** | None, str,  | NoneClass, str,  | Data type for model weights and activations. | [optional] 
**kv_cache_dtype** | None, str,  | NoneClass, str,  | Data type for kv cache storage. \&quot;auto\&quot; will use model data type. | [optional] 
**quantization_param_path** | None, str,  | NoneClass, str,  | Path to the JSON file containing the KV cache scaling factors. | [optional] 
**quantization** | None, str,  | NoneClass, str,  | The quantization method. | [optional] 
**context_length** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The model&#x27;s maximum context length. | [optional] 
**device** | None, str,  | NoneClass, str,  | The device type. | [optional] 
**served_model_name** | None, str,  | NoneClass, str,  | Override the model name returned by the v1/models endpoint in OpenAI API server. | [optional] 
**chat_template** | None, str,  | NoneClass, str,  | The builtin chat template name or path of the chat template file. | [optional] 
**is_embedding** | None, bool,  | NoneClass, BoolClass,  | Whether to use a CausalLM as an embedding model. | [optional] 
**revision** | None, str,  | NoneClass, str,  | The specific model version to use. | [optional] 
**mem_fraction_static** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | The fraction of the memory used for static allocation. | [optional] 
**max_running_requests** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The maximum number of running requests. | [optional] 
**max_total_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The maximum number of tokens in the memory pool. | [optional] 
**chunked_prefill_size** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The maximum number of tokens in a chunk for the chunked prefill. | [optional] 
**max_prefill_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The maximum number of tokens in a prefill batch. | [optional] 
**schedule_policy** | None, str,  | NoneClass, str,  | The scheduling policy of the requests. | [optional] 
**schedule_conservativeness** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | How conservative the schedule policy is. | [optional] 
**cpu_offload_gb** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | How many GBs of RAM to reserve for CPU offloading | [optional] 
**prefill_only_one_req** | None, bool,  | NoneClass, BoolClass,  | If true, we only prefill one request at one prefill batch | [optional] 
**stream_interval** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The interval for streaming in terms of the token length. | [optional] 
**random_seed** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The random seed. | [optional] 
**constrained_json_whitespace_pattern** | None, str,  | NoneClass, str,  | Regex pattern for syntactic whitespaces allowed in JSON constrained output. | [optional] 
**watchdog_timeout** | None, decimal.Decimal, int, float,  | NoneClass, decimal.Decimal,  | Set watchdog timeout in seconds. | [optional] 
**download_dir** | None, str,  | NoneClass, str,  | Model download directory. | [optional] 
**base_gpu_id** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The base GPU ID to start allocating GPUs from. | [optional] 
**log_level** | None, str,  | NoneClass, str,  | The logging level of all loggers. | [optional] 
**log_level_http** | None, str,  | NoneClass, str,  | The logging level of HTTP server. | [optional] 
**log_requests** | None, bool,  | NoneClass, BoolClass,  | Log the inputs and outputs of all requests. | [optional] 
**show_time_cost** | None, bool,  | NoneClass, BoolClass,  | Show time cost of custom marks. | [optional] 
**enable_metrics** | None, bool,  | NoneClass, BoolClass,  | Enable log prometheus metrics. | [optional] 
**decode_log_interval** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The log interval of decode batch. | [optional] 
**api_key** | None, str,  | NoneClass, str,  | Set API key of the server. | [optional] 
**file_storage_pth** | None, str,  | NoneClass, str,  | The path of the file storage in backend. | [optional] 
**enable_cache_report** | None, bool,  | NoneClass, BoolClass,  | Return number of cached tokens in usage.prompt_tokens_details. | [optional] 
**data_parallel_size** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The data parallelism size. | [optional] 
**load_balance_method** | None, str,  | NoneClass, str,  | The load balancing strategy for data parallelism. | [optional] 
**expert_parallel_size** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The expert parallelism size. | [optional] 
**dist_init_addr** | None, str,  | NoneClass, str,  | The host address for initializing distributed backend. | [optional] 
**nnodes** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The number of nodes. | [optional] 
**node_rank** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The node rank. | [optional] 
**json_model_override_args** | None, str,  | NoneClass, str,  | A dictionary in JSON string format used to override default model configurations. | [optional] 
**[lora_paths](#lora_paths)** | list, tuple, None,  | tuple, NoneClass,  | The list of LoRA adapters. | [optional] 
**max_loras_per_batch** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Maximum number of adapters for a running batch. | [optional] 
**attention_backend** | None, str,  | NoneClass, str,  | Choose the kernels for attention layers. | [optional] 
**sampling_backend** | None, str,  | NoneClass, str,  | Choose the kernels for sampling layers. | [optional] 
**grammar_backend** | None, str,  | NoneClass, str,  | Choose the backend for grammar-guided decoding. | [optional] 
**speculative_algorithm** | None, str,  | NoneClass, str,  | Speculative algorithm. | [optional] 
**speculative_draft_model_path** | None, str,  | NoneClass, str,  | The path of the draft model weights. | [optional] 
**speculative_num_steps** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The number of steps sampled from draft model in Speculative Decoding. | [optional] 
**speculative_num_draft_tokens** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The number of token sampled from draft model in Speculative Decoding. | [optional] 
**speculative_eagle_topk** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The number of token sampled from draft model in eagle2 each step. | [optional] 
**enable_double_sparsity** | None, bool,  | NoneClass, BoolClass,  | Enable double sparsity attention | [optional] 
**ds_channel_config_path** | None, str,  | NoneClass, str,  | The path of the double sparsity channel config | [optional] 
**ds_heavy_channel_num** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The number of heavy channels in double sparsity attention | [optional] 
**ds_heavy_token_num** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The number of heavy tokens in double sparsity attention | [optional] 
**ds_heavy_channel_type** | None, str,  | NoneClass, str,  | The type of heavy channels in double sparsity attention | [optional] 
**ds_sparse_decode_threshold** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The threshold for sparse decoding in double sparsity attention | [optional] 
**disable_radix_cache** | None, bool,  | NoneClass, BoolClass,  | Disable RadixAttention for prefix caching. | [optional] 
**disable_jump_forward** | None, bool,  | NoneClass, BoolClass,  | Disable jump-forward for grammar-guided decoding. | [optional] 
**disable_cuda_graph** | None, bool,  | NoneClass, BoolClass,  | Disable cuda graph. | [optional] 
**disable_cuda_graph_padding** | None, bool,  | NoneClass, BoolClass,  | Disable cuda graph when padding is needed. | [optional] 
**disable_outlines_disk_cache** | None, bool,  | NoneClass, BoolClass,  | Disable disk cache of outlines. | [optional] 
**disable_custom_all_reduce** | None, bool,  | NoneClass, BoolClass,  | Disable the custom all-reduce kernel. | [optional] 
**disable_mla** | None, bool,  | NoneClass, BoolClass,  | Disable Multi-head Latent Attention (MLA) for DeepSeek-V2. | [optional] 
**disable_overlap_schedule** | None, bool,  | NoneClass, BoolClass,  | Disable the overlap scheduler. | [optional] 
**enable_mixed_chunk** | None, bool,  | NoneClass, BoolClass,  | Enable mixing prefill and decode in a batch when using chunked prefill. | [optional] 
**enable_dp_attention** | None, bool,  | NoneClass, BoolClass,  | Enable data parallelism for attention and tensor parallelism for FFN. | [optional] 
**enable_ep_moe** | None, bool,  | NoneClass, BoolClass,  | Enable expert parallelism for moe. | [optional] 
**enable_torch_compile** | None, bool,  | NoneClass, BoolClass,  | Optimize the model with torch.compile. | [optional] 
**torch_compile_max_bs** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Set the maximum batch size when using torch compile. | [optional] 
**cuda_graph_max_bs** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Set the maximum batch size for cuda graph. | [optional] 
**[cuda_graph_bs](#cuda_graph_bs)** | list, tuple, None,  | tuple, NoneClass,  | Set the list of batch sizes for cuda graph. | [optional] 
**torchao_config** | None, str,  | NoneClass, str,  | Optimize the model with torchao. | [optional] 
**enable_nan_detection** | None, bool,  | NoneClass, BoolClass,  | Enable the NaN detection for debugging purposes. | [optional] 
**enable_p2p_check** | None, bool,  | NoneClass, BoolClass,  | Enable P2P check for GPU access. | [optional] 
**triton_attention_reduce_in_fp32** | None, bool,  | NoneClass, BoolClass,  | Cast the intermediate attention results to fp32. | [optional] 
**triton_attention_num_kv_splits** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | The number of KV splits in flash decoding Triton kernel. | [optional] 
**num_continuous_decode_steps** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  | Run multiple continuous decoding steps to reduce scheduling overhead. | [optional] 
**delete_ckpt_after_loading** | None, bool,  | NoneClass, BoolClass,  | Delete the model checkpoint after loading the model. | [optional] 
**enable_memory_saver** | None, bool,  | NoneClass, BoolClass,  | Allow saving memory using release_memory_occupation and resume_memory_occupation | [optional] 
**allow_auto_truncate** | None, bool,  | NoneClass, BoolClass,  | Allow automatically truncating requests that exceed the maximum input length. | [optional] 
**enable_custom_logit_processor** | None, bool,  | NoneClass, BoolClass,  | Enable users to pass custom logit processors to the server. | [optional] 
**tool_call_parser** | None, str,  | NoneClass, str,  | Specify the parser for handling tool-call interactions. | [optional] 
**huggingface_repo** | None, str,  | NoneClass, str,  | The Hugging Face repository ID. | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

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

# metadata

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

# labels

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | str,  | str,  | any string name can be used but the value must be the correct type | [optional] 

# lora_paths

The list of LoRA adapters.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | The list of LoRA adapters. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# cuda_graph_bs

Set the list of batch sizes for cuda graph.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple, None,  | tuple, NoneClass,  | Set the list of batch sizes for cuda graph. | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | decimal.Decimal, int,  | decimal.Decimal,  |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

