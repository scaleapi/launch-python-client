# launch.api_client.model.create_text_generation_inference_model_endpoint_request.CreateTextGenerationInferenceModelEndpointRequest

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
**inference_framework** | str,  | str,  |  | [optional] must be one of ["text_generation_inference", ] if omitted the server will use the default value of "text_generation_inference"
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

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

