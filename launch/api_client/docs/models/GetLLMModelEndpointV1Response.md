# launch.api_client.model.get_llm_model_endpoint_v1_response.GetLLMModelEndpointV1Response

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**inference_framework** | [**LLMInferenceFramework**](LLMInferenceFramework.md) | [**LLMInferenceFramework**](LLMInferenceFramework.md) |  | 
**model_name** | str,  | str,  |  | 
**name** | str,  | str,  |  | 
**id** | str,  | str,  |  | 
**source** | [**LLMSource**](LLMSource.md) | [**LLMSource**](LLMSource.md) |  | 
**status** | [**ModelEndpointStatus**](ModelEndpointStatus.md) | [**ModelEndpointStatus**](ModelEndpointStatus.md) |  | 
**inference_framework_image_tag** | None, str,  | NoneClass, str,  |  | [optional] 
**num_shards** | None, decimal.Decimal, int,  | NoneClass, decimal.Decimal,  |  | [optional] 
**quantize** | [**Quantization**](Quantization.md) | [**Quantization**](Quantization.md) |  | [optional] 
**checkpoint_path** | None, str,  | NoneClass, str,  |  | [optional] 
**chat_template_override** | None, str,  | NoneClass, str,  | A Jinja template to use for this endpoint. If not provided, will use the chat template from the checkpoint | [optional] 
**spec** | [**GetModelEndpointV1Response**](GetModelEndpointV1Response.md) | [**GetModelEndpointV1Response**](GetModelEndpointV1Response.md) |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

