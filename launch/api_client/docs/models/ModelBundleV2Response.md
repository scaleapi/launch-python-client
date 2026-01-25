# launch.api_client.model.model_bundle_v2_response.ModelBundleV2Response

Response object for a single Model Bundle.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Response object for a single Model Bundle. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**[flavor](#flavor)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 
**[metadata](#metadata)** | dict, frozendict.frozendict,  | frozendict.frozendict,  |  | 
**[model_artifact_ids](#model_artifact_ids)** | list, tuple,  | tuple,  |  | 
**name** | str,  | str,  |  | 
**created_at** | str, datetime,  | str,  |  | value must conform to RFC-3339 date-time
**id** | str,  | str,  |  | 
**schema_location** | None, str,  | NoneClass, str,  |  | [optional] 
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

# model_artifact_ids

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
list, tuple,  | tuple,  |  | 

### Tuple Items
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
items | str,  | str,  |  | 

# flavor

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### oneOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[CloudpickleArtifactFlavor](CloudpickleArtifactFlavor.md) | [**CloudpickleArtifactFlavor**](CloudpickleArtifactFlavor.md) | [**CloudpickleArtifactFlavor**](CloudpickleArtifactFlavor.md) |  | 
[ZipArtifactFlavor](ZipArtifactFlavor.md) | [**ZipArtifactFlavor**](ZipArtifactFlavor.md) | [**ZipArtifactFlavor**](ZipArtifactFlavor.md) |  | 
[RunnableImageFlavor](RunnableImageFlavor.md) | [**RunnableImageFlavor**](RunnableImageFlavor.md) | [**RunnableImageFlavor**](RunnableImageFlavor.md) |  | 
[StreamingEnhancedRunnableImageFlavor](StreamingEnhancedRunnableImageFlavor.md) | [**StreamingEnhancedRunnableImageFlavor**](StreamingEnhancedRunnableImageFlavor.md) | [**StreamingEnhancedRunnableImageFlavor**](StreamingEnhancedRunnableImageFlavor.md) |  | 
[TritonEnhancedRunnableImageFlavor](TritonEnhancedRunnableImageFlavor.md) | [**TritonEnhancedRunnableImageFlavor**](TritonEnhancedRunnableImageFlavor.md) | [**TritonEnhancedRunnableImageFlavor**](TritonEnhancedRunnableImageFlavor.md) |  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

