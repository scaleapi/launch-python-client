# launch.api_client.model.create_model_bundle_v2_request.CreateModelBundleV2Request

Request object for creating a Model Bundle.

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict,  | frozendict.frozendict,  | Request object for creating a Model Bundle. | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**[flavor](#flavor)** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 
**name** | str,  | str,  |  | 
**schema_location** | str,  | str,  |  | 
**[metadata](#metadata)** | dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | [optional] 
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, int, float, bool, decimal.Decimal, None, list, tuple, bytes, io.FileIO, io.BufferedReader | frozendict.frozendict, str, BoolClass, decimal.Decimal, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

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

# metadata

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, None,  | frozendict.frozendict, NoneClass,  |  | 

### Dictionary Keys
Key | Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | ------------- | -------------
**any_string_name** | dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | any string name can be used but the value must be the correct type | [optional]

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

