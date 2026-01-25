# launch.api_client.model.response_modalities.ResponseModalities

Output types that you would like the model to generate. Most models are capable of generating text, which is the default:  `[\"text\"]`  The `gpt-4o-audio-preview` model can also be used to  [generate audio](/docs/guides/audio). To request that this model generate  both text and audio responses, you can use:  `[\"text\", \"audio\"]` 

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO | Output types that you would like the model to generate. Most models are capable of generating text, which is the default:  &#x60;[\&quot;text\&quot;]&#x60;  The &#x60;gpt-4o-audio-preview&#x60; model can also be used to  [generate audio](/docs/guides/audio). To request that this model generate  both text and audio responses, you can use:  &#x60;[\&quot;text\&quot;, \&quot;audio\&quot;]&#x60;  | 

[[Back to Model list]](../../README.md#documentation-for-models) [[Back to API list]](../../README.md#documentation-for-api-endpoints) [[Back to README]](../../README.md)

