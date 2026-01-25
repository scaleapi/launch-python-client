<a name="__pageTop"></a>
# launch.api_client.apis.tags.default_api.DefaultApi

All URIs are relative to *http://localhost*

Method | HTTP request | Description
------------- | ------------- | -------------
[**batch_completions_v2_batch_completions_post**](#batch_completions_v2_batch_completions_post) | **post** /v2/batch-completions | Batch Completions
[**cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post**](#cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post) | **post** /v2/batch-completions/{batch_completion_id}/actions/cancel | Cancel Batch Completion
[**cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put**](#cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put) | **put** /v1/llm/fine-tunes/{fine_tune_id}/cancel | Cancel Fine Tune
[**chat_completion_v2_chat_completions_post**](#chat_completion_v2_chat_completions_post) | **post** /v2/chat/completions | Chat Completion
[**clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post**](#clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post) | **post** /v1/model-bundles/clone-with-changes | Clone Model Bundle With Changes
[**clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post**](#clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post) | **post** /v2/model-bundles/clone-with-changes | Clone Model Bundle With Changes
[**completion_v2_completions_post**](#completion_v2_completions_post) | **post** /v2/completions | Completion
[**create_async_inference_task_v1_async_tasks_post**](#create_async_inference_task_v1_async_tasks_post) | **post** /v1/async-tasks | Create Async Inference Task
[**create_batch_completions_v1_llm_batch_completions_post**](#create_batch_completions_v1_llm_batch_completions_post) | **post** /v1/llm/batch-completions | Create Batch Completions
[**create_batch_job_v1_batch_jobs_post**](#create_batch_job_v1_batch_jobs_post) | **post** /v1/batch-jobs | Create Batch Job
[**create_completion_stream_task_v1_llm_completions_stream_post**](#create_completion_stream_task_v1_llm_completions_stream_post) | **post** /v1/llm/completions-stream | Create Completion Stream Task
[**create_completion_sync_task_v1_llm_completions_sync_post**](#create_completion_sync_task_v1_llm_completions_sync_post) | **post** /v1/llm/completions-sync | Create Completion Sync Task
[**create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post**](#create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post) | **post** /v1/docker-image-batch-job-bundles | Create Docker Image Batch Job Bundle
[**create_docker_image_batch_job_v1_docker_image_batch_jobs_post**](#create_docker_image_batch_job_v1_docker_image_batch_jobs_post) | **post** /v1/docker-image-batch-jobs | Create Docker Image Batch Job
[**create_fine_tune_v1_llm_fine_tunes_post**](#create_fine_tune_v1_llm_fine_tunes_post) | **post** /v1/llm/fine-tunes | Create Fine Tune
[**create_model_bundle_v1_model_bundles_post**](#create_model_bundle_v1_model_bundles_post) | **post** /v1/model-bundles | Create Model Bundle
[**create_model_bundle_v2_model_bundles_post**](#create_model_bundle_v2_model_bundles_post) | **post** /v2/model-bundles | Create Model Bundle
[**create_model_endpoint_v1_llm_model_endpoints_post**](#create_model_endpoint_v1_llm_model_endpoints_post) | **post** /v1/llm/model-endpoints | Create Model Endpoint
[**create_model_endpoint_v1_model_endpoints_post**](#create_model_endpoint_v1_model_endpoints_post) | **post** /v1/model-endpoints | Create Model Endpoint
[**create_streaming_inference_task_v1_streaming_tasks_post**](#create_streaming_inference_task_v1_streaming_tasks_post) | **post** /v1/streaming-tasks | Create Streaming Inference Task
[**create_sync_inference_task_v1_sync_tasks_post**](#create_sync_inference_task_v1_sync_tasks_post) | **post** /v1/sync-tasks | Create Sync Inference Task
[**create_trigger_v1_triggers_post**](#create_trigger_v1_triggers_post) | **post** /v1/triggers | Create Trigger
[**delete_file_v1_files_file_id_delete**](#delete_file_v1_files_file_id_delete) | **delete** /v1/files/{file_id} | Delete File
[**delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete**](#delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete) | **delete** /v1/llm/model-endpoints/{model_endpoint_name} | Delete Llm Model Endpoint
[**delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete**](#delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete) | **delete** /v1/model-endpoints/{model_endpoint_id} | Delete Model Endpoint
[**delete_trigger_v1_triggers_trigger_id_delete**](#delete_trigger_v1_triggers_trigger_id_delete) | **delete** /v1/triggers/{trigger_id} | Delete Trigger
[**download_model_endpoint_v1_llm_model_endpoints_download_post**](#download_model_endpoint_v1_llm_model_endpoints_download_post) | **post** /v1/llm/model-endpoints/download | Download Model Endpoint
[**get_async_inference_task_v1_async_tasks_task_id_get**](#get_async_inference_task_v1_async_tasks_task_id_get) | **get** /v1/async-tasks/{task_id} | Get Async Inference Task
[**get_batch_completion_v2_batch_completions_batch_completion_id_get**](#get_batch_completion_v2_batch_completions_batch_completion_id_get) | **get** /v2/batch-completions/{batch_completion_id} | Get Batch Completion
[**get_batch_job_v1_batch_jobs_batch_job_id_get**](#get_batch_job_v1_batch_jobs_batch_job_id_get) | **get** /v1/batch-jobs/{batch_job_id} | Get Batch Job
[**get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get**](#get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get) | **get** /v1/docker-image-batch-job-bundles/{docker_image_batch_job_bundle_id} | Get Docker Image Batch Job Model Bundle
[**get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get**](#get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get) | **get** /v1/docker-image-batch-jobs/{batch_job_id} | Get Docker Image Batch Job
[**get_file_content_v1_files_file_id_content_get**](#get_file_content_v1_files_file_id_content_get) | **get** /v1/files/{file_id}/content | Get File Content
[**get_file_v1_files_file_id_get**](#get_file_v1_files_file_id_get) | **get** /v1/files/{file_id} | Get File
[**get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get**](#get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get) | **get** /v1/llm/fine-tunes/{fine_tune_id}/events | Get Fine Tune Events
[**get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get**](#get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get) | **get** /v1/llm/fine-tunes/{fine_tune_id} | Get Fine Tune
[**get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get**](#get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get) | **get** /v1/docker-image-batch-job-bundles/latest | Get Latest Docker Image Batch Job Bundle
[**get_latest_model_bundle_v1_model_bundles_latest_get**](#get_latest_model_bundle_v1_model_bundles_latest_get) | **get** /v1/model-bundles/latest | Get Latest Model Bundle
[**get_latest_model_bundle_v2_model_bundles_latest_get**](#get_latest_model_bundle_v2_model_bundles_latest_get) | **get** /v2/model-bundles/latest | Get Latest Model Bundle
[**get_model_bundle_v1_model_bundles_model_bundle_id_get**](#get_model_bundle_v1_model_bundles_model_bundle_id_get) | **get** /v1/model-bundles/{model_bundle_id} | Get Model Bundle
[**get_model_bundle_v2_model_bundles_model_bundle_id_get**](#get_model_bundle_v2_model_bundles_model_bundle_id_get) | **get** /v2/model-bundles/{model_bundle_id} | Get Model Bundle
[**get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get**](#get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get) | **get** /v1/llm/model-endpoints/{model_endpoint_name} | Get Model Endpoint
[**get_model_endpoint_v1_model_endpoints_model_endpoint_id_get**](#get_model_endpoint_v1_model_endpoints_model_endpoint_id_get) | **get** /v1/model-endpoints/{model_endpoint_id} | Get Model Endpoint
[**get_model_endpoints_api_v1_model_endpoints_api_get**](#get_model_endpoints_api_v1_model_endpoints_api_get) | **get** /v1/model-endpoints-api | Get Model Endpoints Api
[**get_model_endpoints_schema_v1_model_endpoints_schema_json_get**](#get_model_endpoints_schema_v1_model_endpoints_schema_json_get) | **get** /v1/model-endpoints-schema.json | Get Model Endpoints Schema
[**get_trigger_v1_triggers_trigger_id_get**](#get_trigger_v1_triggers_trigger_id_get) | **get** /v1/triggers/{trigger_id} | Get Trigger
[**healthcheck_healthcheck_get**](#healthcheck_healthcheck_get) | **get** /healthcheck | Healthcheck
[**healthcheck_healthz_get**](#healthcheck_healthz_get) | **get** /healthz | Healthcheck
[**healthcheck_readyz_get**](#healthcheck_readyz_get) | **get** /readyz | Healthcheck
[**list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get**](#list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get) | **get** /v1/docker-image-batch-job-bundles | List Docker Image Batch Job Model Bundles
[**list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get**](#list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get) | **get** /v1/docker-image-batch-jobs | List Docker Image Batch Jobs
[**list_files_v1_files_get**](#list_files_v1_files_get) | **get** /v1/files | List Files
[**list_fine_tunes_v1_llm_fine_tunes_get**](#list_fine_tunes_v1_llm_fine_tunes_get) | **get** /v1/llm/fine-tunes | List Fine Tunes
[**list_model_bundles_v1_model_bundles_get**](#list_model_bundles_v1_model_bundles_get) | **get** /v1/model-bundles | List Model Bundles
[**list_model_bundles_v2_model_bundles_get**](#list_model_bundles_v2_model_bundles_get) | **get** /v2/model-bundles | List Model Bundles
[**list_model_endpoints_v1_llm_model_endpoints_get**](#list_model_endpoints_v1_llm_model_endpoints_get) | **get** /v1/llm/model-endpoints | List Model Endpoints
[**list_model_endpoints_v1_model_endpoints_get**](#list_model_endpoints_v1_model_endpoints_get) | **get** /v1/model-endpoints | List Model Endpoints
[**list_triggers_v1_triggers_get**](#list_triggers_v1_triggers_get) | **get** /v1/triggers | List Triggers
[**restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post**](#restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post) | **post** /v1/model-endpoints/{model_endpoint_id}/restart | Restart Model Endpoint
[**update_batch_completion_v2_batch_completions_batch_completion_id_post**](#update_batch_completion_v2_batch_completions_batch_completion_id_post) | **post** /v2/batch-completions/{batch_completion_id} | Update Batch Completion
[**update_batch_job_v1_batch_jobs_batch_job_id_put**](#update_batch_job_v1_batch_jobs_batch_job_id_put) | **put** /v1/batch-jobs/{batch_job_id} | Update Batch Job
[**update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put**](#update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put) | **put** /v1/docker-image-batch-jobs/{batch_job_id} | Update Docker Image Batch Job
[**update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put**](#update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put) | **put** /v1/llm/model-endpoints/{model_endpoint_name} | Update Model Endpoint
[**update_model_endpoint_v1_model_endpoints_model_endpoint_id_put**](#update_model_endpoint_v1_model_endpoints_model_endpoint_id_put) | **put** /v1/model-endpoints/{model_endpoint_id} | Update Model Endpoint
[**update_trigger_v1_triggers_trigger_id_put**](#update_trigger_v1_triggers_trigger_id_put) | **put** /v1/triggers/{trigger_id} | Update Trigger
[**upload_file_v1_files_post**](#upload_file_v1_files_post) | **post** /v1/files | Upload File

# **batch_completions_v2_batch_completions_post**
<a name="batch_completions_v2_batch_completions_post"></a>
> BatchCompletionsJob batch_completions_v2_batch_completions_post(create_batch_completions_v2_request)

Batch Completions

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_batch_completions_v2_request import CreateBatchCompletionsV2Request
from launch.api_client.model.batch_completions_job import BatchCompletionsJob
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateBatchCompletionsV2Request(
        input_data_path="input_data_path_example",
        output_data_path="output_data_path_example",
        labels=dict(
            "key": "key_example",
        ),
        data_parallelism=1,
        max_runtime_sec=86400,
        priority="priority_example",
        tool_config=ToolConfig(
            name="name_example",
            max_iterations=10,
            execution_timeout_seconds=60,
            should_retry_on_error=True,
        ),
        cpus=None,
        gpus=1,
        memory=None,
        gpu_type=GpuType("nvidia-tesla-t4"),
        storage=None,
        nodes_per_worker=1,
        content=None,
        model_config=BatchCompletionsModelConfig(
            max_model_len=1,
            max_num_seqs=1,
            enforce_eager=True,
            trust_remote_code=False,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            quantization="quantization_example",
            disable_log_requests=True,
            chat_template="chat_template_example",
            tool_call_parser="tool_call_parser_example",
            enable_auto_tool_choice=True,
            load_format="load_format_example",
            config_format="config_format_example",
            tokenizer_mode="tokenizer_mode_example",
            limit_mm_per_prompt="limit_mm_per_prompt_example",
            max_num_batched_tokens=1,
            tokenizer="tokenizer_example",
            dtype="dtype_example",
            seed=1,
            revision="revision_example",
            code_revision="code_revision_example",
            rope_scaling=dict(),
            tokenizer_revision="tokenizer_revision_example",
            quantization_param_path="quantization_param_path_example",
            max_seq_len_to_capture=1,
            disable_sliding_window=True,
            skip_tokenizer_init=True,
            served_model_name="served_model_name_example",
            override_neuron_config=dict(),
            mm_processor_kwargs=dict(),
            block_size=1,
            gpu_memory_utilization=3.14,
            swap_space=3.14,
            cache_dtype="cache_dtype_example",
            num_gpu_blocks_override=1,
            enable_prefix_caching=True,
            model="mixtral-8x7b-instruct",
            checkpoint_path="checkpoint_path_example",
            num_shards=1,
            max_context_length=1.0,
            response_role="response_role_example",
        ),
    )
    try:
        # Batch Completions
        api_response = api_instance.batch_completions_v2_batch_completions_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->batch_completions_v2_batch_completions_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateBatchCompletionsV2Request**](../../models/CreateBatchCompletionsV2Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#batch_completions_v2_batch_completions_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#batch_completions_v2_batch_completions_post.ApiResponseFor422) | Validation Error

#### batch_completions_v2_batch_completions_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**BatchCompletionsJob**](../../models/BatchCompletionsJob.md) |  | 


#### batch_completions_v2_batch_completions_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post**
<a name="cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post"></a>
> CancelBatchCompletionsV2Response cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post(batch_completion_id)

Cancel Batch Completion

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.cancel_batch_completions_v2_response import CancelBatchCompletionsV2Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'batch_completion_id': "batch_completion_id_example",
    }
    try:
        # Cancel Batch Completion
        api_response = api_instance.cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
batch_completion_id | BatchCompletionIdSchema | | 

# BatchCompletionIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post.ApiResponseFor422) | Validation Error

#### cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CancelBatchCompletionsV2Response**](../../models/CancelBatchCompletionsV2Response.md) |  | 


#### cancel_batch_completion_v2_batch_completions_batch_completion_id_actions_cancel_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put**
<a name="cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put"></a>
> CancelFineTuneResponse cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put(fine_tune_id)

Cancel Fine Tune

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.cancel_fine_tune_response import CancelFineTuneResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'fine_tune_id': "fine_tune_id_example",
    }
    try:
        # Cancel Fine Tune
        api_response = api_instance.cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
fine_tune_id | FineTuneIdSchema | | 

# FineTuneIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put.ApiResponseFor422) | Validation Error

#### cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CancelFineTuneResponse**](../../models/CancelFineTuneResponse.md) |  | 


#### cancel_fine_tune_v1_llm_fine_tunes_fine_tune_id_cancel_put.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **chat_completion_v2_chat_completions_post**
<a name="chat_completion_v2_chat_completions_post"></a>
> bool, date, datetime, dict, float, int, list, str, none_type chat_completion_v2_chat_completions_post(chat_completion_v2_request)

Chat Completion

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.chat_completion_v2_request import ChatCompletionV2Request
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.chat_completion_v2_stream_error_chunk import ChatCompletionV2StreamErrorChunk
from launch.api_client.model.create_chat_completion_response import CreateChatCompletionResponse
from launch.api_client.model.create_chat_completion_stream_response import CreateChatCompletionStreamResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = ChatCompletionV2Request(
        best_of=1,
        top_k=-1.0,
        min_p=3.14,
        use_beam_search=True,
        length_penalty=3.14,
        repetition_penalty=3.14,
        early_stopping=True,
        stop_token_ids=[
            1
        ],
        include_stop_str_in_output=True,
        ignore_eos=True,
        min_tokens=1,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        echo=True,
        add_generation_prompt=True,
        continue_final_message=True,
        add_special_tokens=True,
        documents=[
            dict(
                "key": "key_example",
            )
        ],
        chat_template="chat_template_example",
        chat_template_kwargs=dict(),
        guided_json=dict(),
        guided_regex="guided_regex_example",
        guided_choice=[
            "guided_choice_example"
        ],
        guided_grammar="guided_grammar_example",
        guided_decoding_backend="guided_decoding_backend_example",
        guided_whitespace_pattern="guided_whitespace_pattern_example",
        priority=1,
        metadata=Metadata(
            key="key_example",
        ),
        temperature=1,
        top_p=1,
        user="user-1234",
        service_tier=ServiceTier("auto"),
        messages=[
            ChatCompletionRequestMessage(None)
        ],
        model="mixtral-8x7b-instruct",
        modalities=ResponseModalities([
            "text"
        ]),
        reasoning_effort=ReasoningEffort("medium"),
        max_completion_tokens=1,
        frequency_penalty=0,
        presence_penalty=0,
        web_search_options=WebSearchOptions(
            user_location=UserLocation(
                type="approximate",
                approximate=WebSearchLocation(
                    country="country_example",
                    region="region_example",
                    city="city_example",
                    timezone="timezone_example",
                ),
            ),
            search_context_size=WebSearchContextSize("low"),
        ),
        top_logprobs=0.0,
        response_format=None,
        audio=Audio2(
            voice=VoiceIdsShared(None),
            format="wav",
        ),
        store=False,
        stream=False,
        stop=StopConfiguration(None),
        logit_bias=dict(
            "key": 1,
        ),
        logprobs=False,
        max_tokens=1,
        n=1,
        prediction=PredictionContent(
            type="content",
            content=None,
        ),
        seed=-9.223372036854776E+18,
        stream_options=ChatCompletionStreamOptions(
            include_usage=True,
        ),
        tools=[
            ChatCompletionTool(
                type="function",
                function=FunctionObject(
                    description="description_example",
                    name="name_example",
                    parameters=FunctionParameters(),
                    strict=False,
                ),
            )
        ],
        tool_choice=ChatCompletionToolChoiceOption(None),
        parallel_tool_calls=True,
        function_call=None,
        functions=[
            ChatCompletionFunctions(
                description="description_example",
                name="name_example",
                parameters=FunctionParameters(),
            )
        ],
    )
    try:
        # Chat Completion
        api_response = api_instance.chat_completion_v2_chat_completions_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->chat_completion_v2_chat_completions_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ChatCompletionV2Request**](../../models/ChatCompletionV2Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#chat_completion_v2_chat_completions_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#chat_completion_v2_chat_completions_post.ApiResponseFor422) | Validation Error

#### chat_completion_v2_chat_completions_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[CreateChatCompletionResponse]({{complexTypePrefix}}CreateChatCompletionResponse.md) | [**CreateChatCompletionResponse**]({{complexTypePrefix}}CreateChatCompletionResponse.md) | [**CreateChatCompletionResponse**]({{complexTypePrefix}}CreateChatCompletionResponse.md) |  | 
[CreateChatCompletionStreamResponse]({{complexTypePrefix}}CreateChatCompletionStreamResponse.md) | [**CreateChatCompletionStreamResponse**]({{complexTypePrefix}}CreateChatCompletionStreamResponse.md) | [**CreateChatCompletionStreamResponse**]({{complexTypePrefix}}CreateChatCompletionStreamResponse.md) |  | 
[ChatCompletionV2StreamErrorChunk]({{complexTypePrefix}}ChatCompletionV2StreamErrorChunk.md) | [**ChatCompletionV2StreamErrorChunk**]({{complexTypePrefix}}ChatCompletionV2StreamErrorChunk.md) | [**ChatCompletionV2StreamErrorChunk**]({{complexTypePrefix}}ChatCompletionV2StreamErrorChunk.md) |  | 

#### chat_completion_v2_chat_completions_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post**
<a name="clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post"></a>
> CreateModelBundleV1Response clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post(clone_model_bundle_v1_request)

Clone Model Bundle With Changes

Creates a ModelBundle by cloning an existing one and then applying changes on top.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_model_bundle_v1_response import CreateModelBundleV1Response
from launch.api_client.model.clone_model_bundle_v1_request import CloneModelBundleV1Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CloneModelBundleV1Request(
        original_model_bundle_id="original_model_bundle_id_example",
        new_app_config=dict(),
    )
    try:
        # Clone Model Bundle With Changes
        api_response = api_instance.clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CloneModelBundleV1Request**](../../models/CloneModelBundleV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post.ApiResponseFor422) | Validation Error

#### clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateModelBundleV1Response**](../../models/CreateModelBundleV1Response.md) |  | 


#### clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post**
<a name="clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post"></a>
> CreateModelBundleV2Response clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post(clone_model_bundle_v2_request)

Clone Model Bundle With Changes

Creates a ModelBundle by cloning an existing one and then applying changes on top.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.clone_model_bundle_v2_request import CloneModelBundleV2Request
from launch.api_client.model.create_model_bundle_v2_response import CreateModelBundleV2Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CloneModelBundleV2Request(
        original_model_bundle_id="original_model_bundle_id_example",
        new_app_config=dict(),
    )
    try:
        # Clone Model Bundle With Changes
        api_response = api_instance.clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CloneModelBundleV2Request**](../../models/CloneModelBundleV2Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post.ApiResponseFor422) | Validation Error

#### clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateModelBundleV2Response**](../../models/CreateModelBundleV2Response.md) |  | 


#### clone_model_bundle_with_changes_v2_model_bundles_clone_with_changes_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **completion_v2_completions_post**
<a name="completion_v2_completions_post"></a>
> bool, date, datetime, dict, float, int, list, str, none_type completion_v2_completions_post(completion_v2_request)

Completion

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_completion_response import CreateCompletionResponse
from launch.api_client.model.completion_v2_stream_error_chunk import CompletionV2StreamErrorChunk
from launch.api_client.model.completion_v2_request import CompletionV2Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CompletionV2Request(
        best_of=1,
        top_k=-1.0,
        min_p=3.14,
        use_beam_search=True,
        length_penalty=3.14,
        repetition_penalty=3.14,
        early_stopping=True,
        stop_token_ids=[
            1
        ],
        include_stop_str_in_output=True,
        ignore_eos=True,
        min_tokens=1,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
        add_special_tokens=True,
        response_format=None,
        guided_json=dict(),
        guided_regex="guided_regex_example",
        guided_choice=[
            "guided_choice_example"
        ],
        guided_grammar="guided_grammar_example",
        guided_decoding_backend="guided_decoding_backend_example",
        guided_whitespace_pattern="guided_whitespace_pattern_example",
        model="mixtral-8x7b-instruct",
        prompt=None,
        echo=False,
        frequency_penalty=0,
        logit_bias=dict(
            "key": 1,
        ),
        logprobs=0.0,
        max_tokens=16,
        n=1,
        presence_penalty=0,
        seed=1,
        stop=StopConfiguration(None),
        stream=False,
        stream_options=ChatCompletionStreamOptions(
            include_usage=True,
        ),
        suffix="test.",
        temperature=1,
        top_p=1,
        user="user-1234",
    )
    try:
        # Completion
        api_response = api_instance.completion_v2_completions_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->completion_v2_completions_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CompletionV2Request**](../../models/CompletionV2Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#completion_v2_completions_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#completion_v2_completions_post.ApiResponseFor422) | Validation Error

#### completion_v2_completions_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Composed Schemas (allOf/anyOf/oneOf/not)
#### anyOf
Class Name | Input Type | Accessed Type | Description | Notes
------------- | ------------- | ------------- | ------------- | -------------
[CreateCompletionResponse]({{complexTypePrefix}}CreateCompletionResponse.md) | [**CreateCompletionResponse**]({{complexTypePrefix}}CreateCompletionResponse.md) | [**CreateCompletionResponse**]({{complexTypePrefix}}CreateCompletionResponse.md) |  | 
[CompletionV2StreamErrorChunk]({{complexTypePrefix}}CompletionV2StreamErrorChunk.md) | [**CompletionV2StreamErrorChunk**]({{complexTypePrefix}}CompletionV2StreamErrorChunk.md) | [**CompletionV2StreamErrorChunk**]({{complexTypePrefix}}CompletionV2StreamErrorChunk.md) |  | 

#### completion_v2_completions_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_async_inference_task_v1_async_tasks_post**
<a name="create_async_inference_task_v1_async_tasks_post"></a>
> CreateAsyncTaskV1Response create_async_inference_task_v1_async_tasks_post(model_endpoint_idendpoint_predict_v1_request)

Create Async Inference Task

Runs an async inference prediction.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_async_task_v1_response import CreateAsyncTaskV1Response
from launch.api_client.model.endpoint_predict_v1_request import EndpointPredictV1Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'model_endpoint_id': "model_endpoint_id_example",
    }
    body = EndpointPredictV1Request(
        url="url_example",
        args=None,
        cloudpickle="cloudpickle_example",
        callback_url="callback_url_example",
        callback_auth=CallbackAuth(
            kind="CallbackBasicAuth",
            username="username_example",
            password="password_example",
        ),
        return_pickled=False,
        destination_path="destination_path_example",
    )
    try:
        # Create Async Inference Task
        api_response = api_instance.create_async_inference_task_v1_async_tasks_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_async_inference_task_v1_async_tasks_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**EndpointPredictV1Request**](../../models/EndpointPredictV1Request.md) |  | 


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_id | ModelEndpointIdSchema | | 


# ModelEndpointIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_async_inference_task_v1_async_tasks_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_async_inference_task_v1_async_tasks_post.ApiResponseFor422) | Validation Error

#### create_async_inference_task_v1_async_tasks_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateAsyncTaskV1Response**](../../models/CreateAsyncTaskV1Response.md) |  | 


#### create_async_inference_task_v1_async_tasks_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_batch_completions_v1_llm_batch_completions_post**
<a name="create_batch_completions_v1_llm_batch_completions_post"></a>
> CreateBatchCompletionsV1Response create_batch_completions_v1_llm_batch_completions_post(create_batch_completions_v1_request)

Create Batch Completions

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_batch_completions_v1_response import CreateBatchCompletionsV1Response
from launch.api_client.model.create_batch_completions_v1_request import CreateBatchCompletionsV1Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateBatchCompletionsV1Request(
        input_data_path="input_data_path_example",
        output_data_path="output_data_path_example",
        labels=dict(
            "key": "key_example",
        ),
        data_parallelism=1,
        max_runtime_sec=86400,
        priority="priority_example",
        tool_config=ToolConfig(
            name="name_example",
            max_iterations=10,
            execution_timeout_seconds=60,
            should_retry_on_error=True,
        ),
        cpus=None,
        gpus=1,
        memory=None,
        gpu_type=GpuType("nvidia-tesla-t4"),
        storage=None,
        nodes_per_worker=1,
        content=CreateBatchCompletionsV1RequestContent(
            prompts=[
                "prompts_example"
            ],
            max_new_tokens=1,
            temperature=0.0,
            stop_sequences=[
                "stop_sequences_example"
            ],
            return_token_log_probs=False,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            top_k=-1.0,
            top_p=3.14,
            skip_special_tokens=True,
        ),
        model_config=CreateBatchCompletionsV1ModelConfig(
            max_model_len=1,
            max_num_seqs=1,
            enforce_eager=True,
            trust_remote_code=False,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            quantization="quantization_example",
            disable_log_requests=True,
            chat_template="chat_template_example",
            tool_call_parser="tool_call_parser_example",
            enable_auto_tool_choice=True,
            load_format="load_format_example",
            config_format="config_format_example",
            tokenizer_mode="tokenizer_mode_example",
            limit_mm_per_prompt="limit_mm_per_prompt_example",
            max_num_batched_tokens=1,
            tokenizer="tokenizer_example",
            dtype="dtype_example",
            seed=1,
            revision="revision_example",
            code_revision="code_revision_example",
            rope_scaling=dict(),
            tokenizer_revision="tokenizer_revision_example",
            quantization_param_path="quantization_param_path_example",
            max_seq_len_to_capture=1,
            disable_sliding_window=True,
            skip_tokenizer_init=True,
            served_model_name="served_model_name_example",
            override_neuron_config=dict(),
            mm_processor_kwargs=dict(),
            block_size=1,
            gpu_memory_utilization=3.14,
            swap_space=3.14,
            cache_dtype="cache_dtype_example",
            num_gpu_blocks_override=1,
            enable_prefix_caching=True,
            model="mixtral-8x7b-instruct",
            checkpoint_path="checkpoint_path_example",
            num_shards=1,
            max_context_length=1.0,
            response_role="response_role_example",
            labels=dict(),
        ),
    )
    try:
        # Create Batch Completions
        api_response = api_instance.create_batch_completions_v1_llm_batch_completions_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_batch_completions_v1_llm_batch_completions_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateBatchCompletionsV1Request**](../../models/CreateBatchCompletionsV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_batch_completions_v1_llm_batch_completions_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_batch_completions_v1_llm_batch_completions_post.ApiResponseFor422) | Validation Error

#### create_batch_completions_v1_llm_batch_completions_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateBatchCompletionsV1Response**](../../models/CreateBatchCompletionsV1Response.md) |  | 


#### create_batch_completions_v1_llm_batch_completions_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_batch_job_v1_batch_jobs_post**
<a name="create_batch_job_v1_batch_jobs_post"></a>
> CreateBatchJobV1Response create_batch_job_v1_batch_jobs_post(create_batch_job_v1_request)

Create Batch Job

Runs a batch job.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_batch_job_v1_request import CreateBatchJobV1Request
from launch.api_client.model.create_batch_job_v1_response import CreateBatchJobV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateBatchJobV1Request(
        model_bundle_id="model_bundle_id_example",
        input_path="input_path_example",
        serialization_format=BatchJobSerializationFormat("JSON"),
        labels=dict(
            "key": "key_example",
        ),
        resource_requests=CreateBatchJobResourceRequests(
            cpus=None,
            memory=None,
            gpus=1,
            gpu_type=GpuType("nvidia-tesla-t4"),
            storage=None,
            max_workers=1,
            per_worker=1,
            concurrent_requests_per_worker=1,
        ),
        timeout_seconds=43200.0,
    )
    try:
        # Create Batch Job
        api_response = api_instance.create_batch_job_v1_batch_jobs_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_batch_job_v1_batch_jobs_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateBatchJobV1Request**](../../models/CreateBatchJobV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_batch_job_v1_batch_jobs_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_batch_job_v1_batch_jobs_post.ApiResponseFor422) | Validation Error

#### create_batch_job_v1_batch_jobs_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateBatchJobV1Response**](../../models/CreateBatchJobV1Response.md) |  | 


#### create_batch_job_v1_batch_jobs_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_completion_stream_task_v1_llm_completions_stream_post**
<a name="create_completion_stream_task_v1_llm_completions_stream_post"></a>
> CompletionStreamV1Response create_completion_stream_task_v1_llm_completions_stream_post(model_endpoint_namecompletion_stream_v1_request)

Create Completion Stream Task

Runs a stream prompt completion on an LLM.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.completion_stream_v1_request import CompletionStreamV1Request
from launch.api_client.model.completion_stream_v1_response import CompletionStreamV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'model_endpoint_name': "model_endpoint_name_example",
    }
    body = CompletionStreamV1Request(
        prompt="prompt_example",
        max_new_tokens=1,
        temperature=0.0,
        stop_sequences=[
            "stop_sequences_example"
        ],
        return_token_log_probs=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_k=-1.0,
        top_p=3.14,
        include_stop_str_in_output=True,
        guided_json=dict(),
        guided_regex="guided_regex_example",
        guided_choice=[
            "guided_choice_example"
        ],
        guided_grammar="guided_grammar_example",
        skip_special_tokens=True,
    )
    try:
        # Create Completion Stream Task
        api_response = api_instance.create_completion_stream_task_v1_llm_completions_stream_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_completion_stream_task_v1_llm_completions_stream_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CompletionStreamV1Request**](../../models/CompletionStreamV1Request.md) |  | 


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_name | ModelEndpointNameSchema | | 


# ModelEndpointNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_completion_stream_task_v1_llm_completions_stream_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_completion_stream_task_v1_llm_completions_stream_post.ApiResponseFor422) | Validation Error

#### create_completion_stream_task_v1_llm_completions_stream_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CompletionStreamV1Response**](../../models/CompletionStreamV1Response.md) |  | 


#### create_completion_stream_task_v1_llm_completions_stream_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_completion_sync_task_v1_llm_completions_sync_post**
<a name="create_completion_sync_task_v1_llm_completions_sync_post"></a>
> CompletionSyncV1Response create_completion_sync_task_v1_llm_completions_sync_post(model_endpoint_namecompletion_sync_v1_request)

Create Completion Sync Task

Runs a sync prompt completion on an LLM.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.completion_sync_v1_request import CompletionSyncV1Request
from launch.api_client.model.completion_sync_v1_response import CompletionSyncV1Response
from launch.api_client.model.http_validation_error import HTTPValidationError
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'model_endpoint_name': "model_endpoint_name_example",
    }
    body = CompletionSyncV1Request(
        prompt="prompt_example",
        max_new_tokens=1,
        temperature=0.0,
        stop_sequences=[
            "stop_sequences_example"
        ],
        return_token_log_probs=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_k=-1.0,
        top_p=3.14,
        include_stop_str_in_output=True,
        guided_json=dict(),
        guided_regex="guided_regex_example",
        guided_choice=[
            "guided_choice_example"
        ],
        guided_grammar="guided_grammar_example",
        skip_special_tokens=True,
    )
    try:
        # Create Completion Sync Task
        api_response = api_instance.create_completion_sync_task_v1_llm_completions_sync_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_completion_sync_task_v1_llm_completions_sync_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CompletionSyncV1Request**](../../models/CompletionSyncV1Request.md) |  | 


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_name | ModelEndpointNameSchema | | 


# ModelEndpointNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_completion_sync_task_v1_llm_completions_sync_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_completion_sync_task_v1_llm_completions_sync_post.ApiResponseFor422) | Validation Error

#### create_completion_sync_task_v1_llm_completions_sync_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CompletionSyncV1Response**](../../models/CompletionSyncV1Response.md) |  | 


#### create_completion_sync_task_v1_llm_completions_sync_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post**
<a name="create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post"></a>
> CreateDockerImageBatchJobBundleV1Response create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post(create_docker_image_batch_job_bundle_v1_request)

Create Docker Image Batch Job Bundle

Creates a docker iamge batch job bundle

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_docker_image_batch_job_bundle_v1_response import CreateDockerImageBatchJobBundleV1Response
from launch.api_client.model.create_docker_image_batch_job_bundle_v1_request import CreateDockerImageBatchJobBundleV1Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateDockerImageBatchJobBundleV1Request(
        name="name_example",
        image_repository="image_repository_example",
        image_tag="image_tag_example",
        command=[
            "command_example"
        ],
        env=dict(
            "key": "key_example",
        ),
        mount_location="mount_location_example",
        resource_requests=CreateDockerImageBatchJobResourceRequests(
            cpus=None,
            memory=None,
            gpus=1,
            gpu_type=GpuType("nvidia-tesla-t4"),
            storage=None,
            nodes_per_worker=1,
        ),
        public=False,
    )
    try:
        # Create Docker Image Batch Job Bundle
        api_response = api_instance.create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateDockerImageBatchJobBundleV1Request**](../../models/CreateDockerImageBatchJobBundleV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post.ApiResponseFor422) | Validation Error

#### create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateDockerImageBatchJobBundleV1Response**](../../models/CreateDockerImageBatchJobBundleV1Response.md) |  | 


#### create_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_docker_image_batch_job_v1_docker_image_batch_jobs_post**
<a name="create_docker_image_batch_job_v1_docker_image_batch_jobs_post"></a>
> CreateDockerImageBatchJobV1Response create_docker_image_batch_job_v1_docker_image_batch_jobs_post(create_docker_image_batch_job_v1_request)

Create Docker Image Batch Job

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_docker_image_batch_job_v1_request import CreateDockerImageBatchJobV1Request
from launch.api_client.model.create_docker_image_batch_job_v1_response import CreateDockerImageBatchJobV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateDockerImageBatchJobV1Request(
        docker_image_batch_job_bundle_name="docker_image_batch_job_bundle_name_example",
        docker_image_batch_job_bundle_id="docker_image_batch_job_bundle_id_example",
        job_config=dict(),
        labels=dict(
            "key": "key_example",
        ),
        resource_requests=CreateDockerImageBatchJobResourceRequests(
            cpus=None,
            memory=None,
            gpus=1,
            gpu_type=GpuType("nvidia-tesla-t4"),
            storage=None,
            nodes_per_worker=1,
        ),
        override_job_max_runtime_s=1,
    )
    try:
        # Create Docker Image Batch Job
        api_response = api_instance.create_docker_image_batch_job_v1_docker_image_batch_jobs_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_docker_image_batch_job_v1_docker_image_batch_jobs_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateDockerImageBatchJobV1Request**](../../models/CreateDockerImageBatchJobV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_docker_image_batch_job_v1_docker_image_batch_jobs_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_docker_image_batch_job_v1_docker_image_batch_jobs_post.ApiResponseFor422) | Validation Error

#### create_docker_image_batch_job_v1_docker_image_batch_jobs_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateDockerImageBatchJobV1Response**](../../models/CreateDockerImageBatchJobV1Response.md) |  | 


#### create_docker_image_batch_job_v1_docker_image_batch_jobs_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_fine_tune_v1_llm_fine_tunes_post**
<a name="create_fine_tune_v1_llm_fine_tunes_post"></a>
> CreateFineTuneResponse create_fine_tune_v1_llm_fine_tunes_post(create_fine_tune_request)

Create Fine Tune

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_fine_tune_request import CreateFineTuneRequest
from launch.api_client.model.create_fine_tune_response import CreateFineTuneResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateFineTuneRequest(
        model="model_example",
        training_file="training_file_example",
        validation_file="validation_file_example",
        hyperparameters=dict(
            "key": None,
        ),
        suffix="suffix_example",
        wandb_config=dict(),
    )
    try:
        # Create Fine Tune
        api_response = api_instance.create_fine_tune_v1_llm_fine_tunes_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_fine_tune_v1_llm_fine_tunes_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateFineTuneRequest**](../../models/CreateFineTuneRequest.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_fine_tune_v1_llm_fine_tunes_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_fine_tune_v1_llm_fine_tunes_post.ApiResponseFor422) | Validation Error

#### create_fine_tune_v1_llm_fine_tunes_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateFineTuneResponse**](../../models/CreateFineTuneResponse.md) |  | 


#### create_fine_tune_v1_llm_fine_tunes_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_model_bundle_v1_model_bundles_post**
<a name="create_model_bundle_v1_model_bundles_post"></a>
> CreateModelBundleV1Response create_model_bundle_v1_model_bundles_post(create_model_bundle_v1_request)

Create Model Bundle

Creates a ModelBundle for the current user.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_model_bundle_v1_response import CreateModelBundleV1Response
from launch.api_client.model.create_model_bundle_v1_request import CreateModelBundleV1Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateModelBundleV1Request(
        name="name_example",
        location="location_example",
        requirements=[
            "requirements_example"
        ],
        env_params=ModelBundleEnvironmentParams(
            framework_type=ModelBundleFrameworkType("pytorch"),
            pytorch_image_tag="pytorch_image_tag_example",
            tensorflow_version="tensorflow_version_example",
            ecr_repo="ecr_repo_example",
            image_tag="image_tag_example",
        ),
        packaging_type=ModelBundlePackagingType("cloudpickle"),
        metadata=dict(),
        app_config=dict(),
        schema_location="schema_location_example",
    )
    try:
        # Create Model Bundle
        api_response = api_instance.create_model_bundle_v1_model_bundles_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_model_bundle_v1_model_bundles_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateModelBundleV1Request**](../../models/CreateModelBundleV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_model_bundle_v1_model_bundles_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_model_bundle_v1_model_bundles_post.ApiResponseFor422) | Validation Error

#### create_model_bundle_v1_model_bundles_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateModelBundleV1Response**](../../models/CreateModelBundleV1Response.md) |  | 


#### create_model_bundle_v1_model_bundles_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_model_bundle_v2_model_bundles_post**
<a name="create_model_bundle_v2_model_bundles_post"></a>
> CreateModelBundleV2Response create_model_bundle_v2_model_bundles_post(create_model_bundle_v2_request)

Create Model Bundle

Creates a ModelBundle for the current user.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_model_bundle_v2_request import CreateModelBundleV2Request
from launch.api_client.model.create_model_bundle_v2_response import CreateModelBundleV2Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateModelBundleV2Request(
        name="name_example",
        metadata=dict(),
        schema_location="schema_location_example",
        flavor=
            requirements=[
                "requirements_example"
            ],
            framework=
                framework_type="CustomFramework",
                image_repository="image_repository_example",
                image_tag="image_tag_example",
            ,
            app_config=dict(),
            location="location_example",
            flavor="CloudpickleArtifactFlavor",
            load_predict_fn="load_predict_fn_example",
            load_model_fn="load_model_fn_example",
        ,
    )
    try:
        # Create Model Bundle
        api_response = api_instance.create_model_bundle_v2_model_bundles_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_model_bundle_v2_model_bundles_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateModelBundleV2Request**](../../models/CreateModelBundleV2Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_model_bundle_v2_model_bundles_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_model_bundle_v2_model_bundles_post.ApiResponseFor422) | Validation Error

#### create_model_bundle_v2_model_bundles_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateModelBundleV2Response**](../../models/CreateModelBundleV2Response.md) |  | 


#### create_model_bundle_v2_model_bundles_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_model_endpoint_v1_llm_model_endpoints_post**
<a name="create_model_endpoint_v1_llm_model_endpoints_post"></a>
> CreateLLMModelEndpointV1Response create_model_endpoint_v1_llm_model_endpoints_post(create_llm_model_endpoint_v1_request)

Create Model Endpoint

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.create_llm_model_endpoint_v1_request import CreateLLMModelEndpointV1Request
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_llm_model_endpoint_v1_response import CreateLLMModelEndpointV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateLLMModelEndpointV1Request(None)
    try:
        # Create Model Endpoint
        api_response = api_instance.create_model_endpoint_v1_llm_model_endpoints_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_model_endpoint_v1_llm_model_endpoints_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateLLMModelEndpointV1Request**](../../models/CreateLLMModelEndpointV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_model_endpoint_v1_llm_model_endpoints_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_model_endpoint_v1_llm_model_endpoints_post.ApiResponseFor422) | Validation Error

#### create_model_endpoint_v1_llm_model_endpoints_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateLLMModelEndpointV1Response**](../../models/CreateLLMModelEndpointV1Response.md) |  | 


#### create_model_endpoint_v1_llm_model_endpoints_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_model_endpoint_v1_model_endpoints_post**
<a name="create_model_endpoint_v1_model_endpoints_post"></a>
> CreateModelEndpointV1Response create_model_endpoint_v1_model_endpoints_post(create_model_endpoint_v1_request)

Create Model Endpoint

Creates a Model for the current user.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.create_model_endpoint_v1_request import CreateModelEndpointV1Request
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_model_endpoint_v1_response import CreateModelEndpointV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateModelEndpointV1Request(
        name="name_example",
        model_bundle_id="model_bundle_id_example",
        endpoint_type=ModelEndpointType("async"),
        metadata=dict(),
        post_inference_hooks=[
            "post_inference_hooks_example"
        ],
        cpus=None,
        gpus=0.0,
        memory=None,
        gpu_type=GpuType("nvidia-tesla-t4"),
        storage=None,
        nodes_per_worker=1,
        optimize_costs=True,
        min_workers=0.0,
        max_workers=0.0,
        per_worker=1,
        concurrent_requests_per_worker=1,
        labels=dict(
            "key": "key_example",
        ),
        prewarm=True,
        high_priority=True,
        billing_tags=dict(),
        default_callback_url="default_callback_url_example",
        default_callback_auth=CallbackAuth(
            kind="CallbackBasicAuth",
            username="username_example",
            password="password_example",
        ),
        public_inference=False,
    )
    try:
        # Create Model Endpoint
        api_response = api_instance.create_model_endpoint_v1_model_endpoints_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_model_endpoint_v1_model_endpoints_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateModelEndpointV1Request**](../../models/CreateModelEndpointV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_model_endpoint_v1_model_endpoints_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_model_endpoint_v1_model_endpoints_post.ApiResponseFor422) | Validation Error

#### create_model_endpoint_v1_model_endpoints_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateModelEndpointV1Response**](../../models/CreateModelEndpointV1Response.md) |  | 


#### create_model_endpoint_v1_model_endpoints_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_streaming_inference_task_v1_streaming_tasks_post**
<a name="create_streaming_inference_task_v1_streaming_tasks_post"></a>
> bool, date, datetime, dict, float, int, list, str, none_type create_streaming_inference_task_v1_streaming_tasks_post(model_endpoint_idsync_endpoint_predict_v1_request)

Create Streaming Inference Task

Runs a streaming inference prediction.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.sync_endpoint_predict_v1_request import SyncEndpointPredictV1Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'model_endpoint_id': "model_endpoint_id_example",
    }
    body = SyncEndpointPredictV1Request(
        url="url_example",
        args=None,
        cloudpickle="cloudpickle_example",
        callback_url="callback_url_example",
        callback_auth=CallbackAuth(
            kind="CallbackBasicAuth",
            username="username_example",
            password="password_example",
        ),
        return_pickled=False,
        destination_path="destination_path_example",
        timeout_seconds=3.14,
        num_retries=0.0,
    )
    try:
        # Create Streaming Inference Task
        api_response = api_instance.create_streaming_inference_task_v1_streaming_tasks_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_streaming_inference_task_v1_streaming_tasks_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**SyncEndpointPredictV1Request**](../../models/SyncEndpointPredictV1Request.md) |  | 


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_id | ModelEndpointIdSchema | | 


# ModelEndpointIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_streaming_inference_task_v1_streaming_tasks_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_streaming_inference_task_v1_streaming_tasks_post.ApiResponseFor422) | Validation Error

#### create_streaming_inference_task_v1_streaming_tasks_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

#### create_streaming_inference_task_v1_streaming_tasks_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_sync_inference_task_v1_sync_tasks_post**
<a name="create_sync_inference_task_v1_sync_tasks_post"></a>
> SyncEndpointPredictV1Response create_sync_inference_task_v1_sync_tasks_post(model_endpoint_idsync_endpoint_predict_v1_request)

Create Sync Inference Task

Runs a sync inference prediction.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.sync_endpoint_predict_v1_request import SyncEndpointPredictV1Request
from launch.api_client.model.sync_endpoint_predict_v1_response import SyncEndpointPredictV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'model_endpoint_id': "model_endpoint_id_example",
    }
    body = SyncEndpointPredictV1Request(
        url="url_example",
        args=None,
        cloudpickle="cloudpickle_example",
        callback_url="callback_url_example",
        callback_auth=CallbackAuth(
            kind="CallbackBasicAuth",
            username="username_example",
            password="password_example",
        ),
        return_pickled=False,
        destination_path="destination_path_example",
        timeout_seconds=3.14,
        num_retries=0.0,
    )
    try:
        # Create Sync Inference Task
        api_response = api_instance.create_sync_inference_task_v1_sync_tasks_post(
            query_params=query_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_sync_inference_task_v1_sync_tasks_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
query_params | RequestQueryParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**SyncEndpointPredictV1Request**](../../models/SyncEndpointPredictV1Request.md) |  | 


### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_id | ModelEndpointIdSchema | | 


# ModelEndpointIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_sync_inference_task_v1_sync_tasks_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_sync_inference_task_v1_sync_tasks_post.ApiResponseFor422) | Validation Error

#### create_sync_inference_task_v1_sync_tasks_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**SyncEndpointPredictV1Response**](../../models/SyncEndpointPredictV1Response.md) |  | 


#### create_sync_inference_task_v1_sync_tasks_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **create_trigger_v1_triggers_post**
<a name="create_trigger_v1_triggers_post"></a>
> CreateTriggerV1Response create_trigger_v1_triggers_post(create_trigger_v1_request)

Create Trigger

Creates and runs a trigger

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.create_trigger_v1_response import CreateTriggerV1Response
from launch.api_client.model.create_trigger_v1_request import CreateTriggerV1Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = CreateTriggerV1Request(
        name="name_example",
        cron_schedule="cron_schedule_example",
        bundle_id="bundle_id_example",
        default_job_config=dict(),
        default_job_metadata=dict(
            "key": "key_example",
        ),
    )
    try:
        # Create Trigger
        api_response = api_instance.create_trigger_v1_triggers_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->create_trigger_v1_triggers_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateTriggerV1Request**](../../models/CreateTriggerV1Request.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#create_trigger_v1_triggers_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#create_trigger_v1_triggers_post.ApiResponseFor422) | Validation Error

#### create_trigger_v1_triggers_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**CreateTriggerV1Response**](../../models/CreateTriggerV1Response.md) |  | 


#### create_trigger_v1_triggers_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **delete_file_v1_files_file_id_delete**
<a name="delete_file_v1_files_file_id_delete"></a>
> DeleteFileResponse delete_file_v1_files_file_id_delete(file_id)

Delete File

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.delete_file_response import DeleteFileResponse
from launch.api_client.model.http_validation_error import HTTPValidationError
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'file_id': "file_id_example",
    }
    try:
        # Delete File
        api_response = api_instance.delete_file_v1_files_file_id_delete(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->delete_file_v1_files_file_id_delete: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
file_id | FileIdSchema | | 

# FileIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#delete_file_v1_files_file_id_delete.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#delete_file_v1_files_file_id_delete.ApiResponseFor422) | Validation Error

#### delete_file_v1_files_file_id_delete.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**DeleteFileResponse**](../../models/DeleteFileResponse.md) |  | 


#### delete_file_v1_files_file_id_delete.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete**
<a name="delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete"></a>
> DeleteLLMEndpointResponse delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete(model_endpoint_name)

Delete Llm Model Endpoint

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.delete_llm_endpoint_response import DeleteLLMEndpointResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_endpoint_name': "model_endpoint_name_example",
    }
    try:
        # Delete Llm Model Endpoint
        api_response = api_instance.delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_name | ModelEndpointNameSchema | | 

# ModelEndpointNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete.ApiResponseFor422) | Validation Error

#### delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**DeleteLLMEndpointResponse**](../../models/DeleteLLMEndpointResponse.md) |  | 


#### delete_llm_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_delete.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete**
<a name="delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete"></a>
> DeleteModelEndpointV1Response delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete(model_endpoint_id)

Delete Model Endpoint

Lists the Models owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.delete_model_endpoint_v1_response import DeleteModelEndpointV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_endpoint_id': "model_endpoint_id_example",
    }
    try:
        # Delete Model Endpoint
        api_response = api_instance.delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_id | ModelEndpointIdSchema | | 

# ModelEndpointIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete.ApiResponseFor422) | Validation Error

#### delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**DeleteModelEndpointV1Response**](../../models/DeleteModelEndpointV1Response.md) |  | 


#### delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **delete_trigger_v1_triggers_trigger_id_delete**
<a name="delete_trigger_v1_triggers_trigger_id_delete"></a>
> DeleteTriggerV1Response delete_trigger_v1_triggers_trigger_id_delete(trigger_id)

Delete Trigger

Deletes the trigger with the given ID

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.delete_trigger_v1_response import DeleteTriggerV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'trigger_id': "trigger_id_example",
    }
    try:
        # Delete Trigger
        api_response = api_instance.delete_trigger_v1_triggers_trigger_id_delete(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->delete_trigger_v1_triggers_trigger_id_delete: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
trigger_id | TriggerIdSchema | | 

# TriggerIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#delete_trigger_v1_triggers_trigger_id_delete.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#delete_trigger_v1_triggers_trigger_id_delete.ApiResponseFor422) | Validation Error

#### delete_trigger_v1_triggers_trigger_id_delete.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**DeleteTriggerV1Response**](../../models/DeleteTriggerV1Response.md) |  | 


#### delete_trigger_v1_triggers_trigger_id_delete.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **download_model_endpoint_v1_llm_model_endpoints_download_post**
<a name="download_model_endpoint_v1_llm_model_endpoints_download_post"></a>
> ModelDownloadResponse download_model_endpoint_v1_llm_model_endpoints_download_post(model_download_request)

Download Model Endpoint

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.model_download_response import ModelDownloadResponse
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_download_request import ModelDownloadRequest
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    body = ModelDownloadRequest(
        model_name="model_name_example",
        download_format="hugging_face",
    )
    try:
        # Download Model Endpoint
        api_response = api_instance.download_model_endpoint_v1_llm_model_endpoints_download_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->download_model_endpoint_v1_llm_model_endpoints_download_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelDownloadRequest**](../../models/ModelDownloadRequest.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#download_model_endpoint_v1_llm_model_endpoints_download_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#download_model_endpoint_v1_llm_model_endpoints_download_post.ApiResponseFor422) | Validation Error

#### download_model_endpoint_v1_llm_model_endpoints_download_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelDownloadResponse**](../../models/ModelDownloadResponse.md) |  | 


#### download_model_endpoint_v1_llm_model_endpoints_download_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_async_inference_task_v1_async_tasks_task_id_get**
<a name="get_async_inference_task_v1_async_tasks_task_id_get"></a>
> GetAsyncTaskV1Response get_async_inference_task_v1_async_tasks_task_id_get(task_id)

Get Async Inference Task

Gets the status of an async inference task.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.get_async_task_v1_response import GetAsyncTaskV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'task_id': "task_id_example",
    }
    try:
        # Get Async Inference Task
        api_response = api_instance.get_async_inference_task_v1_async_tasks_task_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_async_inference_task_v1_async_tasks_task_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
task_id | TaskIdSchema | | 

# TaskIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_async_inference_task_v1_async_tasks_task_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_async_inference_task_v1_async_tasks_task_id_get.ApiResponseFor422) | Validation Error

#### get_async_inference_task_v1_async_tasks_task_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetAsyncTaskV1Response**](../../models/GetAsyncTaskV1Response.md) |  | 


#### get_async_inference_task_v1_async_tasks_task_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_batch_completion_v2_batch_completions_batch_completion_id_get**
<a name="get_batch_completion_v2_batch_completions_batch_completion_id_get"></a>
> GetBatchCompletionV2Response get_batch_completion_v2_batch_completions_batch_completion_id_get(batch_completion_id)

Get Batch Completion

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.get_batch_completion_v2_response import GetBatchCompletionV2Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'batch_completion_id': "batch_completion_id_example",
    }
    try:
        # Get Batch Completion
        api_response = api_instance.get_batch_completion_v2_batch_completions_batch_completion_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_batch_completion_v2_batch_completions_batch_completion_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
batch_completion_id | BatchCompletionIdSchema | | 

# BatchCompletionIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_batch_completion_v2_batch_completions_batch_completion_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_batch_completion_v2_batch_completions_batch_completion_id_get.ApiResponseFor422) | Validation Error

#### get_batch_completion_v2_batch_completions_batch_completion_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetBatchCompletionV2Response**](../../models/GetBatchCompletionV2Response.md) |  | 


#### get_batch_completion_v2_batch_completions_batch_completion_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_batch_job_v1_batch_jobs_batch_job_id_get**
<a name="get_batch_job_v1_batch_jobs_batch_job_id_get"></a>
> GetBatchJobV1Response get_batch_job_v1_batch_jobs_batch_job_id_get(batch_job_id)

Get Batch Job

Gets a batch job.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.get_batch_job_v1_response import GetBatchJobV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'batch_job_id': "batch_job_id_example",
    }
    try:
        # Get Batch Job
        api_response = api_instance.get_batch_job_v1_batch_jobs_batch_job_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_batch_job_v1_batch_jobs_batch_job_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
batch_job_id | BatchJobIdSchema | | 

# BatchJobIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_batch_job_v1_batch_jobs_batch_job_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_batch_job_v1_batch_jobs_batch_job_id_get.ApiResponseFor422) | Validation Error

#### get_batch_job_v1_batch_jobs_batch_job_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetBatchJobV1Response**](../../models/GetBatchJobV1Response.md) |  | 


#### get_batch_job_v1_batch_jobs_batch_job_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get**
<a name="get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get"></a>
> DockerImageBatchJobBundleV1Response get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get(docker_image_batch_job_bundle_id)

Get Docker Image Batch Job Model Bundle

Get details for a given DockerImageBatchJobBundle owned by the current owner

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.docker_image_batch_job_bundle_v1_response import DockerImageBatchJobBundleV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'docker_image_batch_job_bundle_id': "docker_image_batch_job_bundle_id_example",
    }
    try:
        # Get Docker Image Batch Job Model Bundle
        api_response = api_instance.get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
docker_image_batch_job_bundle_id | DockerImageBatchJobBundleIdSchema | | 

# DockerImageBatchJobBundleIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get.ApiResponseFor422) | Validation Error

#### get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**DockerImageBatchJobBundleV1Response**](../../models/DockerImageBatchJobBundleV1Response.md) |  | 


#### get_docker_image_batch_job_model_bundle_v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get**
<a name="get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get"></a>
> GetDockerImageBatchJobV1Response get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get(batch_job_id)

Get Docker Image Batch Job

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.get_docker_image_batch_job_v1_response import GetDockerImageBatchJobV1Response
from launch.api_client.model.http_validation_error import HTTPValidationError
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'batch_job_id': "batch_job_id_example",
    }
    try:
        # Get Docker Image Batch Job
        api_response = api_instance.get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
batch_job_id | BatchJobIdSchema | | 

# BatchJobIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get.ApiResponseFor422) | Validation Error

#### get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetDockerImageBatchJobV1Response**](../../models/GetDockerImageBatchJobV1Response.md) |  | 


#### get_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_file_content_v1_files_file_id_content_get**
<a name="get_file_content_v1_files_file_id_content_get"></a>
> GetFileContentResponse get_file_content_v1_files_file_id_content_get(file_id)

Get File Content

Describe the LLM Model endpoint with given name.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.get_file_content_response import GetFileContentResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'file_id': "file_id_example",
    }
    try:
        # Get File Content
        api_response = api_instance.get_file_content_v1_files_file_id_content_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_file_content_v1_files_file_id_content_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
file_id | FileIdSchema | | 

# FileIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_file_content_v1_files_file_id_content_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_file_content_v1_files_file_id_content_get.ApiResponseFor422) | Validation Error

#### get_file_content_v1_files_file_id_content_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetFileContentResponse**](../../models/GetFileContentResponse.md) |  | 


#### get_file_content_v1_files_file_id_content_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_file_v1_files_file_id_get**
<a name="get_file_v1_files_file_id_get"></a>
> GetFileResponse get_file_v1_files_file_id_get(file_id)

Get File

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.get_file_response import GetFileResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'file_id': "file_id_example",
    }
    try:
        # Get File
        api_response = api_instance.get_file_v1_files_file_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_file_v1_files_file_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
file_id | FileIdSchema | | 

# FileIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_file_v1_files_file_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_file_v1_files_file_id_get.ApiResponseFor422) | Validation Error

#### get_file_v1_files_file_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetFileResponse**](../../models/GetFileResponse.md) |  | 


#### get_file_v1_files_file_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get**
<a name="get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get"></a>
> GetFineTuneEventsResponse get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get(fine_tune_id)

Get Fine Tune Events

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.get_fine_tune_events_response import GetFineTuneEventsResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'fine_tune_id': "fine_tune_id_example",
    }
    try:
        # Get Fine Tune Events
        api_response = api_instance.get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
fine_tune_id | FineTuneIdSchema | | 

# FineTuneIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get.ApiResponseFor422) | Validation Error

#### get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetFineTuneEventsResponse**](../../models/GetFineTuneEventsResponse.md) |  | 


#### get_fine_tune_events_v1_llm_fine_tunes_fine_tune_id_events_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get**
<a name="get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get"></a>
> GetFineTuneResponse get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get(fine_tune_id)

Get Fine Tune

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.get_fine_tune_response import GetFineTuneResponse
from launch.api_client.model.http_validation_error import HTTPValidationError
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'fine_tune_id': "fine_tune_id_example",
    }
    try:
        # Get Fine Tune
        api_response = api_instance.get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
fine_tune_id | FineTuneIdSchema | | 

# FineTuneIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get.ApiResponseFor422) | Validation Error

#### get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetFineTuneResponse**](../../models/GetFineTuneResponse.md) |  | 


#### get_fine_tune_v1_llm_fine_tunes_fine_tune_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get**
<a name="get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get"></a>
> DockerImageBatchJobBundleV1Response get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get(bundle_name)

Get Latest Docker Image Batch Job Bundle

Gets latest Docker Image Batch Job Bundle with given name owned by the current owner

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.docker_image_batch_job_bundle_v1_response import DockerImageBatchJobBundleV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'bundle_name': "bundle_name_example",
    }
    try:
        # Get Latest Docker Image Batch Job Bundle
        api_response = api_instance.get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
bundle_name | BundleNameSchema | | 


# BundleNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get.ApiResponseFor422) | Validation Error

#### get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**DockerImageBatchJobBundleV1Response**](../../models/DockerImageBatchJobBundleV1Response.md) |  | 


#### get_latest_docker_image_batch_job_bundle_v1_docker_image_batch_job_bundles_latest_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_latest_model_bundle_v1_model_bundles_latest_get**
<a name="get_latest_model_bundle_v1_model_bundles_latest_get"></a>
> ModelBundleV1Response get_latest_model_bundle_v1_model_bundles_latest_get(model_name)

Get Latest Model Bundle

Gets the latest Model Bundle with the given name owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_bundle_v1_response import ModelBundleV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'model_name': "model_name_example",
    }
    try:
        # Get Latest Model Bundle
        api_response = api_instance.get_latest_model_bundle_v1_model_bundles_latest_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_latest_model_bundle_v1_model_bundles_latest_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_name | ModelNameSchema | | 


# ModelNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_latest_model_bundle_v1_model_bundles_latest_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_latest_model_bundle_v1_model_bundles_latest_get.ApiResponseFor422) | Validation Error

#### get_latest_model_bundle_v1_model_bundles_latest_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelBundleV1Response**](../../models/ModelBundleV1Response.md) |  | 


#### get_latest_model_bundle_v1_model_bundles_latest_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_latest_model_bundle_v2_model_bundles_latest_get**
<a name="get_latest_model_bundle_v2_model_bundles_latest_get"></a>
> ModelBundleV2Response get_latest_model_bundle_v2_model_bundles_latest_get(model_name)

Get Latest Model Bundle

Gets the latest Model Bundle with the given name owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_bundle_v2_response import ModelBundleV2Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    query_params = {
        'model_name': "model_name_example",
    }
    try:
        # Get Latest Model Bundle
        api_response = api_instance.get_latest_model_bundle_v2_model_bundles_latest_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_latest_model_bundle_v2_model_bundles_latest_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_name | ModelNameSchema | | 


# ModelNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_latest_model_bundle_v2_model_bundles_latest_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_latest_model_bundle_v2_model_bundles_latest_get.ApiResponseFor422) | Validation Error

#### get_latest_model_bundle_v2_model_bundles_latest_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelBundleV2Response**](../../models/ModelBundleV2Response.md) |  | 


#### get_latest_model_bundle_v2_model_bundles_latest_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_model_bundle_v1_model_bundles_model_bundle_id_get**
<a name="get_model_bundle_v1_model_bundles_model_bundle_id_get"></a>
> ModelBundleV1Response get_model_bundle_v1_model_bundles_model_bundle_id_get(model_bundle_id)

Get Model Bundle

Gets the details for a given ModelBundle owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_bundle_v1_response import ModelBundleV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_bundle_id': "model_bundle_id_example",
    }
    try:
        # Get Model Bundle
        api_response = api_instance.get_model_bundle_v1_model_bundles_model_bundle_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_model_bundle_v1_model_bundles_model_bundle_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_bundle_id | ModelBundleIdSchema | | 

# ModelBundleIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_model_bundle_v1_model_bundles_model_bundle_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_model_bundle_v1_model_bundles_model_bundle_id_get.ApiResponseFor422) | Validation Error

#### get_model_bundle_v1_model_bundles_model_bundle_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelBundleV1Response**](../../models/ModelBundleV1Response.md) |  | 


#### get_model_bundle_v1_model_bundles_model_bundle_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_model_bundle_v2_model_bundles_model_bundle_id_get**
<a name="get_model_bundle_v2_model_bundles_model_bundle_id_get"></a>
> ModelBundleV2Response get_model_bundle_v2_model_bundles_model_bundle_id_get(model_bundle_id)

Get Model Bundle

Gets the details for a given ModelBundle owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_bundle_v2_response import ModelBundleV2Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_bundle_id': "model_bundle_id_example",
    }
    try:
        # Get Model Bundle
        api_response = api_instance.get_model_bundle_v2_model_bundles_model_bundle_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_model_bundle_v2_model_bundles_model_bundle_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_bundle_id | ModelBundleIdSchema | | 

# ModelBundleIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_model_bundle_v2_model_bundles_model_bundle_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_model_bundle_v2_model_bundles_model_bundle_id_get.ApiResponseFor422) | Validation Error

#### get_model_bundle_v2_model_bundles_model_bundle_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelBundleV2Response**](../../models/ModelBundleV2Response.md) |  | 


#### get_model_bundle_v2_model_bundles_model_bundle_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get**
<a name="get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get"></a>
> GetLLMModelEndpointV1Response get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get(model_endpoint_name)

Get Model Endpoint

Describe the LLM Model endpoint with given name.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.get_llm_model_endpoint_v1_response import GetLLMModelEndpointV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_endpoint_name': "model_endpoint_name_example",
    }
    try:
        # Get Model Endpoint
        api_response = api_instance.get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_name | ModelEndpointNameSchema | | 

# ModelEndpointNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get.ApiResponseFor422) | Validation Error

#### get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetLLMModelEndpointV1Response**](../../models/GetLLMModelEndpointV1Response.md) |  | 


#### get_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_model_endpoint_v1_model_endpoints_model_endpoint_id_get**
<a name="get_model_endpoint_v1_model_endpoints_model_endpoint_id_get"></a>
> GetModelEndpointV1Response get_model_endpoint_v1_model_endpoints_model_endpoint_id_get(model_endpoint_id)

Get Model Endpoint

Describe the Model endpoint with given ID.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.get_model_endpoint_v1_response import GetModelEndpointV1Response
from launch.api_client.model.http_validation_error import HTTPValidationError
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_endpoint_id': "model_endpoint_id_example",
    }
    try:
        # Get Model Endpoint
        api_response = api_instance.get_model_endpoint_v1_model_endpoints_model_endpoint_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_model_endpoint_v1_model_endpoints_model_endpoint_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_id | ModelEndpointIdSchema | | 

# ModelEndpointIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_model_endpoint_v1_model_endpoints_model_endpoint_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_model_endpoint_v1_model_endpoints_model_endpoint_id_get.ApiResponseFor422) | Validation Error

#### get_model_endpoint_v1_model_endpoints_model_endpoint_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetModelEndpointV1Response**](../../models/GetModelEndpointV1Response.md) |  | 


#### get_model_endpoint_v1_model_endpoints_model_endpoint_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_model_endpoints_api_v1_model_endpoints_api_get**
<a name="get_model_endpoints_api_v1_model_endpoints_api_get"></a>
> bool, date, datetime, dict, float, int, list, str, none_type get_model_endpoints_api_v1_model_endpoints_api_get()

Get Model Endpoints Api

Shows the API of the Model Endpoints owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Get Model Endpoints Api
        api_response = api_instance.get_model_endpoints_api_v1_model_endpoints_api_get()
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_model_endpoints_api_v1_model_endpoints_api_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_model_endpoints_api_v1_model_endpoints_api_get.ApiResponseFor200) | Successful Response

#### get_model_endpoints_api_v1_model_endpoints_api_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_model_endpoints_schema_v1_model_endpoints_schema_json_get**
<a name="get_model_endpoints_schema_v1_model_endpoints_schema_json_get"></a>
> bool, date, datetime, dict, float, int, list, str, none_type get_model_endpoints_schema_v1_model_endpoints_schema_json_get()

Get Model Endpoints Schema

Lists the schemas of the Model Endpoints owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Get Model Endpoints Schema
        api_response = api_instance.get_model_endpoints_schema_v1_model_endpoints_schema_json_get()
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_model_endpoints_schema_v1_model_endpoints_schema_json_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_model_endpoints_schema_v1_model_endpoints_schema_json_get.ApiResponseFor200) | Successful Response

#### get_model_endpoints_schema_v1_model_endpoints_schema_json_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **get_trigger_v1_triggers_trigger_id_get**
<a name="get_trigger_v1_triggers_trigger_id_get"></a>
> GetTriggerV1Response get_trigger_v1_triggers_trigger_id_get(trigger_id)

Get Trigger

Describes the trigger with the given ID

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.get_trigger_v1_response import GetTriggerV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'trigger_id': "trigger_id_example",
    }
    try:
        # Get Trigger
        api_response = api_instance.get_trigger_v1_triggers_trigger_id_get(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->get_trigger_v1_triggers_trigger_id_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
trigger_id | TriggerIdSchema | | 

# TriggerIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#get_trigger_v1_triggers_trigger_id_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#get_trigger_v1_triggers_trigger_id_get.ApiResponseFor422) | Validation Error

#### get_trigger_v1_triggers_trigger_id_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**GetTriggerV1Response**](../../models/GetTriggerV1Response.md) |  | 


#### get_trigger_v1_triggers_trigger_id_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **healthcheck_healthcheck_get**
<a name="healthcheck_healthcheck_get"></a>
> bool, date, datetime, dict, float, int, list, str, none_type healthcheck_healthcheck_get()

Healthcheck

Returns 200 if the app is healthy.

### Example

```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Healthcheck
        api_response = api_instance.healthcheck_healthcheck_get()
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->healthcheck_healthcheck_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#healthcheck_healthcheck_get.ApiResponseFor200) | Successful Response

#### healthcheck_healthcheck_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **healthcheck_healthz_get**
<a name="healthcheck_healthz_get"></a>
> bool, date, datetime, dict, float, int, list, str, none_type healthcheck_healthz_get()

Healthcheck

Returns 200 if the app is healthy.

### Example

```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Healthcheck
        api_response = api_instance.healthcheck_healthz_get()
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->healthcheck_healthz_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#healthcheck_healthz_get.ApiResponseFor200) | Successful Response

#### healthcheck_healthz_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **healthcheck_readyz_get**
<a name="healthcheck_readyz_get"></a>
> bool, date, datetime, dict, float, int, list, str, none_type healthcheck_readyz_get()

Healthcheck

Returns 200 if the app is healthy.

### Example

```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # Healthcheck
        api_response = api_instance.healthcheck_readyz_get()
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->healthcheck_readyz_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#healthcheck_readyz_get.ApiResponseFor200) | Successful Response

#### healthcheck_readyz_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
dict, frozendict.frozendict, str, date, datetime, uuid.UUID, int, float, decimal.Decimal, bool, None, list, tuple, bytes, io.FileIO, io.BufferedReader,  | frozendict.frozendict, str, decimal.Decimal, BoolClass, NoneClass, tuple, bytes, FileIO |  | 

### Authorization

No authorization required

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get**
<a name="list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get"></a>
> ListDockerImageBatchJobBundleV1Response list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get()

List Docker Image Batch Job Model Bundles

Lists docker image batch job bundles owned by current owner

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_bundle_order_by import ModelBundleOrderBy
from launch.api_client.model.list_docker_image_batch_job_bundle_v1_response import ListDockerImageBatchJobBundleV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only optional values
    query_params = {
        'bundle_name': "bundle_name_example",
        'order_by': ModelBundleOrderBy("newest"),
    }
    try:
        # List Docker Image Batch Job Model Bundles
        api_response = api_instance.list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
bundle_name | BundleNameSchema | | optional
order_by | OrderBySchema | | optional


# BundleNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
None, str,  | NoneClass, str,  |  | 

# OrderBySchema
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelBundleOrderBy**](../../models/ModelBundleOrderBy.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get.ApiResponseFor422) | Validation Error

#### list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListDockerImageBatchJobBundleV1Response**](../../models/ListDockerImageBatchJobBundleV1Response.md) |  | 


#### list_docker_image_batch_job_model_bundles_v1_docker_image_batch_job_bundles_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get**
<a name="list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get"></a>
> ListDockerImageBatchJobsV1Response list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get()

List Docker Image Batch Jobs

Lists docker image batch jobs spawned by trigger with given ID

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.list_docker_image_batch_jobs_v1_response import ListDockerImageBatchJobsV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only optional values
    query_params = {
        'trigger_id': "trigger_id_example",
    }
    try:
        # List Docker Image Batch Jobs
        api_response = api_instance.list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
trigger_id | TriggerIdSchema | | optional


# TriggerIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
None, str,  | NoneClass, str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get.ApiResponseFor422) | Validation Error

#### list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListDockerImageBatchJobsV1Response**](../../models/ListDockerImageBatchJobsV1Response.md) |  | 


#### list_docker_image_batch_jobs_v1_docker_image_batch_jobs_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_files_v1_files_get**
<a name="list_files_v1_files_get"></a>
> ListFilesResponse list_files_v1_files_get()

List Files

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.list_files_response import ListFilesResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # List Files
        api_response = api_instance.list_files_v1_files_get()
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_files_v1_files_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_files_v1_files_get.ApiResponseFor200) | Successful Response

#### list_files_v1_files_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListFilesResponse**](../../models/ListFilesResponse.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_fine_tunes_v1_llm_fine_tunes_get**
<a name="list_fine_tunes_v1_llm_fine_tunes_get"></a>
> ListFineTunesResponse list_fine_tunes_v1_llm_fine_tunes_get()

List Fine Tunes

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.list_fine_tunes_response import ListFineTunesResponse
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # List Fine Tunes
        api_response = api_instance.list_fine_tunes_v1_llm_fine_tunes_get()
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_fine_tunes_v1_llm_fine_tunes_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_fine_tunes_v1_llm_fine_tunes_get.ApiResponseFor200) | Successful Response

#### list_fine_tunes_v1_llm_fine_tunes_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListFineTunesResponse**](../../models/ListFineTunesResponse.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_model_bundles_v1_model_bundles_get**
<a name="list_model_bundles_v1_model_bundles_get"></a>
> ListModelBundlesV1Response list_model_bundles_v1_model_bundles_get()

List Model Bundles

Lists the ModelBundles owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.list_model_bundles_v1_response import ListModelBundlesV1Response
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_bundle_order_by import ModelBundleOrderBy
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only optional values
    query_params = {
        'model_name': "model_name_example",
        'order_by': ModelBundleOrderBy("newest"),
    }
    try:
        # List Model Bundles
        api_response = api_instance.list_model_bundles_v1_model_bundles_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_model_bundles_v1_model_bundles_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_name | ModelNameSchema | | optional
order_by | OrderBySchema | | optional


# ModelNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
None, str,  | NoneClass, str,  |  | 

# OrderBySchema
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelBundleOrderBy**](../../models/ModelBundleOrderBy.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_model_bundles_v1_model_bundles_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#list_model_bundles_v1_model_bundles_get.ApiResponseFor422) | Validation Error

#### list_model_bundles_v1_model_bundles_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListModelBundlesV1Response**](../../models/ListModelBundlesV1Response.md) |  | 


#### list_model_bundles_v1_model_bundles_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_model_bundles_v2_model_bundles_get**
<a name="list_model_bundles_v2_model_bundles_get"></a>
> ListModelBundlesV2Response list_model_bundles_v2_model_bundles_get()

List Model Bundles

Lists the ModelBundles owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.list_model_bundles_v2_response import ListModelBundlesV2Response
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_bundle_order_by import ModelBundleOrderBy
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only optional values
    query_params = {
        'model_name': "model_name_example",
        'order_by': ModelBundleOrderBy("newest"),
    }
    try:
        # List Model Bundles
        api_response = api_instance.list_model_bundles_v2_model_bundles_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_model_bundles_v2_model_bundles_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_name | ModelNameSchema | | optional
order_by | OrderBySchema | | optional


# ModelNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
None, str,  | NoneClass, str,  |  | 

# OrderBySchema
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelBundleOrderBy**](../../models/ModelBundleOrderBy.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_model_bundles_v2_model_bundles_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#list_model_bundles_v2_model_bundles_get.ApiResponseFor422) | Validation Error

#### list_model_bundles_v2_model_bundles_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListModelBundlesV2Response**](../../models/ListModelBundlesV2Response.md) |  | 


#### list_model_bundles_v2_model_bundles_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_model_endpoints_v1_llm_model_endpoints_get**
<a name="list_model_endpoints_v1_llm_model_endpoints_get"></a>
> ListLLMModelEndpointsV1Response list_model_endpoints_v1_llm_model_endpoints_get()

List Model Endpoints

Lists the LLM model endpoints owned by the current owner, plus all public_inference LLMs.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_endpoint_order_by import ModelEndpointOrderBy
from launch.api_client.model.list_llm_model_endpoints_v1_response import ListLLMModelEndpointsV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only optional values
    query_params = {
        'name': "name_example",
        'order_by': ModelEndpointOrderBy("newest"),
    }
    try:
        # List Model Endpoints
        api_response = api_instance.list_model_endpoints_v1_llm_model_endpoints_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_model_endpoints_v1_llm_model_endpoints_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
name | NameSchema | | optional
order_by | OrderBySchema | | optional


# NameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
None, str,  | NoneClass, str,  |  | 

# OrderBySchema
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelEndpointOrderBy**](../../models/ModelEndpointOrderBy.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_model_endpoints_v1_llm_model_endpoints_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#list_model_endpoints_v1_llm_model_endpoints_get.ApiResponseFor422) | Validation Error

#### list_model_endpoints_v1_llm_model_endpoints_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListLLMModelEndpointsV1Response**](../../models/ListLLMModelEndpointsV1Response.md) |  | 


#### list_model_endpoints_v1_llm_model_endpoints_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_model_endpoints_v1_model_endpoints_get**
<a name="list_model_endpoints_v1_model_endpoints_get"></a>
> ListModelEndpointsV1Response list_model_endpoints_v1_model_endpoints_get()

List Model Endpoints

Lists the Models owned by the current owner.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.list_model_endpoints_v1_response import ListModelEndpointsV1Response
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.model_endpoint_order_by import ModelEndpointOrderBy
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only optional values
    query_params = {
        'name': "name_example",
        'order_by': ModelEndpointOrderBy("newest"),
    }
    try:
        # List Model Endpoints
        api_response = api_instance.list_model_endpoints_v1_model_endpoints_get(
            query_params=query_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_model_endpoints_v1_model_endpoints_get: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
query_params | RequestQueryParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### query_params
#### RequestQueryParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
name | NameSchema | | optional
order_by | OrderBySchema | | optional


# NameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
None, str,  | NoneClass, str,  |  | 

# OrderBySchema
Type | Description  | Notes
------------- | ------------- | -------------
[**ModelEndpointOrderBy**](../../models/ModelEndpointOrderBy.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_model_endpoints_v1_model_endpoints_get.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#list_model_endpoints_v1_model_endpoints_get.ApiResponseFor422) | Validation Error

#### list_model_endpoints_v1_model_endpoints_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListModelEndpointsV1Response**](../../models/ListModelEndpointsV1Response.md) |  | 


#### list_model_endpoints_v1_model_endpoints_get.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **list_triggers_v1_triggers_get**
<a name="list_triggers_v1_triggers_get"></a>
> ListTriggersV1Response list_triggers_v1_triggers_get()

List Triggers

Lists descriptions of all triggers

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.list_triggers_v1_response import ListTriggersV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example, this endpoint has no required or optional parameters
    try:
        # List Triggers
        api_response = api_instance.list_triggers_v1_triggers_get()
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->list_triggers_v1_triggers_get: %s\n" % e)
```
### Parameters
This endpoint does not need any parameter.

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#list_triggers_v1_triggers_get.ApiResponseFor200) | Successful Response

#### list_triggers_v1_triggers_get.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**ListTriggersV1Response**](../../models/ListTriggersV1Response.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post**
<a name="restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post"></a>
> RestartModelEndpointV1Response restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post(model_endpoint_id)

Restart Model Endpoint

Restarts the Model endpoint.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.restart_model_endpoint_v1_response import RestartModelEndpointV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_endpoint_id': "model_endpoint_id_example",
    }
    try:
        # Restart Model Endpoint
        api_response = api_instance.restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post(
            path_params=path_params,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
path_params | RequestPathParams | |
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_id | ModelEndpointIdSchema | | 

# ModelEndpointIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post.ApiResponseFor422) | Validation Error

#### restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**RestartModelEndpointV1Response**](../../models/RestartModelEndpointV1Response.md) |  | 


#### restart_model_endpoint_v1_model_endpoints_model_endpoint_id_restart_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update_batch_completion_v2_batch_completions_batch_completion_id_post**
<a name="update_batch_completion_v2_batch_completions_batch_completion_id_post"></a>
> UpdateBatchCompletionsV2Response update_batch_completion_v2_batch_completions_batch_completion_id_post(batch_completion_idupdate_batch_completions_v2_request)

Update Batch Completion

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.update_batch_completions_v2_response import UpdateBatchCompletionsV2Response
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.update_batch_completions_v2_request import UpdateBatchCompletionsV2Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'batch_completion_id': "batch_completion_id_example",
    }
    body = UpdateBatchCompletionsV2Request(
        job_id="job_id_example",
        priority="priority_example",
    )
    try:
        # Update Batch Completion
        api_response = api_instance.update_batch_completion_v2_batch_completions_batch_completion_id_post(
            path_params=path_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->update_batch_completion_v2_batch_completions_batch_completion_id_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
path_params | RequestPathParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateBatchCompletionsV2Request**](../../models/UpdateBatchCompletionsV2Request.md) |  | 


### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
batch_completion_id | BatchCompletionIdSchema | | 

# BatchCompletionIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#update_batch_completion_v2_batch_completions_batch_completion_id_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#update_batch_completion_v2_batch_completions_batch_completion_id_post.ApiResponseFor422) | Validation Error

#### update_batch_completion_v2_batch_completions_batch_completion_id_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateBatchCompletionsV2Response**](../../models/UpdateBatchCompletionsV2Response.md) |  | 


#### update_batch_completion_v2_batch_completions_batch_completion_id_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update_batch_job_v1_batch_jobs_batch_job_id_put**
<a name="update_batch_job_v1_batch_jobs_batch_job_id_put"></a>
> UpdateBatchJobV1Response update_batch_job_v1_batch_jobs_batch_job_id_put(batch_job_idupdate_batch_job_v1_request)

Update Batch Job

Updates a batch job.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.update_batch_job_v1_request import UpdateBatchJobV1Request
from launch.api_client.model.update_batch_job_v1_response import UpdateBatchJobV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'batch_job_id': "batch_job_id_example",
    }
    body = UpdateBatchJobV1Request(
        cancel=True,
    )
    try:
        # Update Batch Job
        api_response = api_instance.update_batch_job_v1_batch_jobs_batch_job_id_put(
            path_params=path_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->update_batch_job_v1_batch_jobs_batch_job_id_put: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
path_params | RequestPathParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateBatchJobV1Request**](../../models/UpdateBatchJobV1Request.md) |  | 


### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
batch_job_id | BatchJobIdSchema | | 

# BatchJobIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#update_batch_job_v1_batch_jobs_batch_job_id_put.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#update_batch_job_v1_batch_jobs_batch_job_id_put.ApiResponseFor422) | Validation Error

#### update_batch_job_v1_batch_jobs_batch_job_id_put.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateBatchJobV1Response**](../../models/UpdateBatchJobV1Response.md) |  | 


#### update_batch_job_v1_batch_jobs_batch_job_id_put.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put**
<a name="update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put"></a>
> UpdateDockerImageBatchJobV1Response update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put(batch_job_idupdate_docker_image_batch_job_v1_request)

Update Docker Image Batch Job

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.update_docker_image_batch_job_v1_request import UpdateDockerImageBatchJobV1Request
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.update_docker_image_batch_job_v1_response import UpdateDockerImageBatchJobV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'batch_job_id': "batch_job_id_example",
    }
    body = UpdateDockerImageBatchJobV1Request(
        cancel=True,
    )
    try:
        # Update Docker Image Batch Job
        api_response = api_instance.update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put(
            path_params=path_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
path_params | RequestPathParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateDockerImageBatchJobV1Request**](../../models/UpdateDockerImageBatchJobV1Request.md) |  | 


### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
batch_job_id | BatchJobIdSchema | | 

# BatchJobIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put.ApiResponseFor422) | Validation Error

#### update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateDockerImageBatchJobV1Response**](../../models/UpdateDockerImageBatchJobV1Response.md) |  | 


#### update_docker_image_batch_job_v1_docker_image_batch_jobs_batch_job_id_put.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put**
<a name="update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put"></a>
> UpdateLLMModelEndpointV1Response update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put(model_endpoint_nameupdate_llm_model_endpoint_v1_request)

Update Model Endpoint

Updates an LLM endpoint for the current user.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.update_llm_model_endpoint_v1_response import UpdateLLMModelEndpointV1Response
from launch.api_client.model.update_llm_model_endpoint_v1_request import UpdateLLMModelEndpointV1Request
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_endpoint_name': "model_endpoint_name_example",
    }
    body = UpdateLLMModelEndpointV1Request(None)
    try:
        # Update Model Endpoint
        api_response = api_instance.update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put(
            path_params=path_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
path_params | RequestPathParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateLLMModelEndpointV1Request**](../../models/UpdateLLMModelEndpointV1Request.md) |  | 


### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_name | ModelEndpointNameSchema | | 

# ModelEndpointNameSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put.ApiResponseFor422) | Validation Error

#### update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateLLMModelEndpointV1Response**](../../models/UpdateLLMModelEndpointV1Response.md) |  | 


#### update_model_endpoint_v1_llm_model_endpoints_model_endpoint_name_put.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update_model_endpoint_v1_model_endpoints_model_endpoint_id_put**
<a name="update_model_endpoint_v1_model_endpoints_model_endpoint_id_put"></a>
> UpdateModelEndpointV1Response update_model_endpoint_v1_model_endpoints_model_endpoint_id_put(model_endpoint_idupdate_model_endpoint_v1_request)

Update Model Endpoint

Updates the Model endpoint.

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.update_model_endpoint_v1_request import UpdateModelEndpointV1Request
from launch.api_client.model.update_model_endpoint_v1_response import UpdateModelEndpointV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'model_endpoint_id': "model_endpoint_id_example",
    }
    body = UpdateModelEndpointV1Request(
        model_bundle_id="model_bundle_id_example",
        metadata=dict(),
        post_inference_hooks=[
            "post_inference_hooks_example"
        ],
        cpus=None,
        gpus=0.0,
        memory=None,
        gpu_type=GpuType("nvidia-tesla-t4"),
        storage=None,
        optimize_costs=True,
        min_workers=0.0,
        max_workers=0.0,
        per_worker=1,
        concurrent_requests_per_worker=1,
        labels=dict(
            "key": "key_example",
        ),
        prewarm=True,
        high_priority=True,
        billing_tags=dict(),
        default_callback_url="default_callback_url_example",
        default_callback_auth=CallbackAuth(
            kind="CallbackBasicAuth",
            username="username_example",
            password="password_example",
        ),
        public_inference=True,
    )
    try:
        # Update Model Endpoint
        api_response = api_instance.update_model_endpoint_v1_model_endpoints_model_endpoint_id_put(
            path_params=path_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->update_model_endpoint_v1_model_endpoints_model_endpoint_id_put: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
path_params | RequestPathParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateModelEndpointV1Request**](../../models/UpdateModelEndpointV1Request.md) |  | 


### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
model_endpoint_id | ModelEndpointIdSchema | | 

# ModelEndpointIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#update_model_endpoint_v1_model_endpoints_model_endpoint_id_put.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#update_model_endpoint_v1_model_endpoints_model_endpoint_id_put.ApiResponseFor422) | Validation Error

#### update_model_endpoint_v1_model_endpoints_model_endpoint_id_put.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateModelEndpointV1Response**](../../models/UpdateModelEndpointV1Response.md) |  | 


#### update_model_endpoint_v1_model_endpoints_model_endpoint_id_put.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **update_trigger_v1_triggers_trigger_id_put**
<a name="update_trigger_v1_triggers_trigger_id_put"></a>
> UpdateTriggerV1Response update_trigger_v1_triggers_trigger_id_put(trigger_idupdate_trigger_v1_request)

Update Trigger

Updates the trigger with the given ID

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.update_trigger_v1_request import UpdateTriggerV1Request
from launch.api_client.model.update_trigger_v1_response import UpdateTriggerV1Response
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only required values which don't have defaults set
    path_params = {
        'trigger_id': "trigger_id_example",
    }
    body = UpdateTriggerV1Request(
        cron_schedule="cron_schedule_example",
        suspend=True,
    )
    try:
        # Update Trigger
        api_response = api_instance.update_trigger_v1_triggers_trigger_id_put(
            path_params=path_params,
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->update_trigger_v1_triggers_trigger_id_put: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyApplicationJson] | required |
path_params | RequestPathParams | |
content_type | str | optional, default is 'application/json' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateTriggerV1Request**](../../models/UpdateTriggerV1Request.md) |  | 


### path_params
#### RequestPathParams

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
trigger_id | TriggerIdSchema | | 

# TriggerIdSchema

## Model Type Info
Input Type | Accessed Type | Description | Notes
------------ | ------------- | ------------- | -------------
str,  | str,  |  | 

### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#update_trigger_v1_triggers_trigger_id_put.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#update_trigger_v1_triggers_trigger_id_put.ApiResponseFor422) | Validation Error

#### update_trigger_v1_triggers_trigger_id_put.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UpdateTriggerV1Response**](../../models/UpdateTriggerV1Response.md) |  | 


#### update_trigger_v1_triggers_trigger_id_put.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

# **upload_file_v1_files_post**
<a name="upload_file_v1_files_post"></a>
> UploadFileResponse upload_file_v1_files_post()

Upload File

### Example

* OAuth Authentication (OAuth2PasswordBearer):
* Basic Authentication (HTTPBasic):
```python
import launch.api_client
from launch.api_client.apis.tags import default_api
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.upload_file_response import UploadFileResponse
from launch.api_client.model.body_upload_file_v1_files_post import BodyUploadFileV1FilesPost
from pprint import pprint
# Defining the host is optional and defaults to http://localhost
# See configuration.py for a list of all supported configuration parameters.
configuration = launch.api_client.Configuration(
    host = "http://localhost"
)

# The client must configure the authentication and authorization parameters
# in accordance with the API server security policy.
# Examples for each auth method are provided below, use the example that
# satisfies your auth use case.

# Configure OAuth2 access token for authorization: OAuth2PasswordBearer
configuration = launch.api_client.Configuration(
    host = "http://localhost",
    access_token = 'YOUR_ACCESS_TOKEN'
)

# Configure HTTP basic authorization: HTTPBasic
configuration = launch.api_client.Configuration(
    username = 'YOUR_USERNAME',
    password = 'YOUR_PASSWORD'
)
# Enter a context with an instance of the API client
with launch.api_client.ApiClient(configuration) as api_client:
    # Create an instance of the API class
    api_instance = default_api.DefaultApi(api_client)

    # example passing only optional values
    body = dict(
        file=open('/path/to/file', 'rb'),
    )
    try:
        # Upload File
        api_response = api_instance.upload_file_v1_files_post(
            body=body,
        )
        pprint(api_response)
    except launch.api_client.ApiException as e:
        print("Exception when calling DefaultApi->upload_file_v1_files_post: %s\n" % e)
```
### Parameters

Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
body | typing.Union[SchemaForRequestBodyMultipartFormData, Unset] | optional, default is unset |
content_type | str | optional, default is 'multipart/form-data' | Selects the schema and serialization of the request body
accept_content_types | typing.Tuple[str] | default is ('application/json', ) | Tells the server the content type(s) that are accepted by the client
stream | bool | default is False | if True then the response.content will be streamed and loaded from a file like object. When downloading a file, set this to True to force the code to deserialize the content to a FileSchema file
timeout | typing.Optional[typing.Union[int, typing.Tuple]] | default is None | the timeout used by the rest client
skip_deserialization | bool | default is False | when True, headers and body will be unset and an instance of api_client.ApiResponseWithoutDeserialization will be returned

### body

# SchemaForRequestBodyMultipartFormData
Type | Description  | Notes
------------- | ------------- | -------------
[**BodyUploadFileV1FilesPost**](../../models/BodyUploadFileV1FilesPost.md) |  | 


### Return Types, Responses

Code | Class | Description
------------- | ------------- | -------------
n/a | api_client.ApiResponseWithoutDeserialization | When skip_deserialization is True this response is returned
200 | [ApiResponseFor200](#upload_file_v1_files_post.ApiResponseFor200) | Successful Response
422 | [ApiResponseFor422](#upload_file_v1_files_post.ApiResponseFor422) | Validation Error

#### upload_file_v1_files_post.ApiResponseFor200
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor200ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor200ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**UploadFileResponse**](../../models/UploadFileResponse.md) |  | 


#### upload_file_v1_files_post.ApiResponseFor422
Name | Type | Description  | Notes
------------- | ------------- | ------------- | -------------
response | urllib3.HTTPResponse | Raw response |
body | typing.Union[SchemaFor422ResponseBodyApplicationJson, ] |  |
headers | Unset | headers were not defined |

# SchemaFor422ResponseBodyApplicationJson
Type | Description  | Notes
------------- | ------------- | -------------
[**HTTPValidationError**](../../models/HTTPValidationError.md) |  | 


### Authorization

[OAuth2PasswordBearer](../../../README.md#OAuth2PasswordBearer), [HTTPBasic](../../../README.md#HTTPBasic)

[[Back to top]](#__pageTop) [[Back to API list]](../../../README.md#documentation-for-api-endpoints) [[Back to Model list]](../../../README.md#documentation-for-models) [[Back to README]](../../../README.md)

