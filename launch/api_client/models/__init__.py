# coding: utf-8

# flake8: noqa

# import all models into this package
# if you have many models here with many references from one model to another this may
# raise a RecursionError
# to avoid this, import only the models that you directly need like:
# from launch.api_client.model.pet import Pet
# or import this package, but before doing it, use:
# import sys
# sys.setrecursionlimit(n)

from launch.api_client.model.batch_job_serialization_format import (
    BatchJobSerializationFormat,
)
from launch.api_client.model.batch_job_status import BatchJobStatus
from launch.api_client.model.body_upload_file_v1_files_post import (
    BodyUploadFileV1FilesPost,
)
from launch.api_client.model.callback_auth import CallbackAuth
from launch.api_client.model.callback_basic_auth import CallbackBasicAuth
from launch.api_client.model.callbackm_tls_auth import CallbackmTLSAuth
from launch.api_client.model.cancel_fine_tune_response import (
    CancelFineTuneResponse,
)
from launch.api_client.model.clone_model_bundle_v1_request import (
    CloneModelBundleV1Request,
)
from launch.api_client.model.clone_model_bundle_v2_request import (
    CloneModelBundleV2Request,
)
from launch.api_client.model.cloudpickle_artifact_flavor import (
    CloudpickleArtifactFlavor,
)
from launch.api_client.model.completion_output import CompletionOutput
from launch.api_client.model.completion_stream_output import (
    CompletionStreamOutput,
)
from launch.api_client.model.completion_stream_v1_request import (
    CompletionStreamV1Request,
)
from launch.api_client.model.completion_stream_v1_response import (
    CompletionStreamV1Response,
)
from launch.api_client.model.completion_sync_v1_request import (
    CompletionSyncV1Request,
)
from launch.api_client.model.completion_sync_v1_response import (
    CompletionSyncV1Response,
)
from launch.api_client.model.create_async_task_v1_response import (
    CreateAsyncTaskV1Response,
)
from launch.api_client.model.create_batch_job_resource_requests import (
    CreateBatchJobResourceRequests,
)
from launch.api_client.model.create_batch_job_v1_request import (
    CreateBatchJobV1Request,
)
from launch.api_client.model.create_batch_job_v1_response import (
    CreateBatchJobV1Response,
)
from launch.api_client.model.create_docker_image_batch_job_bundle_v1_request import (
    CreateDockerImageBatchJobBundleV1Request,
)
from launch.api_client.model.create_docker_image_batch_job_bundle_v1_response import (
    CreateDockerImageBatchJobBundleV1Response,
)
from launch.api_client.model.create_docker_image_batch_job_resource_requests import (
    CreateDockerImageBatchJobResourceRequests,
)
from launch.api_client.model.create_docker_image_batch_job_v1_request import (
    CreateDockerImageBatchJobV1Request,
)
from launch.api_client.model.create_docker_image_batch_job_v1_response import (
    CreateDockerImageBatchJobV1Response,
)
from launch.api_client.model.create_fine_tune_request import (
    CreateFineTuneRequest,
)
from launch.api_client.model.create_fine_tune_response import (
    CreateFineTuneResponse,
)
from launch.api_client.model.create_llm_model_endpoint_v1_request import (
    CreateLLMModelEndpointV1Request,
)
from launch.api_client.model.create_llm_model_endpoint_v1_response import (
    CreateLLMModelEndpointV1Response,
)
from launch.api_client.model.create_model_bundle_v1_request import (
    CreateModelBundleV1Request,
)
from launch.api_client.model.create_model_bundle_v1_response import (
    CreateModelBundleV1Response,
)
from launch.api_client.model.create_model_bundle_v2_request import (
    CreateModelBundleV2Request,
)
from launch.api_client.model.create_model_bundle_v2_response import (
    CreateModelBundleV2Response,
)
from launch.api_client.model.create_model_endpoint_v1_request import (
    CreateModelEndpointV1Request,
)
from launch.api_client.model.create_model_endpoint_v1_response import (
    CreateModelEndpointV1Response,
)
from launch.api_client.model.create_trigger_v1_request import (
    CreateTriggerV1Request,
)
from launch.api_client.model.create_trigger_v1_response import (
    CreateTriggerV1Response,
)
from launch.api_client.model.custom_framework import CustomFramework
from launch.api_client.model.delete_file_response import DeleteFileResponse
from launch.api_client.model.delete_model_endpoint_v1_response import (
    DeleteModelEndpointV1Response,
)
from launch.api_client.model.delete_trigger_v1_response import (
    DeleteTriggerV1Response,
)
from launch.api_client.model.docker_image_batch_job import DockerImageBatchJob
from launch.api_client.model.docker_image_batch_job_bundle_v1_response import (
    DockerImageBatchJobBundleV1Response,
)
from launch.api_client.model.endpoint_predict_v1_request import (
    EndpointPredictV1Request,
)
from launch.api_client.model.get_async_task_v1_response import (
    GetAsyncTaskV1Response,
)
from launch.api_client.model.get_batch_job_v1_response import (
    GetBatchJobV1Response,
)
from launch.api_client.model.get_docker_image_batch_job_v1_response import (
    GetDockerImageBatchJobV1Response,
)
from launch.api_client.model.get_file_content_response import (
    GetFileContentResponse,
)
from launch.api_client.model.get_file_response import GetFileResponse
from launch.api_client.model.get_fine_tune_events_response import (
    GetFineTuneEventsResponse,
)
from launch.api_client.model.get_fine_tune_response import GetFineTuneResponse
from launch.api_client.model.get_llm_model_endpoint_v1_response import (
    GetLLMModelEndpointV1Response,
)
from launch.api_client.model.get_model_endpoint_v1_response import (
    GetModelEndpointV1Response,
)
from launch.api_client.model.get_trigger_v1_response import (
    GetTriggerV1Response,
)
from launch.api_client.model.gpu_type import GpuType
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.list_docker_image_batch_job_bundle_v1_response import (
    ListDockerImageBatchJobBundleV1Response,
)
from launch.api_client.model.list_docker_image_batch_jobs_v1_response import (
    ListDockerImageBatchJobsV1Response,
)
from launch.api_client.model.list_files_response import ListFilesResponse
from launch.api_client.model.list_fine_tunes_response import (
    ListFineTunesResponse,
)
from launch.api_client.model.list_llm_model_endpoints_v1_response import (
    ListLLMModelEndpointsV1Response,
)
from launch.api_client.model.list_model_bundles_v1_response import (
    ListModelBundlesV1Response,
)
from launch.api_client.model.list_model_bundles_v2_response import (
    ListModelBundlesV2Response,
)
from launch.api_client.model.list_model_endpoints_v1_response import (
    ListModelEndpointsV1Response,
)
from launch.api_client.model.list_triggers_v1_response import (
    ListTriggersV1Response,
)
from launch.api_client.model.llm_fine_tune_event import LLMFineTuneEvent
from launch.api_client.model.llm_inference_framework import (
    LLMInferenceFramework,
)
from launch.api_client.model.llm_source import LLMSource
from launch.api_client.model.model_bundle_environment_params import (
    ModelBundleEnvironmentParams,
)
from launch.api_client.model.model_bundle_framework_type import (
    ModelBundleFrameworkType,
)
from launch.api_client.model.model_bundle_order_by import ModelBundleOrderBy
from launch.api_client.model.model_bundle_packaging_type import (
    ModelBundlePackagingType,
)
from launch.api_client.model.model_bundle_v1_response import (
    ModelBundleV1Response,
)
from launch.api_client.model.model_bundle_v2_response import (
    ModelBundleV2Response,
)
from launch.api_client.model.model_download_request import ModelDownloadRequest
from launch.api_client.model.model_download_response import (
    ModelDownloadResponse,
)
from launch.api_client.model.model_endpoint_deployment_state import (
    ModelEndpointDeploymentState,
)
from launch.api_client.model.model_endpoint_order_by import (
    ModelEndpointOrderBy,
)
from launch.api_client.model.model_endpoint_resource_state import (
    ModelEndpointResourceState,
)
from launch.api_client.model.model_endpoint_status import ModelEndpointStatus
from launch.api_client.model.model_endpoint_type import ModelEndpointType
from launch.api_client.model.pytorch_framework import PytorchFramework
from launch.api_client.model.quantization import Quantization
from launch.api_client.model.request_schema import RequestSchema
from launch.api_client.model.response_schema import ResponseSchema
from launch.api_client.model.runnable_image_flavor import RunnableImageFlavor
from launch.api_client.model.streaming_enhanced_runnable_image_flavor import (
    StreamingEnhancedRunnableImageFlavor,
)
from launch.api_client.model.sync_endpoint_predict_v1_response import (
    SyncEndpointPredictV1Response,
)
from launch.api_client.model.task_status import TaskStatus
from launch.api_client.model.tensorflow_framework import TensorflowFramework
from launch.api_client.model.token_output import TokenOutput
from launch.api_client.model.triton_enhanced_runnable_image_flavor import (
    TritonEnhancedRunnableImageFlavor,
)
from launch.api_client.model.update_batch_job_v1_request import (
    UpdateBatchJobV1Request,
)
from launch.api_client.model.update_batch_job_v1_response import (
    UpdateBatchJobV1Response,
)
from launch.api_client.model.update_docker_image_batch_job_v1_request import (
    UpdateDockerImageBatchJobV1Request,
)
from launch.api_client.model.update_docker_image_batch_job_v1_response import (
    UpdateDockerImageBatchJobV1Response,
)
from launch.api_client.model.update_model_endpoint_v1_request import (
    UpdateModelEndpointV1Request,
)
from launch.api_client.model.update_model_endpoint_v1_response import (
    UpdateModelEndpointV1Response,
)
from launch.api_client.model.update_trigger_v1_request import (
    UpdateTriggerV1Request,
)
from launch.api_client.model.update_trigger_v1_response import (
    UpdateTriggerV1Response,
)
from launch.api_client.model.upload_file_response import UploadFileResponse
from launch.api_client.model.validation_error import ValidationError
from launch.api_client.model.zip_artifact_flavor import ZipArtifactFlavor
