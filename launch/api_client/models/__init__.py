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
from launch.api_client.model.callback_auth import CallbackAuth
from launch.api_client.model.callback_basic_auth import CallbackBasicAuth
from launch.api_client.model.callbackm_tls_auth import CallbackmTLSAuth
from launch.api_client.model.clone_model_bundle_v1_request import (
    CloneModelBundleV1Request,
)
from launch.api_client.model.clone_model_bundle_v2_request import (
    CloneModelBundleV2Request,
)
from launch.api_client.model.cloudpickle_artifact_flavor import (
    CloudpickleArtifactFlavor,
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
from launch.api_client.model.custom_framework import CustomFramework
from launch.api_client.model.delete_model_endpoint_v1_response import (
    DeleteModelEndpointV1Response,
)
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
from launch.api_client.model.get_model_endpoint_v1_response import (
    GetModelEndpointV1Response,
)
from launch.api_client.model.gpu_type import GpuType
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.list_docker_image_batch_job_bundle_v1_response import (
    ListDockerImageBatchJobBundleV1Response,
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
from launch.api_client.model.validation_error import ValidationError
from launch.api_client.model.zip_artifact_flavor import ZipArtifactFlavor
