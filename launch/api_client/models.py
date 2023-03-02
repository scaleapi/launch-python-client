from datetime import datetime
from enum import Enum
from typing import Any  # noqa
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class BatchJobSerializationFormat(str, Enum):
    JSON = "JSON"
    PICKLE = "PICKLE"


class BatchJobStatus(str, Enum):
    PENDING = "PENDING"
    CREATING_ENDPOINT = "CREATING_ENDPOINT"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    CANCELLED = "CANCELLED"
    UNDEFINED = "UNDEFINED"


class CloneModelBundleRequest(BaseModel):
    new_app_config: "Optional[Any]" = Field(None, alias="new_app_config")
    original_model_bundle_id: "str" = Field(..., alias="original_model_bundle_id")


class CreateAsyncTaskResponse(BaseModel):
    task_id: "str" = Field(..., alias="task_id")


class CreateBatchJobRequest(BaseModel):
    input_path: "str" = Field(..., alias="input_path")
    labels: "Dict[str, str]" = Field(..., alias="labels")
    model_bundle_id: "str" = Field(..., alias="model_bundle_id")
    resource_requests: "CreateBatchJobResourceRequests" = Field(..., alias="resource_requests")
    serialization_format: "BatchJobSerializationFormat" = Field(..., alias="serialization_format")


class CreateBatchJobResourceRequests(BaseModel):
    cpus: "Optional[Any]" = Field(None, alias="cpus")
    gpu_type: "Optional[GpuType]" = Field(None, alias="gpu_type")
    gpus: "Optional[int]" = Field(None, alias="gpus")
    max_workers: "Optional[int]" = Field(None, alias="max_workers")
    memory: "Optional[Any]" = Field(None, alias="memory")
    per_worker: "Optional[int]" = Field(None, alias="per_worker")
    storage: "Optional[Any]" = Field(None, alias="storage")


class CreateBatchJobResponse(BaseModel):
    job_id: "str" = Field(..., alias="job_id")


class CreateModelBundleRequest(BaseModel):
    app_config: "Optional[Any]" = Field(None, alias="app_config")
    env_params: "ModelBundleEnvironmentParams" = Field(..., alias="env_params")
    location: "str" = Field(..., alias="location")
    metadata: "Optional[Any]" = Field(None, alias="metadata")
    name: "str" = Field(..., alias="name")
    packaging_type: "Optional[ModelBundlePackagingType]" = Field(None, alias="packaging_type")
    requirements: "List[str]" = Field(..., alias="requirements")
    schema_location: "Optional[str]" = Field(None, alias="schema_location")


class CreateModelBundleResponse(BaseModel):
    model_bundle_id: "str" = Field(..., alias="model_bundle_id")


class CreateModelEndpointRequest(BaseModel):
    billing_tags: "Optional[Any]" = Field(None, alias="billing_tags")
    cpus: "Any" = Field(..., alias="cpus")
    default_callback_url: "Optional[str]" = Field(None, alias="default_callback_url")
    endpoint_type: "ModelEndpointType" = Field(..., alias="endpoint_type")
    gpu_type: "Optional[GpuType]" = Field(None, alias="gpu_type")
    gpus: "int" = Field(..., alias="gpus")
    labels: "Dict[str, str]" = Field(..., alias="labels")
    max_workers: "int" = Field(..., alias="max_workers")
    memory: "Any" = Field(..., alias="memory")
    metadata: "Any" = Field(..., alias="metadata")
    min_workers: "int" = Field(..., alias="min_workers")
    model_bundle_id: "str" = Field(..., alias="model_bundle_id")
    name: "str" = Field(..., alias="name")
    optimize_costs: "Optional[bool]" = Field(None, alias="optimize_costs")
    per_worker: "int" = Field(..., alias="per_worker")
    post_inference_hooks: "Optional[List[str]]" = Field(None, alias="post_inference_hooks")
    prewarm: "Optional[bool]" = Field(None, alias="prewarm")
    storage: "Optional[Any]" = Field(None, alias="storage")


class CreateModelEndpointResponse(BaseModel):
    endpoint_creation_task_id: "str" = Field(..., alias="endpoint_creation_task_id")


class DeleteModelEndpointResponse(BaseModel):
    deleted: "bool" = Field(..., alias="deleted")


class EndpointPredictRequest(BaseModel):
    args: "Optional[Any]" = Field(None, alias="args")
    callback_url: "Optional[str]" = Field(None, alias="callback_url")
    cloudpickle: "Optional[str]" = Field(None, alias="cloudpickle")
    return_pickled: "Optional[bool]" = Field(None, alias="return_pickled")
    url: "Optional[str]" = Field(None, alias="url")


class GetAsyncTaskResponse(BaseModel):
    result: "Optional[Any]" = Field(None, alias="result")
    status: "TaskStatus" = Field(..., alias="status")
    task_id: "str" = Field(..., alias="task_id")
    traceback: "Optional[str]" = Field(None, alias="traceback")


class GetBatchJobResponse(BaseModel):
    duration: "Optional[float]" = Field(None, alias="duration")
    num_tasks_completed: "Optional[int]" = Field(None, alias="num_tasks_completed")
    num_tasks_pending: "Optional[int]" = Field(None, alias="num_tasks_pending")
    result: "Optional[str]" = Field(None, alias="result")
    status: "BatchJobStatus" = Field(..., alias="status")


class GetModelEndpointResponse(BaseModel):
    aws_role: "Optional[str]" = Field(None, alias="aws_role")
    bundle_name: "str" = Field(..., alias="bundle_name")
    created_at: "datetime" = Field(..., alias="created_at")
    created_by: "str" = Field(..., alias="created_by")
    default_callback_url: "Optional[str]" = Field(None, alias="default_callback_url")
    deployment_name: "Optional[str]" = Field(None, alias="deployment_name")
    deployment_state: "Optional[ModelEndpointDeploymentState]" = Field(None, alias="deployment_state")
    destination: "str" = Field(..., alias="destination")
    endpoint_type: "ModelEndpointType" = Field(..., alias="endpoint_type")
    id: "str" = Field(..., alias="id")
    labels: "Optional[Dict[str, str]]" = Field(None, alias="labels")
    last_updated_at: "datetime" = Field(..., alias="last_updated_at")
    metadata: "Optional[Any]" = Field(None, alias="metadata")
    name: "str" = Field(..., alias="name")
    post_inference_hooks: "Optional[List[str]]" = Field(None, alias="post_inference_hooks")
    resource_state: "Optional[ModelEndpointResourceState]" = Field(None, alias="resource_state")
    results_s3_bucket: "Optional[str]" = Field(None, alias="results_s3_bucket")
    status: "ModelEndpointStatus" = Field(..., alias="status")


class GpuType(str, Enum):
    TESLA_T4 = "nvidia-tesla-t4"
    AMPERE_A10 = "nvidia-ampere-a10"
    A100 = "nvidia-a100"


class HTTPValidationError(BaseModel):
    detail: "Optional[List[ValidationError]]" = Field(None, alias="detail")


class ListModelBundlesResponse(BaseModel):
    model_bundles: "List[ModelBundleResponse]" = Field(..., alias="model_bundles")


class ListModelEndpointsResponse(BaseModel):
    model_endpoints: "List[GetModelEndpointResponse]" = Field(..., alias="model_endpoints")


class ModelBundleEnvironmentParams(BaseModel):
    ecr_repo: "Optional[str]" = Field(None, alias="ecr_repo")
    framework_type: "ModelBundleFramework" = Field(..., alias="framework_type")
    image_tag: "Optional[str]" = Field(None, alias="image_tag")
    pytorch_image_tag: "Optional[str]" = Field(None, alias="pytorch_image_tag")
    tensorflow_version: "Optional[str]" = Field(None, alias="tensorflow_version")


class ModelBundleFramework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    CUSTOM_BASE_IMAGE = "custom_base_image"


class ModelBundleOrderBy(str, Enum):
    NEWEST = "newest"
    OLDEST = "oldest"


class ModelBundlePackagingType(str, Enum):
    CLOUDPICKLE = "cloudpickle"
    ZIP = "zip"


class ModelBundleResponse(BaseModel):
    app_config: "Optional[Any]" = Field(None, alias="app_config")
    created_at: "datetime" = Field(..., alias="created_at")
    env_params: "ModelBundleEnvironmentParams" = Field(..., alias="env_params")
    id: "str" = Field(..., alias="id")
    location: "str" = Field(..., alias="location")
    metadata: "Any" = Field(..., alias="metadata")
    model_artifact_ids: "List[str]" = Field(..., alias="model_artifact_ids")
    name: "str" = Field(..., alias="name")
    packaging_type: "ModelBundlePackagingType" = Field(..., alias="packaging_type")
    requirements: "List[str]" = Field(..., alias="requirements")
    schema_location: "Optional[str]" = Field(None, alias="schema_location")


class ModelEndpointDeploymentState(BaseModel):
    available_workers: "Optional[int]" = Field(None, alias="available_workers")
    max_workers: "int" = Field(..., alias="max_workers")
    min_workers: "int" = Field(..., alias="min_workers")
    per_worker: "int" = Field(..., alias="per_worker")
    unavailable_workers: "Optional[int]" = Field(None, alias="unavailable_workers")


class ModelEndpointOrderBy(str, Enum):
    NEWEST = "newest"
    OLDEST = "oldest"
    ALPHABETICAL = "alphabetical"


class ModelEndpointResourceState(BaseModel):
    cpus: "Any" = Field(..., alias="cpus")
    gpu_type: "Optional[GpuType]" = Field(None, alias="gpu_type")
    gpus: "int" = Field(..., alias="gpus")
    memory: "Any" = Field(..., alias="memory")
    optimize_costs: "Optional[bool]" = Field(None, alias="optimize_costs")
    storage: "Optional[Any]" = Field(None, alias="storage")


class ModelEndpointStatus(str, Enum):
    READY = "READY"
    UPDATE_PENDING = "UPDATE_PENDING"
    UPDATE_IN_PROGRESS = "UPDATE_IN_PROGRESS"
    UPDATE_FAILED = "UPDATE_FAILED"
    DELETE_IN_PROGRESS = "DELETE_IN_PROGRESS"


class ModelEndpointType(str, Enum):
    ASYNC = "async"
    SYNC = "sync"


class SyncEndpointPredictResponse(BaseModel):
    result: "Optional[Any]" = Field(None, alias="result")
    status: "TaskStatus" = Field(..., alias="status")
    traceback: "Optional[str]" = Field(None, alias="traceback")


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNDEFINED = "UNDEFINED"


class UpdateBatchJobRequest(BaseModel):
    cancel: "bool" = Field(..., alias="cancel")


class UpdateBatchJobResponse(BaseModel):
    success: "bool" = Field(..., alias="success")


class UpdateModelEndpointRequest(BaseModel):
    aws_role: "Optional[str]" = Field(None, alias="aws_role")
    billing_tags: "Optional[Any]" = Field(None, alias="billing_tags")
    cpus: "Optional[Any]" = Field(None, alias="cpus")
    default_callback_url: "Optional[str]" = Field(None, alias="default_callback_url")
    gpu_type: "Optional[GpuType]" = Field(None, alias="gpu_type")
    gpus: "Optional[int]" = Field(None, alias="gpus")
    labels: "Optional[Dict[str, str]]" = Field(None, alias="labels")
    max_workers: "Optional[int]" = Field(None, alias="max_workers")
    memory: "Optional[Any]" = Field(None, alias="memory")
    metadata: "Optional[Any]" = Field(None, alias="metadata")
    min_workers: "Optional[int]" = Field(None, alias="min_workers")
    model_bundle_id: "Optional[str]" = Field(None, alias="model_bundle_id")
    optimize_costs: "Optional[bool]" = Field(None, alias="optimize_costs")
    per_worker: "Optional[int]" = Field(None, alias="per_worker")
    post_inference_hooks: "Optional[List[str]]" = Field(None, alias="post_inference_hooks")
    prewarm: "Optional[bool]" = Field(None, alias="prewarm")
    results_s3_bucket: "Optional[str]" = Field(None, alias="results_s3_bucket")
    storage: "Optional[Any]" = Field(None, alias="storage")


class UpdateModelEndpointResponse(BaseModel):
    endpoint_creation_task_id: "str" = Field(..., alias="endpoint_creation_task_id")


class ValidationError(BaseModel):
    loc: "List[Any]" = Field(..., alias="loc")
    msg: "str" = Field(..., alias="msg")
    type: "str" = Field(..., alias="type")
