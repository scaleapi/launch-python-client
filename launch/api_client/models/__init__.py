# flake8: noqa

# import all models into this package
# if you have many models here with many references from one model to another this may
# raise a RecursionError
# to avoid this, import only the models that you directly need like:
# from from launch.api_client.model.pet import Pet
# or import this package, but before doing it, use:
# import sys
# sys.setrecursionlimit(n)

from launch.api_client.model.cpus import Cpus
from launch.api_client.model.create_async_task_response import (
    CreateAsyncTaskResponse,
)
from launch.api_client.model.create_model_bundle_request import (
    CreateModelBundleRequest,
)
from launch.api_client.model.create_model_bundle_response import (
    CreateModelBundleResponse,
)
from launch.api_client.model.create_model_endpoint_request import (
    CreateModelEndpointRequest,
)
from launch.api_client.model.create_model_endpoint_response import (
    CreateModelEndpointResponse,
)
from launch.api_client.model.delete_model_endpoint_response import (
    DeleteModelEndpointResponse,
)
from launch.api_client.model.endpoint_predict_request import (
    EndpointPredictRequest,
)
from launch.api_client.model.get_async_task_response import (
    GetAsyncTaskResponse,
)
from launch.api_client.model.get_model_endpoint_response import (
    GetModelEndpointResponse,
)
from launch.api_client.model.gpu_type import GpuType
from launch.api_client.model.http_validation_error import HTTPValidationError
from launch.api_client.model.list_model_bundles_response import (
    ListModelBundlesResponse,
)
from launch.api_client.model.list_model_endpoints_response import (
    ListModelEndpointsResponse,
)
from launch.api_client.model.location_inner import LocationInner
from launch.api_client.model.memory import Memory
from launch.api_client.model.model_bundle_environment_params import (
    ModelBundleEnvironmentParams,
)
from launch.api_client.model.model_bundle_framework import ModelBundleFramework
from launch.api_client.model.model_bundle_order_by import ModelBundleOrderBy
from launch.api_client.model.model_bundle_packaging_type import (
    ModelBundlePackagingType,
)
from launch.api_client.model.model_bundle_response import ModelBundleResponse
from launch.api_client.model.model_bundle_review_status import (
    ModelBundleReviewStatus,
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
from launch.api_client.model.storage import Storage
from launch.api_client.model.sync_endpoint_predict_response import (
    SyncEndpointPredictResponse,
)
from launch.api_client.model.task_status import TaskStatus
from launch.api_client.model.update_model_endpoint_request import (
    UpdateModelEndpointRequest,
)
from launch.api_client.model.update_model_endpoint_response import (
    UpdateModelEndpointResponse,
)
from launch.api_client.model.validation_error import ValidationError
