import typing_extensions

from launch.api_client.apis.paths.healthcheck import Healthcheck
from launch.api_client.apis.paths.healthz import Healthz
from launch.api_client.apis.paths.readyz import Readyz
from launch.api_client.apis.paths.v1_async_tasks import V1AsyncTasks
from launch.api_client.apis.paths.v1_async_tasks_task_id import (
    V1AsyncTasksTaskId,
)
from launch.api_client.apis.paths.v1_batch_jobs import V1BatchJobs
from launch.api_client.apis.paths.v1_batch_jobs_batch_job_id import (
    V1BatchJobsBatchJobId,
)
from launch.api_client.apis.paths.v1_docker_image_batch_job_bundles import (
    V1DockerImageBatchJobBundles,
)
from launch.api_client.apis.paths.v1_docker_image_batch_job_bundles_docker_image_batch_job_bundle_id import (
    V1DockerImageBatchJobBundlesDockerImageBatchJobBundleId,
)
from launch.api_client.apis.paths.v1_docker_image_batch_job_bundles_latest import (
    V1DockerImageBatchJobBundlesLatest,
)
from launch.api_client.apis.paths.v1_docker_image_batch_jobs import (
    V1DockerImageBatchJobs,
)
from launch.api_client.apis.paths.v1_docker_image_batch_jobs_batch_job_id import (
    V1DockerImageBatchJobsBatchJobId,
)
from launch.api_client.apis.paths.v1_model_bundles import V1ModelBundles
from launch.api_client.apis.paths.v1_model_bundles_clone_with_changes import (
    V1ModelBundlesCloneWithChanges,
)
from launch.api_client.apis.paths.v1_model_bundles_latest import (
    V1ModelBundlesLatest,
)
from launch.api_client.apis.paths.v1_model_bundles_model_bundle_id import (
    V1ModelBundlesModelBundleId,
)
from launch.api_client.apis.paths.v1_model_endpoints import V1ModelEndpoints
from launch.api_client.apis.paths.v1_model_endpoints_api import (
    V1ModelEndpointsApi,
)
from launch.api_client.apis.paths.v1_model_endpoints_model_endpoint_id import (
    V1ModelEndpointsModelEndpointId,
)
from launch.api_client.apis.paths.v1_model_endpoints_schema_json import (
    V1ModelEndpointsSchemaJson,
)
from launch.api_client.apis.paths.v1_sync_tasks import V1SyncTasks
from launch.api_client.apis.paths.v2_model_bundles import V2ModelBundles
from launch.api_client.apis.paths.v2_model_bundles_clone_with_changes import (
    V2ModelBundlesCloneWithChanges,
)
from launch.api_client.apis.paths.v2_model_bundles_latest import (
    V2ModelBundlesLatest,
)
from launch.api_client.apis.paths.v2_model_bundles_model_bundle_id import (
    V2ModelBundlesModelBundleId,
)
from launch.api_client.paths import PathValues

PathToApi = typing_extensions.TypedDict(
    "PathToApi",
    {
        PathValues.HEALTHCHECK: Healthcheck,
        PathValues.HEALTHZ: Healthz,
        PathValues.READYZ: Readyz,
        PathValues.V1_ASYNCTASKS: V1AsyncTasks,
        PathValues.V1_ASYNCTASKS_TASK_ID: V1AsyncTasksTaskId,
        PathValues.V1_BATCHJOBS: V1BatchJobs,
        PathValues.V1_BATCHJOBS_BATCH_JOB_ID: V1BatchJobsBatchJobId,
        PathValues.V1_DOCKERIMAGEBATCHJOBBUNDLES: V1DockerImageBatchJobBundles,
        PathValues.V1_DOCKERIMAGEBATCHJOBBUNDLES_LATEST: V1DockerImageBatchJobBundlesLatest,
        PathValues.V1_DOCKERIMAGEBATCHJOBBUNDLES_DOCKER_IMAGE_BATCH_JOB_BUNDLE_ID: V1DockerImageBatchJobBundlesDockerImageBatchJobBundleId,
        PathValues.V1_DOCKERIMAGEBATCHJOBS: V1DockerImageBatchJobs,
        PathValues.V1_DOCKERIMAGEBATCHJOBS_BATCH_JOB_ID: V1DockerImageBatchJobsBatchJobId,
        PathValues.V1_MODELBUNDLES: V1ModelBundles,
        PathValues.V1_MODELBUNDLES_CLONEWITHCHANGES: V1ModelBundlesCloneWithChanges,
        PathValues.V1_MODELBUNDLES_LATEST: V1ModelBundlesLatest,
        PathValues.V1_MODELBUNDLES_MODEL_BUNDLE_ID: V1ModelBundlesModelBundleId,
        PathValues.V1_MODELENDPOINTS: V1ModelEndpoints,
        PathValues.V1_MODELENDPOINTSAPI: V1ModelEndpointsApi,
        PathValues.V1_MODELENDPOINTSSCHEMA_JSON: V1ModelEndpointsSchemaJson,
        PathValues.V1_MODELENDPOINTS_MODEL_ENDPOINT_ID: V1ModelEndpointsModelEndpointId,
        PathValues.V1_SYNCTASKS: V1SyncTasks,
        PathValues.V2_MODELBUNDLES: V2ModelBundles,
        PathValues.V2_MODELBUNDLES_CLONEWITHCHANGES: V2ModelBundlesCloneWithChanges,
        PathValues.V2_MODELBUNDLES_LATEST: V2ModelBundlesLatest,
        PathValues.V2_MODELBUNDLES_MODEL_BUNDLE_ID: V2ModelBundlesModelBundleId,
    },
)

path_to_api = PathToApi(
    {
        PathValues.HEALTHCHECK: Healthcheck,
        PathValues.HEALTHZ: Healthz,
        PathValues.READYZ: Readyz,
        PathValues.V1_ASYNCTASKS: V1AsyncTasks,
        PathValues.V1_ASYNCTASKS_TASK_ID: V1AsyncTasksTaskId,
        PathValues.V1_BATCHJOBS: V1BatchJobs,
        PathValues.V1_BATCHJOBS_BATCH_JOB_ID: V1BatchJobsBatchJobId,
        PathValues.V1_DOCKERIMAGEBATCHJOBBUNDLES: V1DockerImageBatchJobBundles,
        PathValues.V1_DOCKERIMAGEBATCHJOBBUNDLES_LATEST: V1DockerImageBatchJobBundlesLatest,
        PathValues.V1_DOCKERIMAGEBATCHJOBBUNDLES_DOCKER_IMAGE_BATCH_JOB_BUNDLE_ID: V1DockerImageBatchJobBundlesDockerImageBatchJobBundleId,
        PathValues.V1_DOCKERIMAGEBATCHJOBS: V1DockerImageBatchJobs,
        PathValues.V1_DOCKERIMAGEBATCHJOBS_BATCH_JOB_ID: V1DockerImageBatchJobsBatchJobId,
        PathValues.V1_MODELBUNDLES: V1ModelBundles,
        PathValues.V1_MODELBUNDLES_CLONEWITHCHANGES: V1ModelBundlesCloneWithChanges,
        PathValues.V1_MODELBUNDLES_LATEST: V1ModelBundlesLatest,
        PathValues.V1_MODELBUNDLES_MODEL_BUNDLE_ID: V1ModelBundlesModelBundleId,
        PathValues.V1_MODELENDPOINTS: V1ModelEndpoints,
        PathValues.V1_MODELENDPOINTSAPI: V1ModelEndpointsApi,
        PathValues.V1_MODELENDPOINTSSCHEMA_JSON: V1ModelEndpointsSchemaJson,
        PathValues.V1_MODELENDPOINTS_MODEL_ENDPOINT_ID: V1ModelEndpointsModelEndpointId,
        PathValues.V1_SYNCTASKS: V1SyncTasks,
        PathValues.V2_MODELBUNDLES: V2ModelBundles,
        PathValues.V2_MODELBUNDLES_CLONEWITHCHANGES: V2ModelBundlesCloneWithChanges,
        PathValues.V2_MODELBUNDLES_LATEST: V2ModelBundlesLatest,
        PathValues.V2_MODELBUNDLES_MODEL_BUNDLE_ID: V2ModelBundlesModelBundleId,
    }
)
