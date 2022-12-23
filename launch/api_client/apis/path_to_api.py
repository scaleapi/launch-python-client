import typing_extensions
from launch_client.apis.paths.healthcheck import Healthcheck
from launch_client.apis.paths.healthz import Healthz
from launch_client.apis.paths.readyz import Readyz
from launch_client.apis.paths.v1_async_tasks import V1AsyncTasks
from launch_client.apis.paths.v1_async_tasks_task_id import V1AsyncTasksTaskId
from launch_client.apis.paths.v1_model_bundles import V1ModelBundles
from launch_client.apis.paths.v1_model_bundles_latest import (
    V1ModelBundlesLatest,
)
from launch_client.apis.paths.v1_model_bundles_model_bundle_id import (
    V1ModelBundlesModelBundleId,
)
from launch_client.apis.paths.v1_model_endpoints import V1ModelEndpoints
from launch_client.apis.paths.v1_model_endpoints_model_endpoint_id import (
    V1ModelEndpointsModelEndpointId,
)
from launch_client.apis.paths.v1_sync_tasks import V1SyncTasks
from launch_client.paths import PathValues

PathToApi = typing_extensions.TypedDict(
    "PathToApi",
    {
        PathValues.HEALTHCHECK: Healthcheck,
        PathValues.HEALTHZ: Healthz,
        PathValues.READYZ: Readyz,
        PathValues.V1_ASYNCTASKS: V1AsyncTasks,
        PathValues.V1_ASYNCTASKS_TASK_ID: V1AsyncTasksTaskId,
        PathValues.V1_MODELBUNDLES: V1ModelBundles,
        PathValues.V1_MODELBUNDLES_LATEST: V1ModelBundlesLatest,
        PathValues.V1_MODELBUNDLES_MODEL_BUNDLE_ID: V1ModelBundlesModelBundleId,
        PathValues.V1_MODELENDPOINTS: V1ModelEndpoints,
        PathValues.V1_MODELENDPOINTS_MODEL_ENDPOINT_ID: V1ModelEndpointsModelEndpointId,
        PathValues.V1_SYNCTASKS: V1SyncTasks,
    },
)

path_to_api = PathToApi(
    {
        PathValues.HEALTHCHECK: Healthcheck,
        PathValues.HEALTHZ: Healthz,
        PathValues.READYZ: Readyz,
        PathValues.V1_ASYNCTASKS: V1AsyncTasks,
        PathValues.V1_ASYNCTASKS_TASK_ID: V1AsyncTasksTaskId,
        PathValues.V1_MODELBUNDLES: V1ModelBundles,
        PathValues.V1_MODELBUNDLES_LATEST: V1ModelBundlesLatest,
        PathValues.V1_MODELBUNDLES_MODEL_BUNDLE_ID: V1ModelBundlesModelBundleId,
        PathValues.V1_MODELENDPOINTS: V1ModelEndpoints,
        PathValues.V1_MODELENDPOINTS_MODEL_ENDPOINT_ID: V1ModelEndpointsModelEndpointId,
        PathValues.V1_SYNCTASKS: V1SyncTasks,
    }
)
