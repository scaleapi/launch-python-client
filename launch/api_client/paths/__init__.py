# do not import all endpoints into this module because that uses a lot of memory and stack frames
# if you need the ability to import all endpoints from this module, import them with
# from launch.api_client.apis.path_to_api import path_to_api

import enum


class PathValues(str, enum.Enum):
    HEALTHCHECK = "/healthcheck"
    HEALTHZ = "/healthz"
    READYZ = "/readyz"
    V1_ASYNCTASKS = "/v1/async-tasks"
    V1_ASYNCTASKS_TASK_ID = "/v1/async-tasks/{task_id}"
    V1_BATCHJOBS = "/v1/batch-jobs"
    V1_BATCHJOBS_BATCH_JOB_ID = "/v1/batch-jobs/{batch_job_id}"
    V1_DOCKERIMAGEBATCHJOBBUNDLES = "/v1/docker-image-batch-job-bundles"
    V1_DOCKERIMAGEBATCHJOBBUNDLES_LATEST = "/v1/docker-image-batch-job-bundles/latest"
    V1_DOCKERIMAGEBATCHJOBBUNDLES_DOCKER_IMAGE_BATCH_JOB_BUNDLE_ID = (
        "/v1/docker-image-batch-job-bundles/{docker_image_batch_job_bundle_id}"
    )
    V1_DOCKERIMAGEBATCHJOBS = "/v1/docker-image-batch-jobs"
    V1_DOCKERIMAGEBATCHJOBS_BATCH_JOB_ID = "/v1/docker-image-batch-jobs/{batch_job_id}"
    V1_MODELBUNDLES = "/v1/model-bundles"
    V1_MODELBUNDLES_CLONEWITHCHANGES = "/v1/model-bundles/clone-with-changes"
    V1_MODELBUNDLES_LATEST = "/v1/model-bundles/latest"
    V1_MODELBUNDLES_MODEL_BUNDLE_ID = "/v1/model-bundles/{model_bundle_id}"
    V1_MODELENDPOINTS = "/v1/model-endpoints"
    V1_MODELENDPOINTSAPI = "/v1/model-endpoints-api"
    V1_MODELENDPOINTSSCHEMA_JSON = "/v1/model-endpoints-schema.json"
    V1_MODELENDPOINTS_MODEL_ENDPOINT_ID = "/v1/model-endpoints/{model_endpoint_id}"
    V1_STREAMINGTASKS = "/v1/streaming-tasks"
    V1_SYNCTASKS = "/v1/sync-tasks"
    V2_MODELBUNDLES = "/v2/model-bundles"
    V2_MODELBUNDLES_CLONEWITHCHANGES = "/v2/model-bundles/clone-with-changes"
    V2_MODELBUNDLES_LATEST = "/v2/model-bundles/latest"
    V2_MODELBUNDLES_MODEL_BUNDLE_ID = "/v2/model-bundles/{model_bundle_id}"
