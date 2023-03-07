# flake8: noqa E501
from asyncio import get_event_loop
from typing import TYPE_CHECKING, Awaitable

from fastapi.encoders import jsonable_encoder

from launch.api_client import models as m

if TYPE_CHECKING:
    from launch.api_client.api_client import ApiClient


class _DefaultApi:
    def __init__(self, api_client: "ApiClient"):
        self.api_client = api_client

    def _build_for_clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post(
        self, clone_model_bundle_request: m.CloneModelBundleRequest
    ) -> Awaitable[m.CreateModelBundleResponse]:
        """
        Creates a ModelBundle by cloning an existing one and then applying changes on top.
        """
        body = jsonable_encoder(clone_model_bundle_request)

        return self.api_client.request(
            type_=m.CreateModelBundleResponse,
            method="POST",
            url="/v1/model-bundles/clone-with-changes",
            json=body,
        )

    def _build_for_create_async_inference_task_v1_async_tasks_post(
        self, model_endpoint_id: str, endpoint_predict_request: m.EndpointPredictRequest
    ) -> Awaitable[m.CreateAsyncTaskResponse]:
        """
        Runs an async inference prediction.
        """
        query_params = {"model_endpoint_id": str(model_endpoint_id)}

        body = jsonable_encoder(endpoint_predict_request)

        return self.api_client.request(
            type_=m.CreateAsyncTaskResponse,
            method="POST",
            url="/v1/async-tasks",
            params=query_params,
            json=body,
        )

    def _build_for_create_batch_job_v1_batch_jobs_post(
        self, create_batch_job_request: m.CreateBatchJobRequest
    ) -> Awaitable[m.CreateBatchJobResponse]:
        """
        Runs a sync inference prediction.
        """
        body = jsonable_encoder(create_batch_job_request)

        return self.api_client.request(type_=m.CreateBatchJobResponse, method="POST", url="/v1/batch-jobs", json=body)

    def _build_for_create_model_bundle_v1_model_bundles_post(
        self, create_model_bundle_request: m.CreateModelBundleRequest
    ) -> Awaitable[m.CreateModelBundleResponse]:
        """
        Creates a ModelBundle for the current user.
        """
        body = jsonable_encoder(create_model_bundle_request)

        return self.api_client.request(
            type_=m.CreateModelBundleResponse, method="POST", url="/v1/model-bundles", json=body
        )

    def _build_for_create_model_endpoint_v1_model_endpoints_post(
        self, create_model_endpoint_request: m.CreateModelEndpointRequest
    ) -> Awaitable[m.CreateModelEndpointResponse]:
        """
        Creates a Model for the current user.
        """
        body = jsonable_encoder(create_model_endpoint_request)

        return self.api_client.request(
            type_=m.CreateModelEndpointResponse, method="POST", url="/v1/model-endpoints", json=body
        )

    def _build_for_create_sync_inference_task_v1_sync_tasks_post(
        self, model_endpoint_id: str, endpoint_predict_request: m.EndpointPredictRequest
    ) -> Awaitable[m.SyncEndpointPredictResponse]:
        """
        Runs a sync inference prediction.
        """
        query_params = {"model_endpoint_id": str(model_endpoint_id)}

        body = jsonable_encoder(endpoint_predict_request)

        return self.api_client.request(
            type_=m.SyncEndpointPredictResponse,
            method="POST",
            url="/v1/sync-tasks",
            params=query_params,
            json=body,
        )

    def _build_for_delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete(
        self, model_endpoint_id: str
    ) -> Awaitable[m.DeleteModelEndpointResponse]:
        """
        Lists the Models owned by the current owner.
        """
        path_params = {"model_endpoint_id": str(model_endpoint_id)}

        return self.api_client.request(
            type_=m.DeleteModelEndpointResponse,
            method="DELETE",
            url="/v1/model-endpoints/{model_endpoint_id}",
            path_params=path_params,
        )

    def _build_for_get_async_inference_task_v1_async_tasks_task_id_get(
        self, task_id: str
    ) -> Awaitable[m.GetAsyncTaskResponse]:
        """
        Gets the status of an async inference task.
        """
        path_params = {"task_id": str(task_id)}

        return self.api_client.request(
            type_=m.GetAsyncTaskResponse,
            method="GET",
            url="/v1/async-tasks/{task_id}",
            path_params=path_params,
        )

    def _build_for_get_batch_job_v1_batch_jobs_batch_job_id_get(
        self, batch_job_id: str
    ) -> Awaitable[m.GetBatchJobResponse]:
        """
        Runs a sync inference prediction.
        """
        path_params = {"batch_job_id": str(batch_job_id)}

        return self.api_client.request(
            type_=m.GetBatchJobResponse,
            method="GET",
            url="/v1/batch-jobs/{batch_job_id}",
            path_params=path_params,
        )

    def _build_for_get_latest_model_bundle_v1_model_bundles_latest_get(
        self, model_name: str
    ) -> Awaitable[m.ModelBundleResponse]:
        """
        Gets the the latest Model Bundle with the given name owned by the current owner.
        """
        query_params = {"model_name": str(model_name)}

        return self.api_client.request(
            type_=m.ModelBundleResponse,
            method="GET",
            url="/v1/model-bundles/latest",
            params=query_params,
        )

    def _build_for_get_model_bundle_v1_model_bundles_model_bundle_id_get(
        self, model_bundle_id: str
    ) -> Awaitable[m.ModelBundleResponse]:
        """
        Gets the details for a given ModelBundle owned by the current owner.
        """
        path_params = {"model_bundle_id": str(model_bundle_id)}

        return self.api_client.request(
            type_=m.ModelBundleResponse,
            method="GET",
            url="/v1/model-bundles/{model_bundle_id}",
            path_params=path_params,
        )

    def _build_for_get_model_endpoint_v1_model_endpoints_model_endpoint_id_get(
        self, model_endpoint_id: str
    ) -> Awaitable[m.GetModelEndpointResponse]:
        """
        Lists the Models owned by the current owner.
        """
        path_params = {"model_endpoint_id": str(model_endpoint_id)}

        return self.api_client.request(
            type_=m.GetModelEndpointResponse,
            method="GET",
            url="/v1/model-endpoints/{model_endpoint_id}",
            path_params=path_params,
        )

    def _build_for_get_model_endpoints_api_v1_model_endpoints_api_get(
        self,
    ) -> Awaitable[m.Any]:
        """
        Shows the API of the Model Endpoints owned by the current owner.
        """
        return self.api_client.request(
            type_=m.Any,
            method="GET",
            url="/v1/model-endpoints-api",
        )

    def _build_for_get_model_endpoints_schema_v1_model_endpoints_schema_json_get(
        self,
    ) -> Awaitable[m.Any]:
        """
        Lists the schemas of the Model Endpoints owned by the current owner.
        """
        return self.api_client.request(
            type_=m.Any,
            method="GET",
            url="/v1/model-endpoints-schema.json",
        )

    def _build_for_healthcheck_healthcheck_get(
        self,
    ) -> Awaitable[m.Any]:
        """
        Returns 200 if the app is healthy.
        """
        return self.api_client.request(
            type_=m.Any,
            method="GET",
            url="/healthcheck",
        )

    def _build_for_healthcheck_healthz_get(
        self,
    ) -> Awaitable[m.Any]:
        """
        Returns 200 if the app is healthy.
        """
        return self.api_client.request(
            type_=m.Any,
            method="GET",
            url="/healthz",
        )

    def _build_for_healthcheck_readyz_get(
        self,
    ) -> Awaitable[m.Any]:
        """
        Returns 200 if the app is healthy.
        """
        return self.api_client.request(
            type_=m.Any,
            method="GET",
            url="/readyz",
        )

    def _build_for_list_model_bundles_v1_model_bundles_get(
        self, model_name: str = None, order_by: m.ModelBundleOrderBy = None
    ) -> Awaitable[m.ListModelBundlesResponse]:
        """
        Lists the ModelBundles owned by the current owner.
        """
        query_params = {}
        if model_name is not None:
            query_params["model_name"] = str(model_name)
        if order_by is not None:
            query_params["order_by"] = str(order_by)

        return self.api_client.request(
            type_=m.ListModelBundlesResponse,
            method="GET",
            url="/v1/model-bundles",
            params=query_params,
        )

    def _build_for_list_model_endpoints_v1_model_endpoints_get(
        self, name: str = None, order_by: m.ModelEndpointOrderBy = None
    ) -> Awaitable[m.ListModelEndpointsResponse]:
        """
        Lists the Models owned by the current owner.
        """
        query_params = {}
        if name is not None:
            query_params["name"] = str(name)
        if order_by is not None:
            query_params["order_by"] = str(order_by)

        return self.api_client.request(
            type_=m.ListModelEndpointsResponse,
            method="GET",
            url="/v1/model-endpoints",
            params=query_params,
        )

    def _build_for_update_batch_job_v1_batch_jobs_batch_job_id_put(
        self, batch_job_id: str, update_batch_job_request: m.UpdateBatchJobRequest
    ) -> Awaitable[m.UpdateBatchJobResponse]:
        """
        Runs a sync inference prediction.
        """
        path_params = {"batch_job_id": str(batch_job_id)}

        body = jsonable_encoder(update_batch_job_request)

        return self.api_client.request(
            type_=m.UpdateBatchJobResponse,
            method="PUT",
            url="/v1/batch-jobs/{batch_job_id}",
            path_params=path_params,
            json=body,
        )

    def _build_for_update_model_endpoint_v1_model_endpoints_model_endpoint_id_put(
        self, model_endpoint_id: str, update_model_endpoint_request: m.UpdateModelEndpointRequest
    ) -> Awaitable[m.UpdateModelEndpointResponse]:
        """
        Lists the Models owned by the current owner.
        """
        path_params = {"model_endpoint_id": str(model_endpoint_id)}

        body = jsonable_encoder(update_model_endpoint_request)

        return self.api_client.request(
            type_=m.UpdateModelEndpointResponse,
            method="PUT",
            url="/v1/model-endpoints/{model_endpoint_id}",
            path_params=path_params,
            json=body,
        )


class AsyncDefaultApi(_DefaultApi):
    async def clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post(
        self, clone_model_bundle_request: m.CloneModelBundleRequest
    ) -> m.CreateModelBundleResponse:
        """
        Creates a ModelBundle by cloning an existing one and then applying changes on top.
        """
        return await self._build_for_clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post(
            clone_model_bundle_request=clone_model_bundle_request
        )

    async def create_async_inference_task_v1_async_tasks_post(
        self, model_endpoint_id: str, endpoint_predict_request: m.EndpointPredictRequest
    ) -> m.CreateAsyncTaskResponse:
        """
        Runs an async inference prediction.
        """
        return await self._build_for_create_async_inference_task_v1_async_tasks_post(
            model_endpoint_id=model_endpoint_id, endpoint_predict_request=endpoint_predict_request
        )

    async def create_batch_job_v1_batch_jobs_post(
        self, create_batch_job_request: m.CreateBatchJobRequest
    ) -> m.CreateBatchJobResponse:
        """
        Runs a sync inference prediction.
        """
        return await self._build_for_create_batch_job_v1_batch_jobs_post(
            create_batch_job_request=create_batch_job_request
        )

    async def create_model_bundle_v1_model_bundles_post(
        self, create_model_bundle_request: m.CreateModelBundleRequest
    ) -> m.CreateModelBundleResponse:
        """
        Creates a ModelBundle for the current user.
        """
        return await self._build_for_create_model_bundle_v1_model_bundles_post(
            create_model_bundle_request=create_model_bundle_request
        )

    async def create_model_endpoint_v1_model_endpoints_post(
        self, create_model_endpoint_request: m.CreateModelEndpointRequest
    ) -> m.CreateModelEndpointResponse:
        """
        Creates a Model for the current user.
        """
        return await self._build_for_create_model_endpoint_v1_model_endpoints_post(
            create_model_endpoint_request=create_model_endpoint_request
        )

    async def create_sync_inference_task_v1_sync_tasks_post(
        self, model_endpoint_id: str, endpoint_predict_request: m.EndpointPredictRequest
    ) -> m.SyncEndpointPredictResponse:
        """
        Runs a sync inference prediction.
        """
        return await self._build_for_create_sync_inference_task_v1_sync_tasks_post(
            model_endpoint_id=model_endpoint_id, endpoint_predict_request=endpoint_predict_request
        )

    async def delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete(
        self, model_endpoint_id: str
    ) -> m.DeleteModelEndpointResponse:
        """
        Lists the Models owned by the current owner.
        """
        return await self._build_for_delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete(
            model_endpoint_id=model_endpoint_id
        )

    async def get_async_inference_task_v1_async_tasks_task_id_get(self, task_id: str) -> m.GetAsyncTaskResponse:
        """
        Gets the status of an async inference task.
        """
        return await self._build_for_get_async_inference_task_v1_async_tasks_task_id_get(task_id=task_id)

    async def get_batch_job_v1_batch_jobs_batch_job_id_get(self, batch_job_id: str) -> m.GetBatchJobResponse:
        """
        Runs a sync inference prediction.
        """
        return await self._build_for_get_batch_job_v1_batch_jobs_batch_job_id_get(batch_job_id=batch_job_id)

    async def get_latest_model_bundle_v1_model_bundles_latest_get(self, model_name: str) -> m.ModelBundleResponse:
        """
        Gets the the latest Model Bundle with the given name owned by the current owner.
        """
        return await self._build_for_get_latest_model_bundle_v1_model_bundles_latest_get(model_name=model_name)

    async def get_model_bundle_v1_model_bundles_model_bundle_id_get(
        self, model_bundle_id: str
    ) -> m.ModelBundleResponse:
        """
        Gets the details for a given ModelBundle owned by the current owner.
        """
        return await self._build_for_get_model_bundle_v1_model_bundles_model_bundle_id_get(
            model_bundle_id=model_bundle_id
        )

    async def get_model_endpoint_v1_model_endpoints_model_endpoint_id_get(
        self, model_endpoint_id: str
    ) -> m.GetModelEndpointResponse:
        """
        Lists the Models owned by the current owner.
        """
        return await self._build_for_get_model_endpoint_v1_model_endpoints_model_endpoint_id_get(
            model_endpoint_id=model_endpoint_id
        )

    async def get_model_endpoints_api_v1_model_endpoints_api_get(
        self,
    ) -> m.Any:
        """
        Shows the API of the Model Endpoints owned by the current owner.
        """
        return await self._build_for_get_model_endpoints_api_v1_model_endpoints_api_get()

    async def get_model_endpoints_schema_v1_model_endpoints_schema_json_get(
        self,
    ) -> m.Any:
        """
        Lists the schemas of the Model Endpoints owned by the current owner.
        """
        return await self._build_for_get_model_endpoints_schema_v1_model_endpoints_schema_json_get()

    async def healthcheck_healthcheck_get(
        self,
    ) -> m.Any:
        """
        Returns 200 if the app is healthy.
        """
        return await self._build_for_healthcheck_healthcheck_get()

    async def healthcheck_healthz_get(
        self,
    ) -> m.Any:
        """
        Returns 200 if the app is healthy.
        """
        return await self._build_for_healthcheck_healthz_get()

    async def healthcheck_readyz_get(
        self,
    ) -> m.Any:
        """
        Returns 200 if the app is healthy.
        """
        return await self._build_for_healthcheck_readyz_get()

    async def list_model_bundles_v1_model_bundles_get(
        self, model_name: str = None, order_by: m.ModelBundleOrderBy = None
    ) -> m.ListModelBundlesResponse:
        """
        Lists the ModelBundles owned by the current owner.
        """
        return await self._build_for_list_model_bundles_v1_model_bundles_get(model_name=model_name, order_by=order_by)

    async def list_model_endpoints_v1_model_endpoints_get(
        self, name: str = None, order_by: m.ModelEndpointOrderBy = None
    ) -> m.ListModelEndpointsResponse:
        """
        Lists the Models owned by the current owner.
        """
        return await self._build_for_list_model_endpoints_v1_model_endpoints_get(name=name, order_by=order_by)

    async def update_batch_job_v1_batch_jobs_batch_job_id_put(
        self, batch_job_id: str, update_batch_job_request: m.UpdateBatchJobRequest
    ) -> m.UpdateBatchJobResponse:
        """
        Runs a sync inference prediction.
        """
        return await self._build_for_update_batch_job_v1_batch_jobs_batch_job_id_put(
            batch_job_id=batch_job_id, update_batch_job_request=update_batch_job_request
        )

    async def update_model_endpoint_v1_model_endpoints_model_endpoint_id_put(
        self, model_endpoint_id: str, update_model_endpoint_request: m.UpdateModelEndpointRequest
    ) -> m.UpdateModelEndpointResponse:
        """
        Lists the Models owned by the current owner.
        """
        return await self._build_for_update_model_endpoint_v1_model_endpoints_model_endpoint_id_put(
            model_endpoint_id=model_endpoint_id,
            update_model_endpoint_request=update_model_endpoint_request,
        )


class SyncDefaultApi(_DefaultApi):
    def clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post(
        self, clone_model_bundle_request: m.CloneModelBundleRequest
    ) -> m.CreateModelBundleResponse:
        """
        Creates a ModelBundle by cloning an existing one and then applying changes on top.
        """
        coroutine = self._build_for_clone_model_bundle_with_changes_v1_model_bundles_clone_with_changes_post(
            clone_model_bundle_request=clone_model_bundle_request
        )
        return get_event_loop().run_until_complete(coroutine)

    def create_async_inference_task_v1_async_tasks_post(
        self, model_endpoint_id: str, endpoint_predict_request: m.EndpointPredictRequest
    ) -> m.CreateAsyncTaskResponse:
        """
        Runs an async inference prediction.
        """
        coroutine = self._build_for_create_async_inference_task_v1_async_tasks_post(
            model_endpoint_id=model_endpoint_id, endpoint_predict_request=endpoint_predict_request
        )
        return get_event_loop().run_until_complete(coroutine)

    def create_batch_job_v1_batch_jobs_post(
        self, create_batch_job_request: m.CreateBatchJobRequest
    ) -> m.CreateBatchJobResponse:
        """
        Runs a sync inference prediction.
        """
        coroutine = self._build_for_create_batch_job_v1_batch_jobs_post(
            create_batch_job_request=create_batch_job_request
        )
        return get_event_loop().run_until_complete(coroutine)

    def create_model_bundle_v1_model_bundles_post(
        self, create_model_bundle_request: m.CreateModelBundleRequest
    ) -> m.CreateModelBundleResponse:
        """
        Creates a ModelBundle for the current user.
        """
        coroutine = self._build_for_create_model_bundle_v1_model_bundles_post(
            create_model_bundle_request=create_model_bundle_request
        )
        return get_event_loop().run_until_complete(coroutine)

    def create_model_endpoint_v1_model_endpoints_post(
        self, create_model_endpoint_request: m.CreateModelEndpointRequest
    ) -> m.CreateModelEndpointResponse:
        """
        Creates a Model for the current user.
        """
        coroutine = self._build_for_create_model_endpoint_v1_model_endpoints_post(
            create_model_endpoint_request=create_model_endpoint_request
        )
        return get_event_loop().run_until_complete(coroutine)

    def create_sync_inference_task_v1_sync_tasks_post(
        self, model_endpoint_id: str, endpoint_predict_request: m.EndpointPredictRequest
    ) -> m.SyncEndpointPredictResponse:
        """
        Runs a sync inference prediction.
        """
        coroutine = self._build_for_create_sync_inference_task_v1_sync_tasks_post(
            model_endpoint_id=model_endpoint_id, endpoint_predict_request=endpoint_predict_request
        )
        return get_event_loop().run_until_complete(coroutine)

    def delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete(
        self, model_endpoint_id: str
    ) -> m.DeleteModelEndpointResponse:
        """
        Lists the Models owned by the current owner.
        """
        coroutine = self._build_for_delete_model_endpoint_v1_model_endpoints_model_endpoint_id_delete(
            model_endpoint_id=model_endpoint_id
        )
        return get_event_loop().run_until_complete(coroutine)

    def get_async_inference_task_v1_async_tasks_task_id_get(self, task_id: str) -> m.GetAsyncTaskResponse:
        """
        Gets the status of an async inference task.
        """
        coroutine = self._build_for_get_async_inference_task_v1_async_tasks_task_id_get(task_id=task_id)
        return get_event_loop().run_until_complete(coroutine)

    def get_batch_job_v1_batch_jobs_batch_job_id_get(self, batch_job_id: str) -> m.GetBatchJobResponse:
        """
        Runs a sync inference prediction.
        """
        coroutine = self._build_for_get_batch_job_v1_batch_jobs_batch_job_id_get(batch_job_id=batch_job_id)
        return get_event_loop().run_until_complete(coroutine)

    def get_latest_model_bundle_v1_model_bundles_latest_get(self, model_name: str) -> m.ModelBundleResponse:
        """
        Gets the the latest Model Bundle with the given name owned by the current owner.
        """
        coroutine = self._build_for_get_latest_model_bundle_v1_model_bundles_latest_get(model_name=model_name)
        return get_event_loop().run_until_complete(coroutine)

    def get_model_bundle_v1_model_bundles_model_bundle_id_get(self, model_bundle_id: str) -> m.ModelBundleResponse:
        """
        Gets the details for a given ModelBundle owned by the current owner.
        """
        coroutine = self._build_for_get_model_bundle_v1_model_bundles_model_bundle_id_get(
            model_bundle_id=model_bundle_id
        )
        return get_event_loop().run_until_complete(coroutine)

    def get_model_endpoint_v1_model_endpoints_model_endpoint_id_get(
        self, model_endpoint_id: str
    ) -> m.GetModelEndpointResponse:
        """
        Lists the Models owned by the current owner.
        """
        coroutine = self._build_for_get_model_endpoint_v1_model_endpoints_model_endpoint_id_get(
            model_endpoint_id=model_endpoint_id
        )
        return get_event_loop().run_until_complete(coroutine)

    def get_model_endpoints_api_v1_model_endpoints_api_get(
        self,
    ) -> m.Any:
        """
        Shows the API of the Model Endpoints owned by the current owner.
        """
        coroutine = self._build_for_get_model_endpoints_api_v1_model_endpoints_api_get()
        return get_event_loop().run_until_complete(coroutine)

    def get_model_endpoints_schema_v1_model_endpoints_schema_json_get(
        self,
    ) -> m.Any:
        """
        Lists the schemas of the Model Endpoints owned by the current owner.
        """
        coroutine = self._build_for_get_model_endpoints_schema_v1_model_endpoints_schema_json_get()
        return get_event_loop().run_until_complete(coroutine)

    def healthcheck_healthcheck_get(
        self,
    ) -> m.Any:
        """
        Returns 200 if the app is healthy.
        """
        coroutine = self._build_for_healthcheck_healthcheck_get()
        return get_event_loop().run_until_complete(coroutine)

    def healthcheck_healthz_get(
        self,
    ) -> m.Any:
        """
        Returns 200 if the app is healthy.
        """
        coroutine = self._build_for_healthcheck_healthz_get()
        return get_event_loop().run_until_complete(coroutine)

    def healthcheck_readyz_get(
        self,
    ) -> m.Any:
        """
        Returns 200 if the app is healthy.
        """
        coroutine = self._build_for_healthcheck_readyz_get()
        return get_event_loop().run_until_complete(coroutine)

    def list_model_bundles_v1_model_bundles_get(
        self, model_name: str = None, order_by: m.ModelBundleOrderBy = None
    ) -> m.ListModelBundlesResponse:
        """
        Lists the ModelBundles owned by the current owner.
        """
        coroutine = self._build_for_list_model_bundles_v1_model_bundles_get(model_name=model_name, order_by=order_by)
        return get_event_loop().run_until_complete(coroutine)

    def list_model_endpoints_v1_model_endpoints_get(
        self, name: str = None, order_by: m.ModelEndpointOrderBy = None
    ) -> m.ListModelEndpointsResponse:
        """
        Lists the Models owned by the current owner.
        """
        coroutine = self._build_for_list_model_endpoints_v1_model_endpoints_get(name=name, order_by=order_by)
        return get_event_loop().run_until_complete(coroutine)

    def update_batch_job_v1_batch_jobs_batch_job_id_put(
        self, batch_job_id: str, update_batch_job_request: m.UpdateBatchJobRequest
    ) -> m.UpdateBatchJobResponse:
        """
        Runs a sync inference prediction.
        """
        coroutine = self._build_for_update_batch_job_v1_batch_jobs_batch_job_id_put(
            batch_job_id=batch_job_id, update_batch_job_request=update_batch_job_request
        )
        return get_event_loop().run_until_complete(coroutine)

    def update_model_endpoint_v1_model_endpoints_model_endpoint_id_put(
        self, model_endpoint_id: str, update_model_endpoint_request: m.UpdateModelEndpointRequest
    ) -> m.UpdateModelEndpointResponse:
        """
        Lists the Models owned by the current owner.
        """
        coroutine = self._build_for_update_model_endpoint_v1_model_endpoints_model_endpoint_id_put(
            model_endpoint_id=model_endpoint_id,
            update_model_endpoint_request=update_model_endpoint_request,
        )
        return get_event_loop().run_until_complete(coroutine)
