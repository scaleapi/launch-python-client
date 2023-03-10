import concurrent.futures
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from dataclasses_json import Undefined, dataclass_json
from deprecation import deprecated
from typing_extensions import Literal

from launch.api_client import ApiClient
from launch.api_client.apis.tags.default_api import DefaultApi
from launch.request_validation import validate_task_request

TASK_PENDING_STATE = "PENDING"
TASK_SUCCESS_STATE = "SUCCESS"
TASK_FAILURE_STATE = "FAILURE"


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class ModelEndpoint:
    """
    Represents an Endpoint from the database.
    """

    name: str
    """
    The name of the endpoint. Must be unique across all endpoints owned by the user.
    """

    id: Optional[str] = None
    """
    A globally unique identifier for the endpoint.
    """

    bundle_name: Optional[str] = None
    """
    The name of the bundle for the endpoint. The owner of the bundle must be the same as the owner for the endpoint.
    """

    status: Optional[str] = None
    """
    The status of the endpoint.
    """

    resource_state: Optional[dict] = None
    """
    Resource state for the endpoint.
    """

    deployment_state: Optional[dict] = None
    """
    Deployment state for the endpoint.
    """

    metadata: Optional[dict] = None
    """
    Metadata for the endpoint.
    """

    endpoint_type: Optional[str] = None
    """
    The type of the endpoint. Must be ``'async'`` or ``'sync'``.
    """

    configs: Optional[dict] = None
    """
    Config for the endpoint.
    """

    destination: Optional[str] = None
    """
    Queue identifier for endpoint, use only for debugging.
    """

    post_inference_hooks: Optional[List[str]] = None
    """
    List of post inference hooks for the endpoint.
    """

    default_callback_url: Optional[str] = None
    """
    Default callback url for the endpoint.
    """

    num_queued_items: Optional[int] = None
    """
    Number of items currently queued for the endpoint.
    """

    def __repr__(self):
        return (
            f"ModelEndpoint(name='{self.name}', bundle_name='{self.bundle_name}', "
            f"status='{self.status}', resource_state='{json.dumps(self.resource_state)}', "
            f"deployment_state='{json.dumps(self.deployment_state)}', "
            f"endpoint_type='{self.endpoint_type}', metadata='{self.metadata}', "
            f"num_queued_items='{self.num_queued_items}')"
        )


class EndpointRequest:
    """
    Represents a single request to either a ``SyncEndpoint`` or ``AsyncEndpoint``.

    Parameters:
        url: A url to some file that can be read in to a ModelBundle's predict function. Can be an image, raw text, etc.
            **Note**: the contents of the file located at ``url`` are opened as a sequence of ``bytes`` and passed
            to the predict function. If you instead want to pass the url itself as an input to the predict function,
            see ``args``.

            Exactly one of ``url`` and ``args`` must be specified.

        args: A Dictionary with arguments to a ModelBundle's predict function. If the predict function has signature
            ``predict_fn(foo, bar)``, then the keys in the dictionary should be ``"foo"`` and ``"bar"``.
            Values must be native Python objects.

            Exactly one of ``url`` and ``args`` must be specified.

        return_pickled: Whether the output should be a pickled python object, or directly returned serialized json.

        callback_url: The callback url to use for this task. If None, then the
            default_callback_url of the endpoint is used. The endpoint must specify
            "callback" as a post-inference hook for the callback to be triggered.

        callback_auth_kind: The default callback auth kind to use for async endpoints.
            Either "basic" or "mtls". This can be overridden in the task parameters for each
            individual task.

        callback_auth_username: The default callback auth username to use. This only
            applies if callback_auth_kind is "basic". This can be overridden in the task
            parameters for each individual task.

        callback_auth_password: The default callback auth password to use. This only
            applies if callback_auth_kind is "basic". This can be overridden in the task
            parameters for each individual task.

        callback_auth_cert: The default callback auth cert to use. This only applies
            if callback_auth_kind is "mtls". This can be overridden in the task
            parameters for each individual task.

        callback_auth_key: The default callback auth key to use. This only applies
            if callback_auth_kind is "mtls". This can be overridden in the task
            parameters for each individual task.

        request_id: (deprecated) A user-specifiable id for requests.
            Should be unique among EndpointRequests made in the same batch call.
            If one isn't provided the client will generate its own.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        args: Optional[Dict] = None,
        callback_url: Optional[str] = None,
        callback_auth_kind: Optional[Literal["basic", "mtls"]] = None,
        callback_auth_username: Optional[str] = None,
        callback_auth_password: Optional[str] = None,
        callback_auth_cert: Optional[str] = None,
        callback_auth_key: Optional[str] = None,
        return_pickled: Optional[bool] = False,
        request_id: Optional[str] = None,
    ):
        # TODO: request_id is pretty much here only to support the clientside AsyncEndpointBatchResponse
        # so it should be removed when we get proper batch endpoints working.
        validate_task_request(url=url, args=args)
        if request_id is None:
            request_id = str(uuid.uuid4())
        self.url = url
        self.args = args
        self.callback_url = callback_url
        self.callback_auth_kind = callback_auth_kind
        self.callback_auth_username = callback_auth_username
        self.callback_auth_password = callback_auth_password
        self.callback_auth_cert = callback_auth_cert
        self.callback_auth_key = callback_auth_key
        self.return_pickled = return_pickled
        self.request_id: str = request_id


class EndpointResponse:
    """
    Represents a response received from a Endpoint.

    """

    def __init__(
        self,
        client,
        status: str,
        result_url: Optional[str] = None,
        result: Optional[str] = None,
        traceback: Optional[str] = None,
    ):
        """
        Parameters:
            client: An instance of ``LaunchClient``.

            status: A string representing the status of the request, i.e. ``SUCCESS``, ``FAILURE``, or ``PENDING``

            result_url: A string that is a url containing the pickled python object from the
                Endpoint's predict function.

                Exactly one of ``result_url`` or ``result`` will be populated,
                depending on the value of ``return_pickled`` in the request.

            result: A string that is the serialized return value (in json form) of the Endpoint's predict function.
                Specifically, one can ``json.loads()`` the value of result to get the original python object back.

                Exactly one of ``result_url`` or ``result`` will be populated,
                depending on the value of ``return_pickled`` in the request.

            traceback: The stack trace if the inference endpoint raised an error. Can be used for debugging

        """
        self.client = client
        self.status = status
        self.result_url = result_url
        self.result = result
        self.traceback = traceback

    def __str__(self) -> str:
        return (
            f"status: {self.status}, result: {self.result}, result_url: {self.result_url}, "
            f"traceback: {self.traceback}"
        )


class EndpointResponseFuture:
    """
    Represents a future response from an Endpoint. Specifically, when the ``EndpointResponseFuture`` is ready,
    then its ``get`` method will return an actual instance of ``EndpointResponse``.

    This object should not be directly instantiated by the user.
    """

    def __init__(self, client, endpoint_name: str, async_task_id: str):
        """
        Parameters:
            client: An instance of ``LaunchClient``.

            endpoint_name: The name of the endpoint.

            async_task_id: An async task id.
        """
        self.client = client
        self.endpoint_name = endpoint_name
        self.async_task_id = async_task_id

    def get(self, timeout: Optional[float] = None) -> EndpointResponse:
        """
        Retrieves the ``EndpointResponse`` for the prediction request after it completes. This method blocks.

        Parameters:
            timeout: The maximum number of seconds to wait for the response. If None, then
                the method will block indefinitely until the response is ready.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError("Timeout must be greater than 0.")
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            async_response = self.client._get_async_endpoint_response(  # pylint: disable=W0212
                self.endpoint_name, self.async_task_id
            )
            status = async_response["status"]
            if status == "PENDING":
                time.sleep(2)
            else:
                if status == "SUCCESS":
                    return EndpointResponse(
                        client=self.client,
                        status=status,
                        result_url=async_response.get("result", {}).get("result_url", None),
                        result=async_response.get("result", {}).get("result", None),
                        traceback=None,
                    )
                elif status == "FAILURE":
                    return EndpointResponse(
                        client=self.client,
                        status=status,
                        result_url=None,
                        result=None,
                        traceback=async_response.get("traceback", None),
                    )
                else:
                    raise ValueError(f"Unrecognized status: {async_response['status']}")
        raise TimeoutError


class Endpoint(ABC):
    """An abstract class that represent any kind of endpoints in Scale Launch"""

    def __init__(self, model_endpoint: ModelEndpoint, client):
        self.model_endpoint = model_endpoint
        self.client = client

    def _update_model_endpoint_view(self):
        with ApiClient(self.client.configuration) as api_client:
            api_instance = DefaultApi(api_client)
            query_params = {"name": self.model_endpoint.name}
            response = api_instance.list_model_endpoints_v1_model_endpoints_get(
                query_params=query_params,
                skip_deserialization=True,
            )
            resp = json.loads(response.response.data)
            if len(resp["model_endpoints"]) == 0:
                raise ValueError(f"Could not update model endpoint view for endpoint {self.model_endpoint.name}")
            resp = resp["model_endpoints"][0]
        self.model_endpoint = ModelEndpoint.from_dict(resp)

    def status(self) -> Optional[str]:
        """Gets the status of the Endpoint."""
        self._update_model_endpoint_view()
        return self.model_endpoint.status

    def resource_state(self) -> Optional[dict]:
        """Gets the resource state of the Endpoint."""
        self._update_model_endpoint_view()
        return self.model_endpoint.resource_state

    def deployment_state(self) -> Optional[dict]:
        """Gets the worker settings of the Endpoint."""
        self._update_model_endpoint_view()
        return self.model_endpoint.deployment_state

    @abstractmethod
    def predict(self, request: EndpointRequest):
        """Runs a prediction request."""


class SyncEndpoint(Endpoint):
    """
    A synchronous model endpoint.
    """

    def __init__(self, model_endpoint: ModelEndpoint, client):
        """
        Parameters:
            model_endpoint: ModelEndpoint object.

            client: A LaunchClient object
        """
        super().__init__(model_endpoint=model_endpoint, client=client)

    def __str__(self):
        return f"SyncEndpoint <endpoint_name:{self.model_endpoint.name}>"

    def __repr__(self):
        return (
            f"SyncEndpoint(name='{self.model_endpoint.name}', "
            f"bundle_name='{self.model_endpoint.bundle_name}', "
            f"status='{self.model_endpoint.status}', "
            f"resource_state='{json.dumps(self.model_endpoint.resource_state)}', "
            f"deployment_state='{json.dumps(self.model_endpoint.deployment_state)}', "
            f"endpoint_type='{self.model_endpoint.endpoint_type}', "
            f"metadata='{self.model_endpoint.metadata}')"
        )

    def predict(self, request: EndpointRequest) -> EndpointResponse:
        """
        Runs a synchronous prediction request.

        Parameters:
            request: The ``EndpointRequest`` object that contains the payload.
        """
        raw_response = self.client._sync_request(  # pylint: disable=W0212
            self.model_endpoint.name,
            url=request.url,
            args=request.args,
            return_pickled=request.return_pickled,
        )
        raw_response = {k: v for k, v in raw_response.items() if v is not None}
        return EndpointResponse(
            client=self.client,
            status=raw_response.get("status"),
            result_url=raw_response.get("result", {}).get("result_url", None),
            result=raw_response.get("result", {}).get("result", None),
            traceback=raw_response.get("traceback", None),
        )


class AsyncEndpoint(Endpoint):
    """
    An asynchronous model endpoint.
    """

    def __init__(self, model_endpoint: ModelEndpoint, client):
        """
        Parameters:
            model_endpoint: ModelEndpoint object.

            client: A LaunchClient object
        """
        super().__init__(model_endpoint=model_endpoint, client=client)

    def __str__(self):
        return f"AsyncEndpoint <endpoint_name:{self.model_endpoint.name}>"

    def __repr__(self):
        return (
            f"AsyncEndpoint(name='{self.model_endpoint.name}', "
            f"bundle_name='{self.model_endpoint.bundle_name}', "
            f"status='{self.model_endpoint.status}', "
            f"resource_state='{json.dumps(self.model_endpoint.resource_state)}', "
            f"deployment_state='{json.dumps(self.model_endpoint.deployment_state)}', "
            f"endpoint_type='{self.model_endpoint.endpoint_type}', "
            f"metadata='{self.model_endpoint.metadata}')"
        )

    def predict(self, request: EndpointRequest) -> EndpointResponseFuture:
        """
        Runs an asynchronous prediction request.

        Parameters:
            request: The ``EndpointRequest`` object that contains the payload.

        Returns:
            An ``EndpointResponseFuture`` such the user can use to query the status of the request.
            Example:

            .. code-block:: python

                my_endpoint = AsyncEndpoint(...)
                f: EndpointResponseFuture = my_endpoint.predict(EndpointRequest(...))
                result = f.get()  # blocks on completion
        """
        response = self.client._async_request(  # pylint: disable=W0212
            self.model_endpoint.name,
            url=request.url,
            args=request.args,
            callback_url=request.callback_url,
            callback_auth_kind=request.callback_auth_kind,
            callback_auth_username=request.callback_auth_username,
            callback_auth_password=request.callback_auth_password,
            callback_auth_cert=request.callback_auth_cert,
            callback_auth_key=request.callback_auth_key,
            return_pickled=request.return_pickled,
        )
        async_task_id = response["task_id"]
        return EndpointResponseFuture(
            client=self.client,
            endpoint_name=self.model_endpoint.name,
            async_task_id=async_task_id,
        )

    @deprecated
    def predict_batch(self, requests: Sequence[EndpointRequest]) -> "AsyncEndpointBatchResponse":
        """
        (deprecated)
        Runs inference on the data items specified by urls. Returns a AsyncEndpointResponse.

        Parameters:
            requests: List of EndpointRequests. Request_ids must all be distinct.

        Returns:
            an AsyncEndpointResponse keeping track of the inference requests made
        """
        # Make inference requests to the endpoint,
        # if batches are possible make this aware you can pass batches
        # TODO add batch support once those are out

        if len(requests) != len(set(request.request_id for request in requests)):
            raise ValueError("Request_ids in a batch must be unique")

        def single_request(request):
            # request has keys url and args

            inner_inference_request = self.client._async_request(  # pylint: disable=W0212
                endpoint_name=self.model_endpoint.name,
                url=request.url,
                args=request.args,
                callback_url=request.callback_url,
                callback_auth_kind=request.callback_auth_kind,
                callback_auth_username=request.callback_auth_username,
                callback_auth_password=request.callback_auth_password,
                callback_auth_cert=request.callback_auth_cert,
                callback_auth_key=request.callback_auth_key,
                return_pickled=request.return_pickled,
            )
            request_key = request.request_id
            return request_key, inner_inference_request

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            urls_to_requests = executor.map(single_request, requests)
            request_ids = dict(urls_to_requests)

        return AsyncEndpointBatchResponse(
            self.client,
            endpoint_name=self.model_endpoint.name,
            request_ids=request_ids,
        )


@deprecated
class AsyncEndpointBatchResponse:
    """
    (deprecated)

    Currently represents a list of async inference requests to a specific endpoint. Keeps track of the requests made,
    and gives a way to poll for their status.

    Invariant: set keys for self.request_ids and self.responses are equal

    idk about this abstraction tbh, could use a redesign maybe?

    Also batch inference sort of removes the need for much of the complication in here

    """

    def __init__(
        self,
        client,
        endpoint_name: str,
        request_ids: Dict[str, str],
    ):
        self.client = client
        self.endpoint_name = endpoint_name
        self.request_ids = request_ids.copy()  # custom request_id (clientside) -> task_id (serverside)
        self.responses: Dict[str, Optional[EndpointResponse]] = {req_id: None for req_id in request_ids.keys()}
        # celery task statuses
        self.statuses: Dict[str, Optional[str]] = {req_id: TASK_PENDING_STATE for req_id in request_ids.keys()}

    def poll_endpoints(self):
        """
        Runs one round of polling the endpoint for async task results.
        """

        # TODO: replace with batch endpoint, or make requests in parallel
        # TODO: Make this private.

        def single_request(inner_url, inner_task_id):
            if self.statuses[inner_url] != TASK_PENDING_STATE:
                # Skip polling tasks that are completed
                return None
            inner_response = self.client._get_async_endpoint_response(  # pylint: disable=W0212
                self.endpoint_name, inner_task_id
            )
            print("inner response", inner_response)
            return (
                inner_url,
                inner_task_id,
                inner_response.get("status", None),
                inner_response,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            responses = executor.map(
                single_request,
                self.request_ids.keys(),
                self.request_ids.values(),
            )

        for response in responses:
            if response is None:
                continue
            url, _, status, raw_response = response
            if status:
                self.statuses[url] = status
            if raw_response:
                response_object = EndpointResponse(
                    client=self.client,
                    status=raw_response["status"],
                    result_url=raw_response.get("result_url", None),
                    result=raw_response.get("result", None),
                    traceback=raw_response.get("traceback", None),
                )
                self.responses[url] = response_object

    def is_done(self, poll=True) -> bool:
        """
        Checks the client local status to see if all requests are done.

        Parameters:
            poll: If ``True``, then this will first check the status for a subset
            of the remaining incomplete tasks on the Launch server.
        """
        # TODO: make some request to some endpoint
        if poll:
            self.poll_endpoints()
        return all(resp != TASK_PENDING_STATE for resp in self.statuses.values())

    def get_responses(self) -> Dict[str, Optional[EndpointResponse]]:
        """
        Returns a dictionary, where each key is the request_id for an EndpointRequest passed in, and the corresponding
        object at that key is the corresponding EndpointResponse.
        """
        if not self.is_done(poll=False):
            raise ValueError("Not all responses are done")
        return self.responses.copy()

    def batch_status(self):
        counter = Counter(self.statuses.values())
        return dict(counter)
