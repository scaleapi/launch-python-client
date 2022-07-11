import concurrent.futures
import json
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from dataclasses_json import Undefined, dataclass_json
from deprecation import deprecated

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

    bundle_name: Optional[str] = None
    """
    The name of the bundle for the endpoint. The owner of the bundle must be the same as the owner for the endpoint.
    """

    status: Optional[str] = None
    """
    The status of the endpoint.
    """

    resource_settings: Optional[dict] = None
    """
    Resource settings for the endpoint.
    """

    worker_settings: Optional[dict] = None
    """
    Worker settings for the endpoint.
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

    def __repr__(self):
        return f"ModelEndpoint(name='{self.name}', bundle_name='{self.bundle_name}', status='{self.status}', resource_settings='{json.dumps(self.resource_settings)}', worker_settings='{json.dumps(self.worker_settings)}', endpoint_type='{self.endpoint_type}', metadata='{self.metadata}')"


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

        request_id: (deprecated) A user-specifiable id for requests.
            Should be unique among EndpointRequests made in the same batch call.
            If one isn't provided the client will generate its own.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        args: Optional[Dict] = None,
        return_pickled: Optional[bool] = True,
        request_id: Optional[str] = None,
    ):
        # TODO: request_id is pretty much here only to support the clientside AsyncEndpointBatchResponse
        # so it should be removed when we get proper batch endpoints working.
        validate_task_request(url=url, args=args)
        if request_id is None:
            request_id = str(uuid.uuid4())
        self.url = url
        self.args = args
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

            result_url: A string that is a url containing the pickled python object from the Endpoint's predict function.

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
        return f"status: {self.status}, result: {self.result}, result_url: {self.result_url}, traceback: {self.traceback}"


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

    def get(self) -> EndpointResponse:
        """
        Retrieves the ``EndpointResponse`` for the prediction request after it completes. This method blocks.
        """
        while True:
            async_response = self.client._get_async_endpoint_response(  # pylint: disable=W0212
                self.endpoint_name, self.async_task_id
            )
            if async_response["state"] == "PENDING":
                time.sleep(2)
            else:
                if async_response["state"] == "SUCCESS":
                    return EndpointResponse(
                        client=self.client,
                        status=async_response["state"],
                        result_url=async_response.get("result_url", None),
                        result=async_response.get("result", None),
                        traceback=None,
                    )
                elif async_response["state"] == "FAILURE":
                    return EndpointResponse(
                        client=self.client,
                        status=async_response["state"],
                        result_url=None,
                        result=None,
                        traceback=async_response.get("traceback", None),
                    )
                else:
                    raise ValueError(
                        f"Unrecognized state: {async_response['state']}"
                    )


class Endpoint:
    """An abstract class that represent any kind of endpoints in Scale Launch"""

    def __init__(self, model_endpoint: ModelEndpoint):
        self.model_endpoint = model_endpoint


class SyncEndpoint(Endpoint):
    """
    A synchronous model endpoint.
    """

    def __init__(self, model_endpoint: ModelEndpoint, client):
        super().__init__(model_endpoint=model_endpoint)
        self.client = client

    def __str__(self):
        return f"SyncEndpoint <endpoint_name:{self.model_endpoint.name}>"

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
        return EndpointResponse(
            client=self.client,
            status=raw_response.get("state"),
            result_url=raw_response.get("result_url", None),
            result=raw_response.get("result", None),
            traceback=raw_response.get("traceback", None),
        )

    def status(self):
        """Gets the status of the Endpoint.

        TODO: Implement this by leveraging the LaunchClient object.
        """
        raise NotImplementedError


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
        super().__init__(model_endpoint=model_endpoint)
        self.client = client

    def __str__(self):
        return f"AsyncEndpoint <endpoint_name:{self.model_endpoint.name}>"

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
        async_task_id = self.client._async_request(  # pylint: disable=W0212
            self.model_endpoint.name,
            url=request.url,
            args=request.args,
            return_pickled=request.return_pickled,
        )
        return EndpointResponseFuture(
            client=self.client,
            endpoint_name=self.model_endpoint.name,
            async_task_id=async_task_id,
        )

    @deprecated
    def predict_batch(
        self, requests: Sequence[EndpointRequest]
    ) -> "AsyncEndpointBatchResponse":
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

        if len(requests) != len(
            set(request.request_id for request in requests)
        ):
            raise ValueError("Request_ids in a batch must be unique")

        def single_request(request):
            # request has keys url and args

            inner_inference_request = (
                self.client._async_request(  # pylint: disable=W0212
                    endpoint_name=self.model_endpoint.name,
                    url=request.url,
                    args=request.args,
                    return_pickled=request.return_pickled,
                )
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

    def status(self):
        """Gets the status of the Endpoint.

        TODO: Implement this by leveraging the LaunchClient object.
        """
        raise NotImplementedError


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
        self.request_ids = (
            request_ids.copy()
        )  # custom request_id (clientside) -> task_id (serverside)
        self.responses: Dict[str, Optional[EndpointResponse]] = {
            req_id: None for req_id in request_ids.keys()
        }
        # celery task statuses
        self.statuses: Dict[str, Optional[str]] = {
            req_id: TASK_PENDING_STATE for req_id in request_ids.keys()
        }

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
                inner_response.get("state", None),
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
            url, _, state, raw_response = response
            if state:
                self.statuses[url] = state
            if raw_response:
                response_object = EndpointResponse(
                    client=self.client,
                    status=raw_response["state"],
                    result_url=raw_response.get("result_url", None),
                    result=raw_response.get("result", None),
                    traceback=raw_response.get("traceback", None),
                )
                self.responses[url] = response_object

    def is_done(self, poll=True) -> bool:
        """
        Checks the client local state to see if all requests are done.

        Parameters:
            poll: If ``True``, then this will first check the state for a subset
            of the remaining incomplete tasks on the Launch server.
        """
        # TODO: make some request to some endpoint
        if poll:
            self.poll_endpoints()
        return all(
            resp != TASK_PENDING_STATE for resp in self.statuses.values()
        )

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
