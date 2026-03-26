import json
from datetime import datetime
from unittest.mock import MagicMock

import requests
import requests_mock
from urllib3 import HTTPResponse

import launch
from launch.api_client.api_client import ApiResponseWithoutDeserialization


def _get_mock_client():
    client = launch.LaunchClient(api_key="test")
    return client


def test_status_returns_updated_value(requests_mock):  # noqa: F811
    client = _get_mock_client()

    resp = dict(
        model_endpoints=[
            dict(
                bundle_name="test-returns-1",
                configs=dict(
                    app_config=None,
                    endpoint_config=dict(
                        bundle_name="test-returns-1",
                        endpoint_name="test-endpoint",
                        post_inference_hooks=None,
                    ),
                ),
                destination="launch.xxx",
                endpoint_type="async",
                metadata={},
                name="test-endpoint",
                resource_state=dict(
                    cpus="2",
                    gpus=0,
                    memory="4Gi",
                ),
                deployment_state=dict(
                    available_workers=1,
                    max_workers=1,
                    min_workers=1,
                    per_worker=1,
                    unavailable_workers=0,
                ),
                status="UPDATE_PENDING",
                created_at=str(datetime.now()),
                last_updated_at=str(datetime.now()),
                created_by="test",
                id="test",
            )
        ]
    )

    mock_api_client = MagicMock()
    launch.client.DefaultApi = MagicMock(return_value=mock_api_client)
    launch.model_endpoint.DefaultApi = MagicMock(return_value=mock_api_client)
    mock_api_client.list_model_endpoints_v1_model_endpoints_get.return_value = ApiResponseWithoutDeserialization(
        response=HTTPResponse(body=json.dumps(resp), status=200)
    )
    endpoint = client.get_model_endpoint("test-endpoint")  # UPDATE_PENDING
    assert endpoint.status() == "UPDATE_PENDING"

    resp["model_endpoints"][0]["status"] = "UPDATE_IN_PROGRESS"
    mock_api_client.list_model_endpoints_v1_model_endpoints_get.return_value = ApiResponseWithoutDeserialization(
        response=HTTPResponse(body=json.dumps(resp), status=200)
    )
    assert endpoint.status() == "UPDATE_IN_PROGRESS"

    resp["model_endpoints"][0]["status"] = "SUCCESS"
    mock_api_client.list_model_endpoints_v1_model_endpoints_get.return_value = ApiResponseWithoutDeserialization(
        response=HTTPResponse(body=json.dumps(resp), status=200)
    )
    assert endpoint.status() == "SUCCESS"


def test_endpoint_response_future_failure_preserves_result():
    """FAILURE responses should expose the result string (stringified exception) from the server."""
    mock_client = MagicMock()
    mock_client._get_async_endpoint_response.return_value = {
        "status": "FAILURE",
        "result": "Out of memory",
        "traceback": "Traceback (most recent call last): ...\nRuntimeError: Out of memory",
        "status_code": 500,
    }

    from launch.model_endpoint import EndpointResponseFuture

    future = EndpointResponseFuture(mock_client, "test-endpoint", "task-123")
    response = future.get()

    assert response.status == "FAILURE"
    assert response.status_code == 500
    assert response.result == "Out of memory"
    assert response.traceback == "Traceback (most recent call last): ...\nRuntimeError: Out of memory"
    assert response.result_url is None


def test_endpoint_response_future_failure_no_result_body():
    """FAILURE responses with no result body should still work (e.g. pod crash with no response)."""
    mock_client = MagicMock()
    mock_client._get_async_endpoint_response.return_value = {
        "status": "FAILURE",
        "result": None,
        "traceback": None,
        "status_code": 500,
    }

    from launch.model_endpoint import EndpointResponseFuture

    future = EndpointResponseFuture(mock_client, "test-endpoint", "task-456")
    response = future.get()

    assert response.status == "FAILURE"
    assert response.result is None
    assert response.traceback is None
