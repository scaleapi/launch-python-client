import json
from datetime import datetime
from unittest.mock import MagicMock
from urllib3 import HTTPResponse

import requests
import requests_mock

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
