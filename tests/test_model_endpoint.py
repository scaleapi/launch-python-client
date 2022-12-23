import json
from datetime import datetime
from unittest.mock import MagicMock

import requests
import requests_mock

import launch
from launch.api_client.model.get_model_endpoint_response import (
    GetModelEndpointResponse,
)
from launch.api_client.model.list_model_endpoints_response import (
    ListModelEndpointsResponse,
)
from launch.api_client.model.model_endpoint_deployment_state import (
    ModelEndpointDeploymentState,
)
from launch.api_client.model.model_endpoint_resource_state import (
    ModelEndpointResourceState,
)
from launch.api_client.model.model_endpoint_status import ModelEndpointStatus
from launch.api_client.model.model_endpoint_type import ModelEndpointType


def _get_mock_client():
    client = launch.LaunchClient(api_key="test")
    return client


def test_status_returns_updated_value(requests_mock):  # noqa: F811
    client = _get_mock_client()

    resp = GetModelEndpointResponse(
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
        endpoint_type=ModelEndpointType("async"),
        metadata={},
        name="test-endpoint",
        resource_state=ModelEndpointResourceState(
            cpus="2",
            gpus=0,
            memory="4Gi",
        ),
        deployment_state=ModelEndpointDeploymentState(
            available_workers=1,
            max_workers=1,
            min_workers=1,
            per_worker=1,
            unavailable_workers=0,
        ),
        status=ModelEndpointStatus("UPDATE_PENDING"),
        created_at=datetime.now(),
        last_updated_at=datetime.now(),
        created_by="test",
        id="test",
    )

    mock_api_client = MagicMock()
    launch.client.DefaultApi = MagicMock(return_value=mock_api_client)
    launch.model_endpoint.DefaultApi = MagicMock(return_value=mock_api_client)
    mock_api_client.list_model_endpoints_v1_model_endpoints_get.return_value = ListModelEndpointsResponse(
        model_endpoints=[dict(resp)],
    )
    endpoint = client.get_model_endpoint("test-endpoint")  # UPDATE_PENDING
    assert endpoint.status() == "UPDATE_PENDING"
