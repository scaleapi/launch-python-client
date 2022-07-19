import json

import requests
import requests_mock

import launch


def _get_mock_client():
    client = launch.LaunchClient(api_key="test")
    return client


def test_status_returns_updated_value(requests_mock):  # noqa: F811
    client = _get_mock_client()

    json_str = """{
        "bundle_name": "test-returns-1",
        "configs": {
            "app_config": null,
            "endpoint_config": {
                "bundle_name": "test-returns-1",
                "endpoint_name": "test-endpoint",
                "post_inference_hooks": null
            }
        },
        "destination": "launch.xxx",
        "endpoint_type": "async",
        "metadata": null,
        "name": "test-endpoint",
        "resource_settings": {
            "cpus": "2",
            "gpu_type": null,
            "gpus": 0,
            "memory": "4Gi"
        },
        "worker_settings": {
            "available_workers": 1,
            "max_workers": "1",
            "min_workers": "1",
            "per_worker": "1",
            "unavailable_workers": 0
        }
    }"""
    resp = json.loads(json_str)
    update_pending_resp = {**resp, "status": "UPDATE_PENDING"}
    update_in_progress_resp = {**resp, "status": "UPDATE_IN_PROGRESS"}
    ready_resp = {**resp, "status": "READY"}

    requests_mock.get(
        "https://api.scale.com/v1/hosted_inference/endpoints/test-endpoint",
        [
            {"json": update_pending_resp},
            {"json": update_in_progress_resp},
            {"json": ready_resp},
        ],
    )

    endpoint = client.get_model_endpoint("test-endpoint")
    assert endpoint.status() == "UPDATE_IN_PROGRESS"
    assert endpoint.status() == "READY"
