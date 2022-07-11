import logging

import launch
from launch.model_endpoint import EndpointRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_endpoint_params():
    endpoint_params = {
        # "endpoint_name": "test-endpoint",
        "cpus": 1,
        "memory": "4Gi",
        "gpus": 0,
        "min_workers": 1,
        "max_workers": 1,
        "per_worker": 1,
        # "endpoint_type": "async",
    }

    return endpoint_params


def _make_request(endpoint, endpoint_type, expected_output):
    # TODO: make this configurable
    dummy_input = "s3://scale-ml-hosted-model-inference/demos/pennfudan/FudanPed00001.png"

    if endpoint_type == "async":
        future = endpoint.predict(
            request=EndpointRequest(url=dummy_input, return_pickled=False)
        )
        response = future.get()
    elif endpoint_type == "sync":
        response = endpoint.predict(
            request=EndpointRequest(url=dummy_input, return_pickled=False)
        )

    assert response.status == "SUCCESS"
    assert (
        response.result == expected_output
    )  # question: why it's a string here?


def _edit_model_bundle(client, endpoint, endpoint_type):
    # edit model bundle
    # TODO test_returns_2 was created beforehand using internal client to bypass the review process
    returns2_bundle = client.get_model_bundle("test_returns_2")
    client.edit_model_endpoint(
        endpoint_name=endpoint.model_endpoint.name,
        model_bundle=returns2_bundle,
    )

    _make_request(endpoint, endpoint_type, expected_output="2")


def _test_model_endpoint(client, endpoint_type):

    endpoint_params = get_endpoint_params()
    endpoint_name = f"test-endpoint-{endpoint_type}"
    endpoint_params["endpoint_name"] = endpoint_name  # TODO: add a timestamp
    endpoint_params["endpoint_type"] = endpoint_type

    # TODO test_returns_1 was created beforehand using internal client to bypass the review process
    bundle = launch.ModelBundle(name="test_returns_1")

    logger.info(f"creating model endpoint {endpoint_name} ...")
    endpoint = client.create_model_endpoint(
        model_bundle=bundle, **endpoint_params
    )
    logger.info(
        f"successfully created {endpoint_type} model endpoint {endpoint_name}"
    )

    try:
        logger.info(f"sending request to model endpoint {endpoint_name}")
        _make_request(endpoint, endpoint_type, expected_output="1")
        logger.info(f"got response from model endpoint {endpoint_name}")

        _edit_model_bundle(client, endpoint, endpoint_type)

    finally:
        # delete the endpoint
        assert (
            client.delete_model_endpoint(endpoint.model_endpoint)
            == "true"
        )
        logger.info(f"successfully deleted model endpoint {endpoint_name}")


def test_model_endpoint(client):
    _test_model_endpoint(client, endpoint_type="async")
    # TODO uncomment this when sync endpoint is ready for testing
    # _test_model_endpoint(client, endpoint_type="sync")