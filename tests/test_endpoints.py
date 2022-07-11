import logging
import time

import launch
from launch.model_endpoint import EndpointRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


ENDPOINT_PARAMS = {
    # "endpoint_name": "test-endpoint",
    "cpus": 1,
    "memory": "4Gi",
    "gpus": 0,
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    # "endpoint_type": "async",
}


def _make_request(endpoint, endpoint_type, expected_output):
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
    returns2_bundle = client.get_model_bundle("test-returns-2")
    client.edit_model_endpoint(
        model_endpoint=endpoint.model_endpoint.name,
        model_bundle=returns2_bundle,
    )

    _make_request(endpoint, endpoint_type, expected_output="2")


def _test_model_endpoint(client, endpoint_type):

    endpoint_name = f"test-endpoint-{endpoint_type}"
    ENDPOINT_PARAMS["endpoint_name"] = '-'.join([endpoint_name, str(int(time.time()))])
    ENDPOINT_PARAMS["endpoint_type"] = endpoint_type

    bundle = launch.ModelBundle(name="test-returns-1")

    logger.info(f"creating model endpoint {endpoint_name} ...")
    endpoint = client.create_model_endpoint(
        model_bundle=bundle, **ENDPOINT_PARAMS
    )
    logger.info(
        f"successfully created {endpoint_type} model endpoint {endpoint_name}"
    )

    try:
        logger.info(f"sending request to model endpoint {endpoint_name}")
        _make_request(endpoint, endpoint_type, expected_output="1")
        logger.info(f"got response from model endpoint {endpoint_name}")

        # TODO edit other params
        _edit_model_bundle(client, endpoint, endpoint_type)

    finally:
        # delete the endpoint
        assert (
            client.delete_model_endpoint(endpoint.model_endpoint)
            == "true"
        )
        logger.info(f"successfully deleted model endpoint {endpoint_name}")


def test_model_endpoint(client):
    # NOTE this will take some time
    _test_model_endpoint(client, endpoint_type="async")
    # TODO uncomment this when sync endpoint is ready for testing
    # _test_model_endpoint(client, endpoint_type="sync")