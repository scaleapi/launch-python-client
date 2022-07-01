import io
import logging
import os
import shutil
import tempfile
import time
from zipfile import ZipFile

import pytest
import requests
import requests_mock
import smart_open
from boto3 import Session

import launch
from launch.errors import APIError
from launch.model_endpoint import EndpointRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _upload_bundle_fn(serialized_bundle, bundle_url):
    # transport_params = {"session": Session(profile_name="ml-worker")}  # smart-open < 5
    transport_params = {
        "client": Session(profile_name="ml-worker").client("s3")
    }  # smart-open 5
    with smart_open.open(
        bundle_url, "wb", transport_params=transport_params
    ) as f:
        f.write(serialized_bundle)


def _endpoint_auth_fn(payload):
    payload["aws_role"] = "ml-worker"
    payload["results_s3_bucket"] = "scale-ml"
    return payload


def _bundle_location_fn():
    bundle_name = str(int(time.time()))
    return f"s3://scale-ml/tmp/launch-test/bundle/{bundle_name}"


@pytest.fixture()
def launch_client():
    # TODO: change API_KEY
    internal_endpoint = "http://hostedinference.ml-staging-internal.scale.com"
    tianwei_user = "61a67d767bce560024c7eb96"
    # use self_hosted version so that the client can access s3 scale-ml bucket
    client = launch.LaunchClient(
        api_key=tianwei_user, endpoint=internal_endpoint, self_hosted=True
    )
    client.register_upload_bundle_fn(_upload_bundle_fn)
    client.register_endpoint_auth_decorator(_endpoint_auth_fn)
    client.register_bundle_location_fn(_bundle_location_fn)
    return client


@pytest.fixture()
def fake_project_dir():
    tmpdir = tempfile.mkdtemp()
    logger.info(f"tmpdir: {tmpdir}")

    try:
        os.mkdir(os.path.join(tmpdir, "project_root"))
        os.mkdir(os.path.join(tmpdir, "project_root", "my_module1"))

        with open(
            os.path.join(
                tmpdir, "project_root", "my_module1", "requirements.txt"
            ),
            "w",
        ):
            pass

        os.mkdir(os.path.join(tmpdir, "project_root", "my_module2"))
        with open(
            os.path.join(tmpdir, "project_root", "my_module2", "foobar.txt"),
            "w",
        ):
            pass
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


def test_model_bundle_from_dirs(
    # mock_client, requests_mock, fake_project_dir  # noqa: F811
    launch_client,
    fake_project_dir,  # noqa: F811
):
    # TODO: change this to a valid dirs like in the client_e2e
    bundle_name = "tmp-bundle-from-dirs"

    bundle = launch_client.create_model_bundle_from_dirs(
        model_bundle_name=bundle_name,
        base_paths=[
            os.path.join(fake_project_dir, "project_root/my_module1"),
            os.path.join(fake_project_dir, "project_root/my_module2"),
        ],
        requirements_path=os.path.join(
            fake_project_dir, "project_root/my_module1/requirements.txt"
        ),
        env_params={},
        load_predict_fn_module_path="a.b.c",
        load_model_fn_module_path="a.b.c",
        app_config=None,
    )

    assert launch_client.delete_model_bundle(bundle) == "true"
    logger.info("successfully deleted model bundle {bundle_name}")


# @pytest.fixture()
def returns_returns_1(x):
    def returns_1(y):
        return 1

    return returns_1


# @pytest.fixture()
def get_bundle_params():
    # copied from hosted_model_inference/tests/integration/endpoint_builder/client_e2e.py
    env_params = {
        "framework_type": "pytorch",
        "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",  # python version was 3.7.11 in a later image gg
    }
    # TODO: make sure the args covers what we'd like to test

    bundle_params = {
        "model_bundle_name": "tmp-bundle",
        "model": 1,
        "load_predict_fn": returns_returns_1,
        "env_params": env_params,
        "requirements": [],
    }

    return bundle_params


# @pytest.fixture()
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


def _edit_endpoint(client, endpoint, endpoint_type):
    # edit model bundle
    # test_returns_2 was created beforehand using internal client to bypass the review process
    returns2_bundle = client.get_model_bundle("test_returns_2")
    client.edit_model_endpoint(
        endpoint_name=endpoint.model_endpoint.name,
        model_bundle=returns2_bundle,
    )
    _make_request(endpoint, endpoint_type, expected_output="2")


def _test_model_endpoint(launch_client, endpoint_type):

    endpoint_params = get_endpoint_params()
    endpoint_name = f"test-endpoint-{endpoint_type}"
    endpoint_params["endpoint_name"] = endpoint_name  # TODO: add a timestamp
    endpoint_params["endpoint_type"] = endpoint_type

    # test_returns_1 was created beforehand using internal client to bypass the review process
    bundle = launch.ModelBundle(name="test_returns_1")

    logger.info(f"creating model endpoint {endpoint_name} ...")
    endpoint = launch_client.create_model_endpoint(
        model_bundle=bundle, **endpoint_params
    )
    logger.info(
        f"successfully created {endpoint_type} model endpoint {endpoint_name}"
    )

    logger.info("sleeping for 30s")
    time.sleep(30)

    try:
        logger.info(f"sending request to model endpoint {endpoint_name}")
        _make_request(endpoint, endpoint_type, expected_output="1")
        logger.info(f"got response from model endpoint {endpoint_name}")

        _edit_endpoint(launch_client, endpoint, endpoint_type)

    finally:
        # delete the endpoint
        assert (
            launch_client.delete_model_endpoint(endpoint.model_endpoint)
            == "true"
        )
        logger.info(f"successfully deleted model endpoint {endpoint_name}")


def test_model_bundle(launch_client):

    bundle_params = get_bundle_params()
    bundle = launch_client.create_model_bundle(**bundle_params)
    logger.info("successfully created model bundle tmp-bundle")

    # create a bundle with the same name - this should error out
    with pytest.raises(APIError):
        bundle = launch_client.create_model_bundle(**bundle_params)

    # delete the bundle
    assert launch_client.delete_model_bundle(bundle) == "true"
    logger.info("successfully deleted model bundle tmp-bundle")


def test_model_endpoint(launch_client):
    _test_model_endpoint(launch_client, endpoint_type="async")
    # TODO: currently blocked
    # _test_model_endpoint(launch_client, endpoint_type="sync")


# def test_main(fake_project_dir):
#     # TODO: add a acript for setting up the test client
#     client = launch_client()
#     _test_model_bundle_from_dirs(client, fake_project_dir)

# _test_model_bundle(client)

# _test_model_endpoint(
#     client, endpoint_type="async"
# )
# _test_model_endpoint(
#     client, get_endpoint_params, endpoint_type="sync"
# )
