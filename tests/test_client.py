import io
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
def mock_client():
    # TODO: change API_KEY
    # tianwei_api_key = (
    #     "scaleint_b03c26486af74848a237e0df9f6971b0|61a67d767bce560024c7eb96"
    # )
    # # client = launch.LaunchClient(api_key="test")
    # client = launch.LaunchClient(api_key=tianwei_api_key)

    # copied from hosted_model_inference/tests/integration/endpoint_builder/client_e2e.py
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


# def test_create_model_bundle_from_dirs_bundle_contents_correct(
#     mock_client, requests_mock, fake_project_dir  # noqa: F811
# ):
#     def check_bundle_upload_data(request):
#         request_body = request._request.body
#         zf = ZipFile(io.BytesIO(request_body), "r")
#         try:
#             actual_zip_filenames = set(zf.namelist())
#             assert actual_zip_filenames == set(
#                 ["my_module1/requirements.txt", "my_module2/foobar.txt"]
#             )
#             return True
#         finally:
#             zf.close()

#     requests_mock.post(
#         "https://api.scale.com/v1/hosted_inference/model_bundle_upload",
#         json={
#             "signedUrl": "s3://my-fake-bucket/path/to/bundle",
#             "bucket": "my-fake-bucket",
#             "key": "path/to/bundle",
#         },
#     )
#     requests_mock.put(
#         "s3://my-fake-bucket/path/to/bundle",
#         additional_matcher=check_bundle_upload_data,
#     )
#     requests_mock.post(
#         "https://api.scale.com/v1/hosted_inference/model_bundle", json={}
#     )

#     mock_client.create_model_bundle_from_dirs(
#         model_bundle_name="my_test_bundle",
#         base_paths=[
#             os.path.join(fake_project_dir, "project_root/my_module1"),
#             os.path.join(fake_project_dir, "project_root/my_module2"),
#         ],
#         requirements_path=os.path.join(
#             fake_project_dir, "project_root/my_module1/requirements.txt"
#         ),
#         env_params={},
#         load_predict_fn_module_path="a.b.c",
#         load_model_fn_module_path="a.b.c",
#         app_config=None,
#     )


@pytest.fixture()
def returns_returns_1(x):
    def returns_1(y):
        return 1

    return returns_1


@pytest.fixture()
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


@pytest.fixture()
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


def test_model_bundle(mock_client, get_bundle_params):

    bundle = mock_client.create_model_bundle(**get_bundle_params)
    print("successfully created model bundle tmp-bundle")

    # create a bundle with the same name - this should error out
    with pytest.raises(APIError):
        bundle = mock_client.create_model_bundle(**get_bundle_params)

    # delete the bundle
    assert mock_client.delete_model_bundle(bundle) == "true"
    print("successfully deleted model bundle tmp-bundle")


def _test_model_endpoint(mock_client, get_endpoint_params, endpoint_type):

    endpoint_params = get_endpoint_params
    endpoint_name = f"test-endpoint-{endpoint_type}"
    endpoint_params["endpoint_name"] = endpoint_name
    endpoint_params["endpoint_type"] = endpoint_type

    # test-bundle was created beforehand using internal client to bypass the review process
    bundle = launch.ModelBundle(name="test-bundle")

    print(f"creating model endpoint {endpoint_name} ...")
    endpoint = mock_client.create_model_endpoint(
        model_bundle=bundle, **endpoint_params
    )
    print(
        f"successfully created {endpoint_type} model endpoint {endpoint_name}"
    )

    print("sleeping for 30s")
    time.sleep(30)

    try:
        print(f"sending request to model endpoint {endpoint_name}")
        _make_request(endpoint, endpoint_type)
        print(f"got response from model endpoint {endpoint_name}")

    finally:
        # delete the endpoint
        assert (
            mock_client.delete_model_endpoint(endpoint.model_endpoint)
            == "true"
        )
        print(f"successfully deleted model endpoint {endpoint_name}")


def _make_request(endpoint, endpoint_type):
    # TODO: make this configurable
    image_url = "s3://scale-ml-hosted-model-inference/demos/pennfudan/FudanPed00001.png"

    if endpoint_type == "async":
        future = endpoint.predict(
            request=EndpointRequest(url=image_url, return_pickled=False)
        )
        response = future.get()
    elif endpoint_type == "sync":
        response = endpoint.predict(
            request=EndpointRequest(url=image_url, return_pickled=False)
        )

    assert response.status == "SUCCESS"
    assert response.result == "1"  # question: why it's a string here?


def test_model_endpoint(mock_client, get_endpoint_params):
    # _test_model_endpoint(
    #     mock_client, get_endpoint_params, endpoint_type="async"
    # )
    _test_model_endpoint(
        mock_client, get_endpoint_params, endpoint_type="sync"
    )
