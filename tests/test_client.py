import io
import os
import shutil
import tempfile
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest
import requests
import requests_mock

import launch
from launch.api_client.model.list_model_endpoints_response import (
    ListModelEndpointsResponse,
)


def _get_mock_client():
    client = launch.LaunchClient(api_key="test")
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


def test_create_model_bundle_from_dirs_bundle_contents_correct(
    requests_mock, fake_project_dir  # noqa: F811
):
    def check_bundle_upload_data(request):
        request_body = request._request.body
        zf = ZipFile(io.BytesIO(request_body), "r")
        try:
            actual_zip_filenames = set(zf.namelist())
            assert actual_zip_filenames == set(
                ["my_module1/requirements.txt", "my_module2/foobar.txt"]
            )
            return True
        finally:
            zf.close()

    requests_mock.post(
        "https://api.scale.com/v1/hosted_inference/model_bundle_upload",
        json={
            "signedUrl": "s3://my-fake-bucket/path/to/bundle",
            "bucket": "my-fake-bucket",
            "key": "path/to/bundle",
        },
    )
    requests_mock.put(
        "s3://my-fake-bucket/path/to/bundle",
        additional_matcher=check_bundle_upload_data,
    )
    launch.client.DefaultApi = MagicMock()

    client = _get_mock_client()
    client.create_model_bundle_from_dirs(
        model_bundle_name="my_test_bundle",
        base_paths=[
            os.path.join(fake_project_dir, "project_root/my_module1"),
            os.path.join(fake_project_dir, "project_root/my_module2"),
        ],
        requirements_path=os.path.join(
            fake_project_dir, "project_root/my_module1/requirements.txt"
        ),
        env_params={
            "framework_type": "pytorch",
            "pytorch_image_tag": "1.10.0-cuda11.3-cudnn8-runtime",
        },
        load_predict_fn_module_path="a.b.c",
        load_model_fn_module_path="a.b.c",
        app_config=None,
    )


def test_get_non_existent_model_endpoint(requests_mock):  # noqa: F811
    client = _get_mock_client()
    mock_api_client = MagicMock()
    mock_api_client.list_model_endpoints_v1_model_endpoints_get.return_value = ListModelEndpointsResponse(
        model_endpoints=[]
    )
    launch.client.DefaultApi = MagicMock(return_value=mock_api_client)
    endpoint = client.get_model_endpoint("non-existent-endpoint")
    assert endpoint is None
