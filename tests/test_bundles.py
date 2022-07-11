import logging
import os
import shutil
import tempfile

import pytest

from launch.errors import APIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def returns_returns_1(x):
    def returns_1(y):
        return 1

    return returns_1


ENV_PARAMS = {
    "framework_type": "pytorch",
    "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
}

BUNDLE_PARAMS = {
    "model_bundle_name": "tmp-bundle",
    "model": 1,
    "load_predict_fn": returns_returns_1,
    "env_params": ENV_PARAMS,
    "requirements": [],
}

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


def test_model_bundle_from_dirs(client, fake_project_dir):
    bundle_name = "tmp-bundle-from-dirs"

    # create a bundle
    bundle = client.create_model_bundle_from_dirs(
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

    # create a bundle with the same name - this should error out
    with pytest.raises(APIError):
        bundle = client.create_model_bundle_from_dirs(
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

    # delete the bundle
    assert client.delete_model_bundle(bundle) == "true"
    logger.info("successfully deleted model bundle {bundle_name}")


def test_model_bundle(client):
    # create a bundle
    bundle = client.create_model_bundle(**BUNDLE_PARAMS)
    logger.info("successfully created model bundle tmp-bundle")

    # create a bundle with the same name - this should error out
    with pytest.raises(APIError):
        bundle = client.create_model_bundle(**BUNDLE_PARAMS)

    # delete the bundle
    assert client.delete_model_bundle(bundle) == "true"
    logger.info("successfully deleted model bundle tmp-bundle")