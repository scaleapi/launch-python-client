import logging
import os
import shutil
import tempfile
import time

import pytest

from launch.errors import APIError
from .conftest import BUNDLE_PARAMS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    bundle_name = '-'.join(["tmp-bundle-from-dirs", str(int(time.time()))])

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
    logger.info(f"successfully created model bundle {bundle_name}")

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
    logger.info(f"successfully deleted model bundle {bundle_name}")


def test_model_bundle(client):
    # create a bundle
    bundle_name = '-'.join(["tmp-bundle", str(int(time.time()))])
    BUNDLE_PARAMS["model_bundle_name"] = bundle_name
    bundle = client.create_model_bundle(**BUNDLE_PARAMS)
    logger.info(f"successfully created model bundle {bundle_name}")

    # create a bundle with the same name - this should error out
    with pytest.raises(APIError):
        bundle = client.create_model_bundle(**BUNDLE_PARAMS)

    # delete the bundle
    assert client.delete_model_bundle(bundle) == "true"
    logger.info(f"successfully deleted model bundle {bundle_name}")