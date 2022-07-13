import pytest
import os

import launch

from scaleml.utils.logging import logger_name, make_logger

logger = make_logger(logger_name())


def returns_returns_1(x):
    def returns_1(y):
        return 1

    return returns_1


def returns_returns_2(x):
    def returns_2(y):
        return 2

    return returns_2


ENV_PARAMS = {
    "framework_type": "pytorch",
    "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
}


BUNDLE_PARAMS = {
    "model_bundle_name": "test-returns-1",
    "model": 1,
    "load_predict_fn": returns_returns_1,
    "env_params": ENV_PARAMS,
    "requirements": [],
}


BUNDLE_PARAMS_2 = {
    "model_bundle_name": "test-returns-2",
    "model": 1,
    "load_predict_fn": returns_returns_2,
    "env_params": ENV_PARAMS,
    "requirements": [],
}


@pytest.fixture()
def client():
    # to run this test locally, you should add the API KEY to your environment variables
    launch_test_api_key = os.getenv("LAUNCH_TEST_API_KEY")
    client = launch.LaunchClient(api_key=launch_test_api_key)

    # these two model bundles are used in test_endpoints.py
    # please do not delete them
    # client.create_model_bundle(**BUNDLE_PARAMS)
    # client.create_model_bundle(**BUNDLE_PARAMS_2)

    return client