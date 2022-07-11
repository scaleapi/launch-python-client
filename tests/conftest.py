import logging
import pytest
import os

import launch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture()
def client():
    # To run this test locally, you should add the API KEY to your environment variables
    launch_test_api_key = os.getenv("LAUNCH_TEST_API_KEY")
    client = launch.LaunchClient(api_key=launch_test_api_key)
    return client