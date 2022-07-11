import logging
import time
import launch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_batch_job(client):
    dummy_input = "s3://scale-ml-hosted-model-inference/demos/pennfudan/FudanPed00001.png"

    urls = [dummy_input] * 100
    batch_job = client.batch_async_request("test-returns-1", urls=urls)
    logger.info(f"created batch job {batch_job}")

    logger.info("waiting for the batch job to complete ...")
    status = "SUBMITTED"
    while status != "COMPLETED" and status != "FAILED":
        time.sleep(30)
        status = client.get_batch_async_response(batch_job)['status']
        logger.info(f"the batch job is {status}")

    # TODO check result as well
    assert status == "COMPLETED"
