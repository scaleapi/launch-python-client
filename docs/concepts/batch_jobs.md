# Batch Jobs

For predicting over a larger set of tasks (> 50) at once, it is recommended to
use batch jobs. Batch jobs are a way to send a large number of tasks to a model
bundle. The tasks are processed in parallel, and the results are returned as a
list of predictions.

Batch jobs are created using the
[`batch_async_request`](/api/client/#launch.client.LaunchClient.batch_async_request)
method of the
[`LaunchClient`](/api/client/#launch.client.LaunchClient).

```py title="Creating and Following a Batch Job"
import os
import time
from launch import LaunchClient

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
batch_job = client.batch_async_request(
    model_bundle="test-bundle",
    inputs=[
        {"x": 2, "y": "hello"},
        {"x": 3, "y": "world"},
    ],
    labels={
        "team": "MY_TEAM",
        "product": "MY_PRODUCT",
    }
)

status = "PENDING"
res = None
while status != "SUCCESS" and status != "FAILURE" and status != "CANCELLED":
    time.sleep(30)
    res = client.get_batch_async_response(batch_job["job_id"])
    status = res["status"]
    print(f"the batch job is {status}")

print(res)
```
