# Model Endpoints

Model Endpoints are deployments of models that can receive requests and return
predictions containing the results of the model's inference. Each model endpoint
is associated with a model bundle, which contains the model's code. An endpoint
specifies deployment parameters, such as the minimum and maximum number of
workers, as well as the requested resources for each worker, such as the number
of CPUs, amount of memory, GPU count, and type of GPU.

Endpoints can be asynchronous or synchronous. Asynchronous endpoints return
a future immediately after receiving a request, and the future can be used to
retrieve the prediction once it is ready. Synchronous endpoints return the
prediction directly after receiving a request.

## Creating Async Model Endpoints

Async model endpoints are the most cost-efficient way to perform inference on
tasks that are less latency-sensitive.

```py title="Creating an Async Model Endpoint"
import os
from launch import LaunchClient

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
endpoint = client.create_model_endpoint(
    endpoint_name="demo-endpoint-async",
    model_bundle="test-bundle",
    cpus=1,
    min_workers=0,
    endpoint_type="async",
    update_if_exists=True,
    labels={
        "team": "infra",
        "product": "MY_PRODUCT"
    },
)
```

## Creating Sync Model Endpoints

Sync model endpoints are useful for latency-sensitive tasks, such as real-time
inference. Sync endpoints are more expensive than async endpoints.
!!! Note
    Sync model endpoints require at least 1 `min_worker`.

```py title="Creating a Sync Model Endpoint"
import os
from launch import LaunchClient

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
endpoint = client.create_model_endpoint(
    endpoint_name="demo-endpoint-sync",
    model_bundle="test-bundle",
    cpus=1,
    min_workers=1,
    endpoint_type="sync",
    update_if_exists=True,
    labels={
        "team": "infra",
        "product": "MY_PRODUCT"
    },
)
```

## Managing Model Endpoints

Model endpoints can be listed, updated, and deleted using the Launch API.

```py title="Listing Model Endpoints"
import os
from launch import LaunchClient

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
endpoints = client.list_model_endpoints()
```

```py title="Updating a Model Endpoint"
import os
from launch import LaunchClient

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
client.edit_model_endpoint(
    model_endpoint="demo-endpoint",
    max_workers=2,
)
```

```py title="Deleting a Model Endpoint"
import os
from launch import LaunchClient

client = LaunchClient(api_key=os.getenv("LAUNCH_API_KEY"))
endpoint = client.create_model_endpoint(
    endpoint_name="demo-endpoint-tmp",
    model_bundle="test-bundle",
    cpus=1,
    min_workers=0,
    endpoint_type="async",
    update_if_exists=True,
    labels={
        "team": "infra",
        "product": "MY_PRODUCT"
    },
)
client.delete_model_endpoint(model_endpoint="demo-endpoint-tmp")
```
