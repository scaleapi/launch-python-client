import json
from typing import Optional

import click

from launch.cli.client import init_client
from launch.model_endpoint import EndpointRequest


@click.group("tasks")
@click.pass_context
def tasks(ctx: click.Context):
    """Tasks is a wrapper around sending requests to endpoints"""


@tasks.command("send")
@click.argument("endpoint_name")
@click.option("-r", "--request", help="input request as a json string")
@click.option("-f", "--json-file", help="json file containing request")
@click.pass_context
def send(
    ctx: click.Context,
    endpoint_name: str,
    request: Optional[str],
    json_file: Optional[str],
):
    """Sends request to launch endpoint"""

    # Only allowed one kind of input
    assert (request is not None) ^ (
        json_file is not None
    ), "Please supply EITHER --request OR --json-file"

    if request is not None:
        json_input = json.loads(request)
    elif json_file is not None:
        with open(json_file, "rb") as f:
            json_input = json.load(f)

    client = init_client(ctx)

    model_endpoint = client.get_model_endpoint(endpoint_name)
    print(f"Sending request to {endpoint_name} at {ctx.obj.gateway_endpoint}")
    if model_endpoint.status() is None:
        raise ValueError(f"Unable to find endpoint {endpoint_name}")

    if model_endpoint.status() != "READY":
        print(f"Warning: endpoint is not ready get: {model_endpoint.status()}")
    else:
        kwargs = {
            "request": EndpointRequest(args=json_input, return_pickled=False)
        }
        if model_endpoint.model_endpoint.endpoint_type == "async":
            future = model_endpoint.predict(**kwargs)
            response = future.get()  # blocks until completion
        else:
            response = model_endpoint.predict(**kwargs)

        print(response)
