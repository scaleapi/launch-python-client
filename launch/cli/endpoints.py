from pprint import pformat
from typing import NamedTuple

import click
import questionary as q
from rich.table import Table
from typing_extensions import Literal

from launch.cli.client import init_client
from launch.cli.console import pretty_print, spinner
from launch.hooks import PostInferenceHooks
from launch.model_endpoint import ModelEndpoint


@click.group("endpoints")
@click.pass_context
def endpoints(ctx: click.Context):
    """Endpoints is a wrapper around model endpoints in Scale Launch"""


class EndpointRow(NamedTuple):
    id: str
    endpoint_name: str
    bundle_name: str
    status: Literal["READY", "UPDATE_PENDING", "UPDATE_IN_PROGRESS", "UPDATE_FAILED", "DELETE_IN_PROGRESS"]
    endpoint_type: Literal["async", "sync"]
    min_workers: str  # rich.table requires all strings
    max_workers: str
    available_workers: str
    unavailable_workers: str
    num_gpus: str
    metadata: str


@click.pass_context
@endpoints.command("list")
@click.option("-o", "--orderby", required=False, type=click.Choice(EndpointRow._fields), help="How to order the table")
@click.option(
    "-d",
    "--descending",
    required=False,
    is_flag=True,
    type=bool,
    default=False,
    help="Whether to sort in descending order",
)
@click.pass_context
def list_endpoints(ctx: click.Context, orderby, descending):
    """List all of your Endpoints"""
    client = init_client(ctx)

    table = Table(
        "Endpoint ID",
        "Endpoint name",
        "Bundle name",
        "Status",
        "Endpoint\ntype",
        "Min\nWorkers",
        "Max\nWorkers",
        "Available\nWorkers",
        "Unavailable\nWorkers",
        "Num\nGPUs",
        "Metadata",
        title="Endpoints",
        title_justify="left",
    )

    with spinner("Fetching model endpoints"):
        model_endpoints = client.list_model_endpoints()
        endpoint_rows = []
        for servable_endpoint in model_endpoints:
            row = EndpointRow(
                servable_endpoint.model_endpoint.id,
                servable_endpoint.model_endpoint.name,
                servable_endpoint.model_endpoint.bundle_name,
                servable_endpoint.model_endpoint.status,
                servable_endpoint.model_endpoint.endpoint_type,
                str((servable_endpoint.model_endpoint.deployment_state or {}).get("min_workers", "")),
                str((servable_endpoint.model_endpoint.deployment_state or {}).get("max_workers", "")),
                str((servable_endpoint.model_endpoint.deployment_state or {}).get("available_workers", "")),
                str((servable_endpoint.model_endpoint.deployment_state or {}).get("unavailable_workers", "")),
                str((servable_endpoint.model_endpoint.resource_state or {}).get("gpus", "0")),
                servable_endpoint.model_endpoint.metadata or "{}",
            )
            endpoint_rows.append(row)

        if orderby is not None:
            endpoint_rows = sorted(endpoint_rows, key=lambda x: getattr(x, orderby), reverse=descending)

        for row in endpoint_rows:
            table.add_row(*row)

        pretty_print(table)

    pretty_print(table)


@endpoints.command("delete")
@click.argument("endpoint_name")
@click.pass_context
def delete_endpoint(ctx: click.Context, endpoint_name: str):
    """Delete a model endpoint"""
    client = init_client(ctx)

    with spinner(f"Deleting model endpoint '{endpoint_name}'"):
        endpoint = ModelEndpoint(name=endpoint_name)
        res = client.delete_model_endpoint(endpoint)

    pretty_print(res)


@endpoints.command("creation-logs")
@click.argument("endpoint_name")
@click.pass_context
def read_endpoint_creation_logs(ctx: click.Context, endpoint_name: str):
    """Reads the creation logs for an endpoint"""
    client = init_client(ctx)

    with spinner(f"Fetching creation logs for endpoint '{endpoint_name}'"):
        res = client.read_endpoint_creation_logs(endpoint_name)

    # rich fails to render the text because it's already formatted
    print(res)


@endpoints.command("get")
@click.argument("endpoint_name")
@click.pass_context
def get_endpoint(ctx: click.Context, endpoint_name: str):
    """Print bundle info"""
    client = init_client(ctx)

    with spinner(f"Fetching endpoint '{endpoint_name}'"):
        model_endpoint = client.get_model_endpoint(endpoint_name).model_endpoint

    pretty_print(f"endpoint_id: {model_endpoint.id}")
    pretty_print(f"endpoint_name: {model_endpoint.name}")
    pretty_print(f"bundle_name: {model_endpoint.bundle_name}")
    pretty_print(f"status: {model_endpoint.status}")
    pretty_print(f"resource_state: {model_endpoint.resource_state}")
    pretty_print(f"deployment_state: {model_endpoint.deployment_state}")
    pretty_print(f"metadata: {model_endpoint.metadata}")
    pretty_print(f"endpoint_type: {model_endpoint.endpoint_type}")
    pretty_print(f"configs: {model_endpoint.configs}")
    pretty_print(f"destination: {model_endpoint.destination}")
    pretty_print(f"post-inference hooks: {model_endpoint.post_inference_hooks}")
    pretty_print(f"default callback url: {model_endpoint.default_callback_url}")


def _validate_int(val: str) -> int:
    try:
        int(val)
        return True
    except ValueError:
        pass
    return False


def _dict_not_none_or_empty(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None and v != "" and v != []}


@endpoints.command("edit")
@click.argument("endpoint_name")
@click.pass_context
def edit_endpoint(ctx: click.Context, endpoint_name: str):
    """Edit an endpoint"""
    client = init_client(ctx)

    with spinner(f"Fetching endpoint '{endpoint_name}'"):
        model_endpoint = client.get_model_endpoint(endpoint_name).model_endpoint

        model_bundles = client.list_model_bundles()
        model_bundle_choices = [
            q.Choice(
                f"Current bundle ({model_endpoint.bundle_name})",
                value="",
                checked=True,
            )
        ]
        for bundle in model_bundles:
            model_bundle_choices.append(q.Choice(title=pformat(bundle), value=bundle))

        post_inference_hooks_choices = []
        post_inference_hooks = model_endpoint.post_inference_hooks or []
        for hook in PostInferenceHooks:
            value = hook.value  # type: ignore
            post_inference_hooks_choices.append(q.Choice(title=value, checked=(value in post_inference_hooks)))

    if model_endpoint.status != "READY":
        pretty_print(f"Endpoint '{endpoint_name}' is not ready. Please wait for it to be ready " "before editing.")
        return

    model_bundle = q.select("Model bundle: ", choices=model_bundle_choices).ask()
    resource_state = _dict_not_none_or_empty(**(model_endpoint.resource_state or {}))
    deployment_state = _dict_not_none_or_empty(**(model_endpoint.deployment_state or {}))
    cpus = q.text("Cpus: ", default=resource_state.get("cpus", "")).ask()
    gpu_raw = q.text(
        "Gpus: ",
        default=str(resource_state.get("gpus", "")),
        validate=_validate_int,
    ).ask()
    gpus = int(gpu_raw)
    memory = q.text("Memory: ", default=resource_state.get("memory", "")).ask()
    storage = q.text("Storage (optional): ", default=resource_state.get("storage", "")).ask()
    gpu_type_prompt = "Gpu type (optional): " if gpus == 0 else "Gpu type: "
    gpu_type = q.select(
        gpu_type_prompt,
        choices=["", "nvidia-tesla-t4", "nvidia-ampere-a10", "nvidia-a100"],
    ).ask()
    min_workers = q.text(
        "Min workers: ",
        default=str(deployment_state.get("min_workers", "")),
        validate=_validate_int,
    ).ask()
    min_workers = int(min_workers)
    max_workers = q.text(
        "Max workers: ",
        default=str(deployment_state.get("max_workers", "")),
        validate=_validate_int,
    ).ask()
    max_workers = int(max_workers)
    per_worker = q.text(
        "Per worker: ",
        default=str(deployment_state.get("per_worker", "")),
        validate=_validate_int,
    ).ask()
    per_worker = int(per_worker)
    post_inference_hooks = q.checkbox("Post-inference hooks: ", choices=post_inference_hooks_choices).ask()
    default_callback_url = q.text(
        "Default callback url (optional): ",
        default=model_endpoint.default_callback_url or "",
    ).ask()

    kwargs = _dict_not_none_or_empty(
        model_bundle=model_bundle,
        cpus=cpus,
        memory=memory,
        storage=storage,
        gpus=gpus,
        min_workers=min_workers,
        max_workers=max_workers,
        per_worker=per_worker,
        gpu_type=gpu_type,
        post_inference_hooks=post_inference_hooks,
        default_callback_url=default_callback_url,
    )

    with spinner(f"Editing endpoint '{endpoint_name}'"):
        # TODO: Print out a nice error message if the user passes in arguments
        # that fail server-side validation.
        client.edit_model_endpoint(model_endpoint=model_endpoint, **kwargs)
