import click
from rich.table import Table

from launch.cli.client import init_client
from launch.cli.console import pretty_print, spinner
from launch.model_endpoint import ModelEndpoint


@click.group("endpoints")
@click.pass_context
def endpoints(ctx: click.Context):
    """Endpoints is a wrapper around model endpoints in Scale Launch"""


@click.pass_context
@endpoints.command("list")
@click.pass_context
def list_endpoints(ctx: click.Context):
    """List all of your Endpoints"""
    client = init_client(ctx)

    table = Table(
        "Endpoint ID",
        "Endpoint name",
        "Bundle name",
        "Status",
        "Endpoint type",
        "Min Workers",
        "Max Workers",
        "Available Workers",
        "Unavailable Workers",
        "Metadata",
        title="Endpoints",
        title_justify="left",
    )

    with spinner("Fetching model endpoints"):
        model_endpoints = client.list_model_endpoints()
        for servable_endpoint in model_endpoints:
            table.add_row(
                servable_endpoint.model_endpoint.id,
                servable_endpoint.model_endpoint.name,
                servable_endpoint.model_endpoint.bundle_name,
                servable_endpoint.model_endpoint.status,
                servable_endpoint.model_endpoint.endpoint_type,
                str(
                    (
                        servable_endpoint.model_endpoint.deployment_state or {}
                    ).get("min_workers", "")
                ),
                str(
                    (
                        servable_endpoint.model_endpoint.deployment_state or {}
                    ).get("max_workers", "")
                ),
                str(
                    (
                        servable_endpoint.model_endpoint.deployment_state or {}
                    ).get("available_workers", "")
                ),
                str(
                    (
                        servable_endpoint.model_endpoint.deployment_state or {}
                    ).get("unavailable_workers", "")
                ),
                servable_endpoint.model_endpoint.metadata or "{}",
            )

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
        model_endpoint = client.get_model_endpoint(
            endpoint_name
        ).model_endpoint

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
