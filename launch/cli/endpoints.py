import click
from rich.console import Console
from rich.table import Table

from launch.cli.client import init_client
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

    for servable_endpoint in client.list_model_endpoints():
        table.add_row(
            servable_endpoint.model_endpoint.id,
            servable_endpoint.model_endpoint.name,
            servable_endpoint.model_endpoint.bundle_name,
            servable_endpoint.model_endpoint.status,
            servable_endpoint.model_endpoint.endpoint_type,
            str(
                (servable_endpoint.model_endpoint.deployment_state or {}).get(
                    "min_workers", ""
                )
            ),
            str(
                (servable_endpoint.model_endpoint.deployment_state or {}).get(
                    "max_workers", ""
                )
            ),
            str(
                (servable_endpoint.model_endpoint.deployment_state or {}).get(
                    "available_workers", ""
                )
            ),
            str(
                (servable_endpoint.model_endpoint.deployment_state or {}).get(
                    "unavailable_workers", ""
                )
            ),
            servable_endpoint.model_endpoint.metadata or "{}",
        )
    console = Console()
    console.print(table)


@endpoints.command("delete")
@click.argument("endpoint_name")
@click.pass_context
def delete_endpoint(ctx: click.Context, endpoint_name: str):
    """Delete a model endpoint"""
    client = init_client(ctx)

    console = Console()
    endpoint = ModelEndpoint(name=endpoint_name)
    res = client.delete_model_endpoint(endpoint)
    console.print(res)


@endpoints.command("creation-logs")
@click.argument("endpoint_name")
@click.pass_context
def read_endpoint_creation_logs(ctx: click.Context, endpoint_name: str):
    """Reads the creation logs for an endpoint"""
    client = init_client(ctx)

    res = client.read_endpoint_creation_logs(endpoint_name)
    # rich fails to render the text because it's already formatted
    print(res)


@endpoints.command("get")
@click.argument("endpoint_name")
@click.pass_context
def get_endpoint(ctx: click.Context, endpoint_name: str):
    """Print bundle info"""
    client = init_client(ctx)

    model_endpoint = client.get_model_endpoint(endpoint_name).model_endpoint

    console = Console()
    console.print(f"endpoint_id: {model_endpoint.id}")
    console.print(f"endpoint_name: {model_endpoint.name}")
    console.print(f"bundle_name: {model_endpoint.bundle_name}")
    console.print(f"status: {model_endpoint.status}")
    console.print(f"resource_state: {model_endpoint.resource_state}")
    console.print(f"deployment_state: {model_endpoint.deployment_state}")
    console.print(f"metadata: {model_endpoint.metadata}")
    console.print(f"endpoint_type: {model_endpoint.endpoint_type}")
    console.print(f"configs: {model_endpoint.configs}")
    console.print(f"destination: {model_endpoint.destination}")
