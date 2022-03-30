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
        "Endpoint name",
        "Metadata",
        "Endpoint type",
        title="Endpoints",
        title_justify="left",
    )

    for endpoint_sync_async in client.list_model_endpoints():
        endpoint = endpoint_sync_async.endpoint
        table.add_row(
            endpoint.name,
            endpoint.metadata,
            endpoint.endpoint_type,
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
