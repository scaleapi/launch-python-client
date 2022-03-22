import click
from rich.console import Console
from rich.table import Table

from launch.cli.client import init_client
from launch.model_endpoint import AsyncModelEndpoint, Endpoint


@click.group("endpoints")
@click.pass_context
def endpoints(ctx):
    """Endpoints is a wrapper around model bundles in Scale Launch"""


@click.pass_context
@endpoints.command("list")
def list_endpoints(ctx):
    """List all of your Bundles"""
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
def delete_endpoint(ctx, endpoint_name):
    """Delete a model bundle"""
    client = init_client(ctx)

    console = Console()
    endpoint = Endpoint(name=endpoint_name)
    dummy_endpoint = AsyncModelEndpoint(endpoint=endpoint, client=client)
    res = client.delete_model_endpoint(dummy_endpoint)
    console.print(res)
