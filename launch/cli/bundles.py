import click
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Column, Table

from launch.cli.client import init_client


@click.group("bundles")
def bundles():
    """Bundles is a wrapper around model bundles in Scale Launch"""


@bundles.command("list")
@click.pass_context
def list_bundles(ctx: click.Context):
    """List all of your Bundles"""
    client = init_client(ctx)

    table = Table(
        Column("Bundle Id", overflow="fold", min_width=24),
        "Bundle name",
        "Location",
        "Packaging type",
        title="Bundles",
        title_justify="left",
    )

    for model_bundle in client.list_model_bundles():
        table.add_row(
            model_bundle.bundle_id,
            model_bundle.bundle_name,
            model_bundle.location,
            model_bundle.packaging_type,
        )
    console = Console()
    console.print(table)


@bundles.command("get")
@click.argument("bundle_name")
@click.pass_context
def get_bundle(ctx: click.Context, bundle_name: str):
    """Print bundle info"""
    client = init_client(ctx)

    model_bundle = client.get_model_bundle(bundle_name)

    console = Console()
    console.print(f"bundle_id: {model_bundle.bundle_id}")
    console.print(f"bundle_name: {model_bundle.bundle_name}")
    console.print(f"location: {model_bundle.location}")
    console.print(f"packaging_type: {model_bundle.packaging_type}")
    console.print(f"env_params: {model_bundle.env_params}")
    console.print(f"requirements: {model_bundle.requirements}")

    console.print("metadata:")
    for meta_name, meta_value in model_bundle.metadata.items():
        # TODO print non-code metadata differently
        console.print(f"{meta_name}:", style="yellow")
        syntax = Syntax(meta_value, "python")
        console.print(syntax)


@bundles.command("delete")
@click.argument("bundle_name")
@click.pass_context
def delete_bundle(ctx: click.Context, bundle_name: str):
    """Delete a model bundle"""
    client = init_client(ctx)

    console = Console()
    model_bundle = client.get_model_bundle(bundle_name)
    res = client.delete_model_bundle(model_bundle)
    console.print(res)
