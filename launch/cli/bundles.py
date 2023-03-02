import re
from typing import Optional

import click
from rich.syntax import Syntax
from rich.table import Column, Table

from launch.cli.client import init_client
from launch.cli.console import pretty_print, spinner


@click.group("bundles")
@click.pass_context
def bundles(ctx: click.Context):
    """
    Bundles is a wrapper around model bundles in Scale Launch
    """


@bundles.command("list")
@click.option("--name", "-n", help="Regex to use to filter by name", default=None)
@click.pass_context
def list_bundles(ctx: click.Context, name: Optional[str]):
    """
    List all of your Bundles
    """
    client = init_client(ctx)

    table = Table(
        Column("Bundle Id", overflow="fold", min_width=24),
        "Bundle name",
        "Location",
        "Packaging type",
        title="Bundles",
        title_justify="left",
    )
    with spinner("Fetching bundles"):
        model_bundles = client.list_model_bundles()
        for model_bundle in model_bundles:
            if name is None or re.match(name, model_bundle.name):
                table.add_row(
                    model_bundle.id,
                    model_bundle.name,
                    model_bundle.location,
                    model_bundle.packaging_type,
                )
    pretty_print(table)


@bundles.command("get")
@click.argument("bundle_name")
@click.pass_context
def get_bundle(ctx: click.Context, bundle_name: str):
    """Print bundle info"""
    client = init_client(ctx)

    with spinner(f"Fetching bundle '{bundle_name}'"):
        model_bundle = client.get_model_bundle(bundle_name)

    pretty_print(f"bundle_id: {model_bundle.id}")
    pretty_print(f"bundle_name: {model_bundle.name}")
    pretty_print(f"location: {model_bundle.location}")
    pretty_print(f"packaging_type: {model_bundle.packaging_type}")
    pretty_print(f"env_params: {model_bundle.env_params}")
    pretty_print(f"requirements: {model_bundle.requirements}")
    pretty_print(f"app_config: {model_bundle.app_config}")

    pretty_print("metadata:")
    for meta_name, meta_value in model_bundle.metadata.items():
        # TODO print non-code metadata differently
        pretty_print(f"{meta_name}:", style="yellow")
        syntax = Syntax(meta_value, "python")
        pretty_print(syntax)
