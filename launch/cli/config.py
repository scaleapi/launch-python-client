import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import click
import questionary as q
from rich.console import Console
from rich.table import Table


@dataclass
class ContextObject:
    self_hosted: Optional[bool] = False
    gateway_endpoint: Optional[str] = None
    api_key: Optional[str] = None

    @staticmethod
    def config_path():
        config_dir = click.get_app_dir("launch")
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        return os.path.join(config_dir, "config.json")

    def load(self):
        try:
            with open(self.config_path(), "r", encoding="utf-8") as f:
                new_items = json.load(f)
                for key, value in new_items.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
        except FileNotFoundError:
            pass

        return self

    def save(self):
        with open(self.config_path(), "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)


@click.group("config")
@click.pass_context
def config(ctx: click.Context):
    """
    Config is a wrapper around getting and setting your Scale Launch configuration
    """


@config.command("get")
@click.pass_context
def get_config(ctx: click.Context):
    table = Table(
        "Self-Hosted",
        "API Key",
        "Gateway Endpoint",
    )

    table.add_row(
        str(ctx.obj.self_hosted), ctx.obj.api_key, ctx.obj.gateway_endpoint
    )
    console = Console()
    console.print(table)


@config.command("set")
@click.pass_context
def set_config(ctx: click.Context):
    ctx.obj.api_key = q.text(
        message="Your Scale API Key?",
        default=ctx.obj.api_key or "",
        validate=lambda x: isinstance(x, str)
        and len(x) > 16,  # Arbitrary length right now
    ).ask()
    ctx.obj.self_hosted = q.confirm(
        message="Is your installation of Launch self-hosted?",
        default=ctx.obj.self_hosted,
    ).ask()
    if ctx.obj.self_hosted:
        ctx.obj.gateway_endpoint = q.text(
            message="Your Gateway Endpoint?",
            default=ctx.obj.gateway_endpoint or "",
        ).ask()

    ctx.obj.save()
