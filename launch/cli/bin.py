from dataclasses import dataclass
from typing import Optional

import click

from launch.cli.bundles import bundles
from launch.cli.endpoints import endpoints


@dataclass
class ContextObject:
    self_hosted: bool
    gateway_endpoint: Optional[str] = None
    api_key: Optional[str] = None


@click.group("cli")
@click.option(
    "-s",
    "--self-hosted",
    is_flag=True,
    help="Use this flag if Scale Launch is self hosted",
)
@click.option(
    "-e",
    "--gateway-endpoint",
    envvar="LAUNCH_GATEWAY_ENDPOINT",
    default=None,
    type=str,
    help="Redefine Scale Launch gateway endpoint. Mandatory parameter when using self-hosted Scale Launch.",
)
@click.option(
    "-a",
    "--api-key",
    envvar="LAUNCH_API_KEY",
    required=True,
    type=str,
    help="Scale Lunch API key",
)
@click.pass_context
def entry_point(ctx, **kwargs):
    """Launch CLI

        \b
    ██╗      █████╗ ██╗   ██╗███╗   ██╗ ██████╗██╗  ██╗
    ██║     ██╔══██╗██║   ██║████╗  ██║██╔════╝██║  ██║
    ██║     ███████║██║   ██║██╔██╗ ██║██║     ███████║
    ██║     ██╔══██║██║   ██║██║╚██╗██║██║     ██╔══██║
    ███████╗██║  ██║╚██████╔╝██║ ╚████║╚██████╗██║  ██║
    ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝

        `scale-launch` is a command line interface to interact with Scale Launch
    """
    ctx.obj = ContextObject(**kwargs)


entry_point.add_command(bundles)  # type: ignore
entry_point.add_command(endpoints)  # type: ignore

if __name__ == "__main__":
    entry_point()  # pylint: disable=no-value-for-parameter
