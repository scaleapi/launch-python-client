import click

from launch.cli.batch_jobs import batch_jobs
from launch.cli.bundles import bundles
from launch.cli.config import ContextObject, config, set_config
from launch.cli.endpoints import endpoints
from launch.cli.tasks import tasks


class RichGroup(click.Group):
    def format_help(self, ctx, formatter):
        formatter.write(
            """
    This is the command line interface (CLI) package for Scale Launch.

       ██╗      █████╗ ██╗   ██╗███╗   ██╗ ██████╗██╗  ██╗
       ██║     ██╔══██╗██║   ██║████╗  ██║██╔════╝██║  ██║
       ██║     ███████║██║   ██║██╔██╗ ██║██║     ███████║
       ██║     ██╔══██║██║   ██║██║╚██╗██║██║     ██╔══██║
       ███████╗██║  ██║╚██████╔╝██║ ╚████║╚██████╗██║  ██║
       ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝

"""
        )
        super().format_help(ctx, formatter)


@click.group("cli", cls=RichGroup)
@click.pass_context
def entry_point(ctx, **kwargs):
    ctx.obj = ContextObject().load()
    if ctx.obj.api_key is None:
        ctx.invoke(set_config)


entry_point.add_command(batch_jobs)  # type: ignore
entry_point.add_command(bundles)  # type: ignore
entry_point.add_command(config)  # type: ignore
entry_point.add_command(endpoints)  # type: ignore
entry_point.add_command(tasks)  # type: ignore

if __name__ == "__main__":
    entry_point()  # pylint: disable=no-value-for-parameter
