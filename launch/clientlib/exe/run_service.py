import logging

import click as click

logger = logging.getLogger("run-service")


@click.command()
@click.option(
    "--config",
    help="Path to configuration. Either YAML describing a service or Python module that constructs a service.",
    required=True,
    type=str,
)
def entrypoint(config: str) -> None:
    logger.info(f"--config: {config}")
    raise NotImplementedError


if __name__ == "__main__":
    entrypoint()
