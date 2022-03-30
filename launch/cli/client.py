import functools

import click

import launch


@functools.lru_cache()
def init_client(ctx: click.Context):
    client = launch.LaunchClient(
        api_key=ctx.obj.api_key,
        endpoint=ctx.obj.gateway_endpoint,
        self_hosted=ctx.obj.self_hosted,
    )
    return client
