import functools

import click

import launch


# TODO: Does it make sense to instantiate the client in the context?
@functools.lru_cache()
def init_client(ctx: click.Context):
    client = launch.LaunchClient(
        api_key=ctx.obj.api_key,
        endpoint=ctx.obj.gateway_endpoint,
        self_hosted=ctx.obj.self_hosted,
    )
    return client
