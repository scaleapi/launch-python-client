from datetime import timedelta

import click

from launch.cli.client import init_client
from launch.cli.console import pretty_print, spinner


@click.group("batch-jobs")
@click.pass_context
def batch_jobs(ctx: click.Context):
    """
    Batch Jobs is a wrapper around batch jobs in Scale Launch
    """


@batch_jobs.command("get")
@click.argument("job_id")
@click.pass_context
def get_bundle(ctx: click.Context, job_id: str):
    """Print bundle info"""
    client = init_client(ctx)

    with spinner(f"Fetching batch job '{job_id}'"):
        batch_job = client.get_batch_async_response(job_id)

    pretty_print(f"status: {batch_job['status']}")
    pretty_print(f"result: {batch_job['result']}")
    pretty_print(f"duration: {timedelta(seconds=batch_job['duration'])}")
    pretty_print(f"# tasks pending: {batch_job['num_tasks_pending']}")
    pretty_print(f"# tasks completed: {batch_job['num_tasks_completed']}")
