import functools
import os

import launch


@functools.lru_cache()
def init_client():
    api_key = os.environ.get("LAUNCH_API_KEY", None)
    if api_key:
        client = launch.DeployClient(api_key)
    else:
        raise RuntimeError("No LAUNCH_API_KEY set")
    return client
