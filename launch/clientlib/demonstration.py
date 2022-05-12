from launch_api.core import Service
from launch_api.types import I, O


def demo_service_with_triton():
    class Downloader(Service):
        def call(self, request: I) -> O:
            pass
