import inspect
from typing import Any, Callable, Dict, List, Optional

from launch.pipeline.deploy import Deployment
from launch.pipeline.runtime import Runtime


class ServiceDescription:
    """Base class."""


class SingleServiceDescription(ServiceDescription):
    def __init__(
        self,
        service: Callable,
        runtime: Runtime,
        deploy: Deployment,
        init_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.service = service
        self.runtime = runtime
        self.deploy = deploy
        self.init_kwargs = init_kwargs

        self._callable_service: Optional[Callable] = None

    def _init_callable_service(self):
        if self._callable_service:
            return
        if inspect.isclass(self.service):
            kwargs = self.init_kwargs or {}
            self._callable_service = self.service(**kwargs)
        else:
            assert (
                not self.init_kwargs
            ), "`init_kwargs` is given, but the service is not a class"
            self._callable_service = self.service

    def call(self, *args, **kwargs):
        self._init_callable_service()
        return self._callable_service(*args, **kwargs)


class SeqPipelineServiceDescription(ServiceDescription):
    def __init__(
        self,
        items: List[SingleServiceDescription],
    ):
        super().__init__()
        self.items = items

    def call(self, *args):
        for item in self.items:
            res = item.call(*args)
            if isinstance(res, tuple):
                args = res
            else:
                args = (res,)
        return res
