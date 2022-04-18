from dataclasses import dataclass
from typing import List, Optional

from launch.pipeline import (
    Deployment,
    Runtime,
    make_sequential_pipeline,
    make_service,
)


class Joiner:
    sep: str

    def __init__(self, sep: str = "/"):
        """class based inference model"""
        self.sep = sep

    def __call__(self, tokens: List[str]) -> str:
        return self.sep.join(tokens)


def splitter(text: str, sep: str = ",") -> List[str]:
    return text.split(sep)


@dataclass
class DummyDeployment(Deployment):
    gpus: int = 1
    machine_type: str = "t4-nvidia-gpu"
    min_workers: int = 1
    max_workers: int = 10
    concurrency: int = 4
    cpu: int = 4
    memory: int = 4000


def test_pipeline():
    TEST_CASE = "hello,world"

    step_1 = make_service(
        service=splitter,
        runtime=Runtime.SYNC,
        deployment=DummyDeployment(),
    )

    step_2 = make_service(
        service=Joiner,
        runtime=Runtime.SYNC,
        deployment=DummyDeployment(),
        init_kwargs={"sep": "-"},
    )

    # test individual steps
    assert ["hello", "world"] == step_1.call(TEST_CASE)
    assert "hello-world" == step_2.call(step_1.call(TEST_CASE))

    # test sequential pipeline
    replace_pipeline = make_sequential_pipeline([step_1, step_2])
    assert "hello-world" == replace_pipeline.call("hello,world")
