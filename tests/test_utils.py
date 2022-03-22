from launch.utils import trim_kwargs


def test_trim_kwargs():
    kwargs1 = {"cpus": 0.5, "gpus": None, "memory": "3Gi"}
    expected1 = {"cpus": 0.5, "memory": "3Gi"}

    kwargs2 = {"cpus": 0.5, "memory": "3Gi"}
    expected2 = {"cpus": 0.5, "memory": "3Gi"}

    kwargs3 = {}
    expected3 = {}

    kwargs4 = {1: 2, 3: "", 4: 0, 5: None}
    expected4 = {1: 2, 3: "", 4: 0}

    assert trim_kwargs(kwargs1) == expected1
    assert trim_kwargs(kwargs2) == expected2
    assert trim_kwargs(kwargs3) == expected3
    assert trim_kwargs(kwargs4) == expected4
