from typing import List, Sequence, get_args

__all__: Sequence[str] = ("get_args_all",)


def get_args_all(t: type) -> List[type]:
    all_args: List[type] = []
    all_args.extend(get_args(t))
    try:
        orig_bases = t.__orig_bases__
    except AttributeError:
        pass
    else:
        for x in orig_bases:
            all_args.extend(get_args_all(x))
    return all_args
