from enum import Enum


class PostInferenceHooks(str, Enum):
    """
    Post-inference hooks are functions that are called after inference is complete.

    Attributes:
        CALLBACK: The callback hook is called with the inference response and the task ID.
    """

    # INSIGHT = "insight"
    CALLBACK: str = "callback"
