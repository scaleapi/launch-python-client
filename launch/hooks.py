from enum import Enum


class PostInferenceHooks(str, Enum):
    INSIGHT = "insight"
    CALLBACK = "callback"
