from launch.logger import logger


def default_make_request(_, fn, args, kwargs):
    return fn(*args, **kwargs)


try:
    # Keep in line with Scale Launch serverside code!
    from hosted_model_inference.inference.service_requests import make_request
except ImportError:
    logger.info(
        "Unable to import serverside request maker; falling back to local request"
    )
    make_request = default_make_request


def step_decorator(fn):
    """
    Decorator to mark a function as a separate Servable.
    """

    # Identifiers will just be the name of the function. Must be unique. TODO enforce this uniqueness
    fn_name = fn.__name__

    def modified_fn(*args, **kwargs):
        return make_request(fn_name, fn, args, kwargs)

    return modified_fn


def step_decorator_class(class_obj):
    """
    Decorator to mark a class as a separate Servable. The class must implement __call__.
    """
    old_call = class_obj.__call__

    def modified_call(*args, **kwargs):
        return make_request(class_obj.__name__, old_call, args, kwargs)

    class_obj.__call__ = modified_call
    return class_obj
