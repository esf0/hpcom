import functools
from datetime import datetime


def execution_time(func):
    @functools.wraps(func)
    def wrapper_execution_time(*args, **kwargs):
        start_time = datetime.now()
        value = func(*args, **kwargs)
        end_time = datetime.now()
        print("Function [" + func.__name__ + "] execution took", (end_time - start_time).total_seconds() * 1000, "ms")
        return value

    return wrapper_execution_time
