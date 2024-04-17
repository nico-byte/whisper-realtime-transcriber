import time
import functools


def async_timer(_func=None, *, print_value: bool=False, statement: str=None):
    """Return the runtime of the decorated function"""
    def decorator_async_timer(func):
        @functools.wraps(func)
        async def wrapper_async_timer(*args, **kwargs):
            start_time = time.perf_counter()
            await func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            if print_value:
                print(f"{statement} in {run_time:.4f} secs\n")
            return run_time
        return wrapper_async_timer
    if _func is None:
        return decorator_async_timer
    else:
        return decorator_async_timer(_func)