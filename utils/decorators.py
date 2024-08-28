import time
import functools


def async_timer(_func=None, *, print_statement: str = None):
    """Return the runtime of the decorated function"""

    def decorator_async_timer(func):
        @functools.wraps(func)
        async def wrapper_async_timer(*args, **kwargs):
            start_time = time.perf_counter()
            await func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            if print_statement is not None:
                print(f"{print_statement} in {run_time:.4f} secs\n")
            return run_time

        return wrapper_async_timer

    if _func is None:
        return decorator_async_timer
    else:
        return decorator_async_timer(_func)


def sync_timer(_func=None, *, print_statement: str = None, return_some: bool = True):
    """Return the runtime of the decorated function"""

    def decorator_sync_timer(func):
        @functools.wraps(func)
        def wrapper_sync_timer(*args, **kwargs):
            start_time = time.perf_counter()
            func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            if print_statement is not None:
                print(f"{print_statement} in {run_time:.4f} secs\n")
            if return_some:
                return run_time
            return None

        return wrapper_sync_timer

    if _func is None:
        return decorator_sync_timer
    else:
        return decorator_sync_timer(_func)
