# WaveletMatrix
from typing import *
from functools import singledispatch, update_wrapper

T = TypeVar('T', int, float)
D = TypeVar('D', int, float)


def singledispatchmethod(func):
    dispatcher = singledispatch(func)

    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def chmax(a: T, b: T) -> Tuple[T, bool]:
    if (a < b):
        a = b  # aをbで更新
        return (a, True)
    return (a, False)


def chmin(a: T, b: T) -> Tuple[T, bool]:

    if (a > b):
        a = b  # aをbで更新
        return (a, True)
    return (a, False)
