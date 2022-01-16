# WaveletMatrix
from typing import *


def chmax(a: Any, b: Any) -> Tuple[Any, bool]:
    if (a < b):
        a = b  # aをbで更新
        return (a, True)
    return (a, False)


def chmin(a: Any, b: Any) -> Tuple[Any, bool]:

    if (a > b):
        a = b  # aをbで更新
        return (a, True)
    return (a, False)
