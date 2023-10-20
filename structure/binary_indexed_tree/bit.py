from typing import *

T = TypeVar('T')


# https://atcoder.jp/contests/abc231/submissions/28482771
# class FenwickTree:
#     def __init__(self, n: int) -> None:
#         self._n = n
#         self.data = [0] * n

#     def add(self, p: int, x: T) -> None:
#         assert 0 <= p < self._n
#         p += 1
#         while p <= self._n:
#             self.data[p - 1] += x
#             p += p & -p

#     def __sum__(self, r: int) -> T:
#         s = 0
#         while r > 0:
#             s += self.data[r - 1]
#             r -= r & -r
#         return s

#     def sum(self, l: int, r: int) -> T:
#         assert 0 <= l <= r <= self._n
#         return self.__sum__(r) - self.__sum__(l)


# https://atcoder.jp/contests/abc231/submissions/28447585
class BinaryIndexedTree:
    def __init__(self, length):
        self.n_leaves = 2**length.bit_length()
        self.tree = [0] * (self.n_leaves + 1)

    def add(self, i, x):
        i += 1
        while i <= self.n_leaves:
            self.tree[i] += x
            i += i & -i

    def initialize(self, vector):
        for i in range(len(vector)):
            self.add(i, vector[i])

    def sum(self, i):
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s


# https://atcoder.jp/contests/abc231/submissions/27890257
class BIT:
    def __init__(self, arg, is_MOD_enabled=False):
        self.is_MOD_enabled = MOD
        if isinstance(arg, int):
            self.N = arg
            self.dat = [0] * (self.N + 1)
        else:
            self.N = len(arg)
            self.dat = [0] + list(arg)
            for i in range(self.N.bit_length() - 1):
                for j in range(self.N >> (i + 1)):
                    idx = (j << 1 | 1) << i
                    if self.is_MOD_enabled:
                        self.dat[idx] %= MOD
                    nidx = idx + (idx & -idx)
                    if nidx < self.N + 1:
                        self.dat[nidx] += self.dat[idx]
                        if self.is_MOD_enabled:
                            self.dat[nidx] %= MOD

    def add(self, idx, x):
        idx += 1
        while idx < self.N + 1:
            self.dat[idx] += x
            if self.is_MOD_enabled:
                self.dat[idx] %= MOD
            idx += idx & -idx

    def sum(self, idx):
        idx += 1
        sum_ = 0
        while idx > 0:
            sum_ += self.dat[idx]
            if self.is_MOD_enabled:
                sum_ %= MOD
            idx -= idx & -idx
        return sum_

    def rangesum(self, l, r):
        return self.sum(r - 1) - self.sum(l - 1)

    def lower_bound(self, x):
        sum_ = 0
        idx = 1 << (self.N.bit_length() - 1)
        while True:
            if idx < self.N + 1 and sum_ + self.dat[idx] < x:
                sum_ += self.dat[idx]
                if idx & -idx == 1:
                    break
                idx += (idx & -idx) >> 1
            else:
                if idx & -idx == 1:
                    idx -= 1
                    break
                idx -= (idx & -idx) >> 1
        return idx, sum_


# https://atcoder.jp/contests/abc231/submissions/27875122
class BinaryIndexedTree():
    def __init__(self, n: int) -> None:
        self.n = 1 << (n.bit_length())
        self.BIT = [0] * (self.n + 1)

    def build(self, init_lis: list) -> None:
        for i, v in enumerate(init_lis):
            self.add(i, v)

    def add(self, i: int, x: int) -> None:
        i += 1
        while i <= self.n:
            self.BIT[i] += x
            i += i & -i

    def sum(self, l: int, r: int) -> int:
        return self._sum(r) - self._sum(l)

    def _sum(self, i: int) -> int:
        res = 0
        while i > 0:
            res += self.BIT[i]
            i -= i & -i
        return res

    def binary_search(self, x: int) -> int:
        i = self.n
        while True:
            if i & 1:
                if x > self.BIT[i]:
                    i += 1
                break
            if x > self.BIT[i]:
                x -= self.BIT[i]
                i += (i & -i) >> 1
            else:
                i -= (i & -i) >> 1
        return i


# https://atcoder.jp/contests/abc231/submissions/27855064
class FenwickTree:
    def __init__(self, n=0, *, array=None):
        assert (n > 0 and array is None) or (n == 0 and array)

        if array:
            n = len(array)
            self.__array = array[:]  # get用
            _container = array[:]
            for i in range(n):
                j = i | (i + 1)
                if j < n:
                    _container[j] += _container[i]
            self.__size = len(array) + 1
            self.__container = [0] + _container[:]
            self.__depth = n.bit_length()
        else:
            self.__array = [0] * n
            self.__size = n + 1
            self.__container = [0] * (n + 1)
            self.__depth = n.bit_length()

    def add(self, i, x):
        """a[i]にxを加算"""
        self.__array[i] += x
        i += 1
        while i < self.__size:
            self.__container[i] += x
            i += i & (-i)

    def sum(self, r):
        """[0, r) の総和"""
        s = 0
        while r > 0:
            s += self.__container[r]
            r -= r & (-r)
        return s

    def sum_range(self, l, r):
        """[l, r) の総和"""
        return self.sum(r) - self.sum(l)

    def upper_bound(self, s):
        """[0, r) の総和 <= s となる最大のrを求める"""
        w, r = 0, 0
        for i in reversed(range(self.__depth)):
            k = r + (1 << i)
            if k < self.__size and w + self.__container[k] <= s:
                w += self.__container[k]
                r += 1 << i
        return r

    def set(self, i, x):
        """a[i]をxに変更"""
        self.add(i, x - self.__array[i])

    def get(self, i):
        """a[i]を返す"""
        return self.__array[i]

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop = key.start, key.stop
            if start is None:
                start = 0
            if stop is None:
                stop = self.__size - 1
            if start == 0:
                return self.sum(stop)
            return self.sum_range(start, stop)
        else:
            return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)


# https://atcoder.jp/contests/abc231/submissions/27849103
import operator
from typing import TypeVar, Generic

T = TypeVar("T")


class BIT(Generic[T]):
    """v1.2 @cexen"""

    from typing import Callable

    def __init__(self, n: int, f: Callable[[T, T], T], e: T, increasing: bool = None):
        """
        increasing: Required at bisect.
        True if grasp(i) <= grasp(i + 1).
        False if grasp(i) >= grasp(i + 1).
        """
        self.size = n
        self.tree = [e] * (n + 1)
        self.f = f
        self.e = e
        self.increasing = increasing

    def __len__(self) -> int:
        return self.size

    def grasp(self, i: int = None) -> T:
        """O(log n). reduce(f, data[:i], e)."""
        if i is None:
            i = self.size
        i = min(i, self.size)
        s = self.e
        while i > 0:
            s = self.f(s, self.tree[i])
            i -= i & -i
        return s

    def operate(self, i: int, v: T) -> None:
        """O(log n). bit[i] = f(bit[i], v)."""
        i += 1  # to 1-indexed
        while i <= self.size:
            self.tree[i] = self.f(self.tree[i], v)
            i += i & -i

    def bisect_left(self, v: T) -> int:
        return self._bisect_any(v, left=True)

    def bisect_right(self, v: T) -> int:
        return self._bisect_any(v, left=False)

    def _bisect_any(self, v: T, left: bool = True) -> int:
        if self.increasing is None:
            raise RuntimeError("Specify increasing.")
        incr = self.increasing  # type: ignore
        i = 0  # 0-indexed
        u = self.e
        for s in reversed(range(self.size.bit_length())):
            k = i + (1 << s)  # 1-indexed
            if not k <= self.size:
                continue
            w = self.f(u, self.tree[k])
            if left and incr and not w < v:  # type: ignore
                continue
            if not left and incr and not w <= v:  # type: ignore
                continue
            if left and not incr and not v < w:  # type: ignore
                continue
            if not left and not incr and not v <= w:  # type: ignore
                continue
            i = k  # 0-indexed
            u = w
        return i  # 0-indexed


class BITInt(BIT[int]):
    """
    >>> b = BITInt(5, increasing=True)
    >>> b.operate(0, 10)
    >>> b.operate(1, 10)
    >>> b.operate(3, 10)
    >>> b.grasp(1)
    10
    >>> b.grasp(2)
    20
    >>> b.grasp(3)
    20
    >>> b.grasp(4)
    30
    >>> b.grasp(5)
    30
    >>> b.grasp()
    30
    >>> b.bisect_left(10)
    0
    >>> b.bisect_left(11)
    1
    >>> b.bisect_left(20)
    1
    >>> b.bisect_left(21)
    3
    >>> b.bisect_left(30)
    3
    >>> b.bisect_left(31)
    5
    >>> b.bisect_right(29)
    3
    >>> b.bisect_right(30)
    5

    >>> b = BITInt(3, f=min, e=10**9, increasing=False)
    >>> b.bisect_left(0), b.bisect_right(0)
    (3, 3)
    >>> b.operate(1, 5)
    >>> b.operate(2, 2)
    >>> b.bisect_left(6), b.bisect_right(6)
    (1, 1)
    >>> b.bisect_left(5), b.bisect_right(5)
    (1, 2)
    >>> b.bisect_left(4), b.bisect_right(4)
    (2, 2)
    """

    from typing import Callable

    def __init__(
        self,
        n: int,
        f: Callable[[int, int], int] = operator.add,
        e: int = 0,
        increasing: bool = None,
    ):
        super().__init__(n, f, e, increasing)


class BITFloat(BIT[float]):
    from typing import Callable

    def __init__(
        self,
        n: int,
        f: Callable[[float, float], float] = operator.add,
        e: float = 0.0,
        increasing: bool = None,
    ):
        super().__init__(n, f, e, increasing)


# https://atcoder.jp/contests/abc231/submissions/27897032
class Bit:
    def __init__(self, n: int):
        # sizeは元の配列の数
        self.size = n
        # treeは1-index
        # 1-indexの方が実装が簡便であり、また引数として使う値が
        # indexとしての数でない場合が多いため、1-indexのままとしている。
        self.tree = [0] * (n + 1)

    # i番目の数までの総和を求める.
    # 引数は1-index
    def sum(self, i):
        res = 0
        while i > 0:
            res += self.tree[i]
            i -= i & -i
        return res

    # i番目の数にxを加える。
    # 引数は1-index
    def add(self, i, x):
        while i <= self.size:
            self.tree[i] += x
            # i&-iは、iにおいて立っている最小のbitを表す
            i += i & -i


# https://atcoder.jp/contests/abc231/submissions/27892796
class fenwick_tree(object):
    def __init__(self, n):
        self.n = n
        self.log = n.bit_length()
        self.data = [0] * n

    def __sum(self, r):
        """rまでの値の合計"""
        s = 0
        while r > 0:
            s += self.data[r - 1]
            r -= r & -r
        return s

    def add(self, p, x):
        """ a[p] += xを行う"""
        p += 1
        while p <= self.n:
            self.data[p - 1] += x
            p += p & -p

    def sum(self, l, r):
        """a[l] + a[l+1] + .. + a[r-1]を返す"""
        return self.__sum(r) - self.__sum(l)

    def lower_bound(self, x):
        """a[0] + a[1] + .. a[i] >= x となる最小のiを返す"""
        if x <= 0:
            return -1
        i = 0
        k = 1 << self.log
        while k:
            if i + k <= self.n and self.data[i + k - 1] < x:
                x -= self.data[i + k - 1]
                i += k
            k >>= 1
        return i


# https://atcoder.jp/contests/arc136/submissions/29742197
class fenwick_tree:
    """Given an array of length n, it processes the following queries in O(log n) time.

    >   Updating an element

    >   Calculating the sum of the elements of an interval
    """

    __slots__ = ["n", "data"]

    def __init__(self, n):
        """It creates an array a[0], a[1], ... , a[n-1] of length n.
        All the elements are initialized to 0.

        Constraints
        -----------

        >   0 <= n <= 10 ** 8

        Complexity
        ----------

        >   O(n)
        """
        self.n = n
        self.data = [0] * self.n

    def add(self, p, x):
        """It processes a[p] += x.

        Constraints
        -----------

        >   0 <= p < n

        Complexity

        >   O(log n)
        """
        # assert 0 <= p < self.n
        p += 1
        while p <= self.n:
            self.data[p - 1] += x
            p += p & -p

    def sum(self, l, r):
        """It returns a[l] + a[l-1] + ... + a[r-1].

        Constraints
        -----------

        >   0 <= l <= r <= n

        Complexity
        ----------

        >   O(log n)
        """
        # assert 0 <= l <= r <= self.n
        return self._sum(r) - self._sum(l)

    def _sum(self, r):
        s = 0
        while r > 0:
            s += self.data[r - 1]
            r -= r & -r
        return s
