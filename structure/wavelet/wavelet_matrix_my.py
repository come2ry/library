# WaveletMatrix
"""
概要
2次元平面上にある点が事前に与えられているとき, オンラインでいろいろなクエリを処理するデータ構造.
基本的には事前に要素の値を要素数に圧縮する CompressedWaveletMatrix を用いると高速に動作する.

使い方
- WaveletMatrix(v): 各要素の高さ v を初期値として構築する.
- access(k): k 番目の要素を返す.
- rank(x, r): 区間 [0, r) に含まれる x の個数を返す.
- kth_smallest(l, r, k): 区間 [l, r) に含まれる要素のうち k 番目(0-indexed) に小さいものを返す.
- kth_largest(l, r, k): 区間 [l, r) に含まれる要素のうち k 番目 (0-indexed) に大きいものを返す.
- range_freq(l, r, lower, upper): 区間 [l, r) に含まれる要素のうち [lower, upper) である要素数を返す.
- prev_value(l, r, upper): 区間 [l, r) に含まれる要素のうち upper の次に小さいものを返す.
- next_value(l, r, lower): 区間 [l, r) に含まれる要素のうち lower の次に大きいものを返す.

計算量
構築: O( N log V )
クエリ: O( log V )
"""
from typing import *
import bisect


def popcount(x):
    '''xの立っているビット数をカウントする関数
    (xは64bit整数)'''

    # 2bitごとの組に分け、立っているビット数を2bitで表現する
    x = x - ((x >> 1) & 0x5555555555555555)

    # 4bit整数に 上位2bit + 下位2bit を計算した値を入れる
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)

    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0f  # 8bitごと
    x = x + (x >> 8)  # 16bitごと
    x = x + (x >> 16)  # 32bitごと
    x = x + (x >> 32)  # 64bitごと = 全部の合計
    return x & 0x0000007f


class SuccinctIndexableDictionary:
    length: int
    blocks: int
    bit: List[int]
    sum: List[int]

    def __init__(self, _length: int = None) -> None:
        if _length is None:
            self.length = 0
            self.blocks = 0
            self.bit = []
            self.sum = []
        else:
            self.length = _length
            self.blocks = (_length + 31) >> 5
            self.bit = [0] * self.blocks
            self.sum = [0] * self.blocks

    def set(self, k: int) -> None:
        self.bit[k >> 5] |= (1 << (k & 31))

    def build(self) -> None:
        self.sum[0] = 0
        for i in range(1, self.blocks):
            self.sum[i] = self.sum[i - 1] + popcount(self.bit[i - 1])
            # self.sum[i] = self.sum[i - 1] + bin(self.bit[i - 1]).count("1")

    def __getitem__(self, k: int) -> bool:
        return bool((self.bit[k >> 5] >> (k & 31)) & 1)

    def rank(self, *args: Any) -> int:
        """
        def rank(self, k: int) -> int:
        def rank(self, val: bool, k: int) -> int:
        """
        ret: Optional[int] = None
        val: bool = False
        k: int = 0
        if len(args) == 1:
            k = args[0]
            ret = self.sum[k >> 5] + \
                bin(self.bit[k >> 5] & ((1 << (k & 31)) - 1)).count("1")
            return ret

        elif len(args) == 2:
            val = args[0]
            k = args[1]
            ret = self.rank(k) if (val) else k - self.rank(k)
            return ret

        assert (
            ret is not None), f"ERROR: rankの引数の個数がおかしい\nExpected 1 or 2, Actual {len(args)}"
        return ret

    def select(self, val: bool, *args: Any) -> int:
        """
        def select(self, val: bool, k: int) -> int:
        def select(self, val: bool, i: int, l: int) -> int:
        """
        ret: Optional[int] = None
        k: int = 0
        i: int = 0
        l: int = 0
        if len(args) == 1:
            k = args[0]
            if (k < 0) or (self.rank(val, self.length) <= k):
                return -1
            low: int = 0
            high: int = self.length
            while(high - low > 1):
                mid: int = (low + high) >> 1
                if (self.rank(val, mid) >= k + 1):
                    high = mid
                else:
                    low = mid

            ret = high - 1
            return ret

        elif len(args) == 2:
            i = args[0]
            l = args[1]
            ret = self.select(val, i + self.rank(val, l))
            return ret

        assert (ret is not None), f"ERROR: selectの引数の個数がおかしい\nExpected 2 or 3, Actual {1+len(args)}"
        return ret


class WaveletMatrix:
    MAXLOG: int

    length: int
    matrix: List[SuccinctIndexableDictionary]
    mid: List[int]

    def __init__(self, _MAXLOG: int, v: List[int]) -> None:
        self.length = len(v)
        self.MAXLOG = _MAXLOG
        l: List[int] = [0] * self.length
        r: List[int] = [0] * self.length

        self.matrix = [None] * self.MAXLOG  # type: ignore
        self.mid = [0] * self.MAXLOG
        for level in range(self.MAXLOG - 1, -1, -1):
            self.matrix[level] = SuccinctIndexableDictionary(self.length + 1)
            left: int = 0
            right: int = 0
            for i in range(self.length):
                f: bool = bool((v[i] >> level) & 1)
                if (f):
                    self.matrix[level].set(i)
                    r[right] = v[i]
                    right += 1
                else:
                    l[left] = v[i]
                    left += 1
            self.mid[level] = left
            self.matrix[level].build()
            v, l = (l, v)
            for i in range(right):
                v[left + i] = r[i]

    def succ(self, f: bool, l: int, r: int, level: int) -> Tuple[int, int]:
        return (self.matrix[level].rank(f, l) + self.mid[level] * f, self.matrix[level].rank(f, r) + self.mid[level] * f)

    def access(self, k: int) -> int:
        ret: int = 0
        for level in range(self.MAXLOG - 1, -1, -1):
            f: bool = self.matrix[level][k]
            if (f):
                ret |= int(1) << level  # type: ignore
            k = self.matrix[level].rank(f, k) + self.mid[level] * f
        return ret

    def __getitem__(self, k: int) -> int:
        return self.access(k)

    def rank(self, x: int, r: int) -> int:
        l: int = 0
        for level in range(self.MAXLOG - 1, -1, -1):
            (l, r) = self.succ(bool((x >> level) & 1), l, r, level)  # type: ignore
        return r - l

    def kth_smallest(self, l: int, r: int, k: int) -> int:
        assert ((0 <= k) and (k < r - l)), "ERROR: 0 <= k && k < r - lではない"
        ret: int = 0
        for level in range(self.MAXLOG - 1, -1, -1):
            cnt: int = self.matrix[level].rank(
                False, r) - self.matrix[level].rank(False, l)
            f: bool = (cnt <= k)
            if (f):
                ret |= int(1) << level  # type: ignore
                k -= cnt
            (l, r) = self.succ(f, l, r, level)
        return ret

    def kth_largest(self, l: int, r: int, k: int) -> int:
        return self.kth_smallest(l, r, r - l - k - 1)

    def range_freq(self, l: int, r: int, *args: int) -> int:
        """
        def range_freq(self, l: int, r: int, upper: int) -> int:
        def range_freq(self, l: int, r: int, lower: int, upper: int) -> int:
        """
        ret: Optional[int] = None
        lower: int = 0
        upper: int = 0

        if len(args) == 1:
            upper = args[0]
            ret = 0
            for level in range(self.MAXLOG - 1, -1, -1):
                f: bool = bool((upper >> level) & 1)
                if (f):
                    ret += self.matrix[level].rank(False, r) - \
                        self.matrix[level].rank(False, l)
                (l, r) = self.succ(f, l, r, level)
            return ret

        elif len(args) == 2:
            lower = args[0]
            upper = args[1]
            ret = self.range_freq(l, r, upper) - self.range_freq(l, r, lower)
            return ret

        assert (
            ret is not None), f"ERROR: range_freqの引数の個数がおかしい\nExpected 3 or 4, Actual {2+len(args)}"
        return ret

    def prev_value(self, l: int, r: int, upper: int) -> int:
        cnt: int = self.range_freq(l, r, upper)
        return int(-1) if (cnt == 0) else self.kth_smallest(l, r, cnt - 1)

    def next_value(self, l: int, r: int, lower: int) -> int:
        cnt: int = self.range_freq(l, r, lower)
        return int(-1) if (cnt == (r - l)) else self.kth_smallest(l, r, cnt)


class CompressedWaveletMatrix:
    MAXLOG: int

    mat: WaveletMatrix
    ys: List[int]

    def __init__(self, _MAXLOG: int, v: List[int]) -> None:
        self.ys = v.copy()
        self.MAXLOG = _MAXLOG
        self.ys.sort()
        seen: set = set()
        seen_add = seen.add
        self.ys = [x for x in self.ys if x not in seen and not seen_add(x)]
        t: List[int] = [0] * len(v)
        for i in range(len(v)):
            t[i] = self.get(v[i])
        self.mat = WaveletMatrix(self.MAXLOG, t)

    def get(self, x: int) -> int:
        return bisect.bisect_left(self.ys, x)

    def access(self, k: int) -> int:
        return self.ys[self.mat.access(k)]

    def __getitem__(self, k: int) -> int:
        return self.access(k)

    def rank(self, x: int, r: int) -> int:
        pos: int = self.get(x)
        if (pos == len(self.ys) or self.ys[pos] != x):
            return 0
        return self.mat.rank(pos, r)

    def kth_smallest(self, l: int, r: int, k: int) -> int:
        return self.ys[self.mat.kth_smallest(l, r, k)]

    def kth_largest(self, l: int, r: int, k: int) -> int:
        return self.ys[self.mat.kth_largest(l, r, k)]

    def range_freq(self, l: int, r: int, *args: Any) -> int:
        """
        def range_freq(self, l: int, r: int, upper: int) -> int:
        def range_freq(self, l: int, r: int, lower: int, upper: int) -> int:
        """
        ret: Optional[int] = None
        lower: int = 0
        upper: int = 0

        if len(args) == 1:
            upper = args[0]
            ret = self.mat.range_freq(l, r, self.get(upper))
            return ret

        elif len(args) == 2:
            lower = args[0]
            upper = args[1]
            ret = self.mat.range_freq(l, r, self.get(lower), self.get(upper))
            return ret

        assert (
            ret is not None), f"ERROR: range_freqの引数の個数がおかしい\nExpected 3 or 4, Actual {2+len(args)}"
        return ret

    def prev_value(self, l: int, r: int, upper: int) -> int:
        ret: int = self.mat.prev_value(l, r, self.get(upper))
        return int(-1) if (ret == -1) else self.ys[ret]  # type: ignore

    def next_value(self, l: int, r: int, lower: int) -> int:
        ret: int = self.mat.next_value(l, r, self.get(lower))
        return int(-1) if (ret == -1) else self.ys[ret]  # type: ignore


class WaveletMatrixRectangleSum:
    MAXLOG: int

    length: int
    matrix: List[SuccinctIndexableDictionary]
    ds: List[List[int]]
    mid: List[int]

    def __init__(self, _MAXLOG: int, v: List[int], d: List[int]) -> None:
        assert (len(v) == len(d)), ""
        self.length = len(v)
        self.MAXLOG = _MAXLOG
        l: List[int] = [0] * self.length
        r: List[int] = [0] * self.length
        ord: List[int] = list(range(self.length))

        self.matrix = [None] * self.MAXLOG   # type: ignore
        self.mid = [0] * self.MAXLOG
        self.ds = [[int(0)] * self.length for _ in range(self.MAXLOG)]
        for level in range(self.MAXLOG - 1, -1, -1):
            self.matrix[level] = SuccinctIndexableDictionary(self.length + 1)
            left: int = 0
            right: int = 0
            for i in range(self.length):
                if (((v[ord[i]] >> level) & 1)):  # type: ignore
                    self.matrix[level].set(i)
                    r[right] = ord[i]
                    right += 1
                else:
                    l[left] = ord[i]
                    left += 1
            self.mid[level] = left
            self.matrix[level].build()
            ord, l = (l, ord)
            for i in range(right):
                ord[left + i] = r[i]

            self.ds[level] = [0] * (self.length + 1)
            self.ds[level][0] = 0
            for i in range(self.length):
                self.ds[level][i + 1] = self.ds[level][i] + d[ord[i]]

    def succ(self, f: bool, l: int, r: int, level: int) -> Tuple[int, int]:
        return (self.matrix[level].rank(f, l) + self.mid[level] * f, self.matrix[level].rank(f, r) + self.mid[level] * f)

    def rect_sum(self, l: int, r: int, *args: Any) -> int:
        """
        def rect_sum(self, l: int, r: int, upper: int) -> int:
        def rect_sum(self, l: int, r: int, lower: int, upper: int) -> int:
        """
        ret: Optional[int] = None
        lower: int = 0
        upper: int = 0

        if len(args) == 1:
            upper = args[0]
            ret = 0
            for level in range(self.MAXLOG - 1, -1, -1):
                f: bool = ((upper >> level) & 1)   # type: ignore
                if (f):
                    ret += self.ds[level][self.matrix[level].rank(
                        False, r)] - self.ds[level][self.matrix[level].rank(False, l)]
                (l, r) = self.succ(f, l, r, level)
            return ret

        elif len(args) == 2:
            lower = args[0]
            upper = args[1]
            ret = self.rect_sum(l, r, upper) - self.rect_sum(l, r, lower)
            return ret

        assert (
            ret is not None), f"ERROR: rect_sumの引数の個数がおかしい\nExpected 3 or 4, Actual {2+len(args)}"
        return ret


class CompressedWaveletMatrixRectangleSum:
    MAXLOG: int

    mat: WaveletMatrixRectangleSum
    ys: List[int]

    def __init__(self, _MAXLOG: int, v: List[int], d: List[int]) -> None:
        self.ys = v.copy()
        self.MAXLOG = _MAXLOG
        self.ys.sort()
        seen: set = set()
        seen_add = seen.add
        self.ys = [x for x in self.ys if x not in seen and not seen_add(x)]
        t: List[int] = [0] * len(v)
        for i in range(len(v)):
            t[i] = self.get(v[i])
        self.mat = WaveletMatrixRectangleSum(self.MAXLOG, t, d)

    def get(self, x: int) -> int:
        return bisect.bisect_left(self.ys, x)

    def rect_sum(self, l: int, r: int, *args: Any) -> int:
        """
        def rect_sum(self, l: int, r: int, upper: int) -> int:
        def rect_sum(self, l: int, r: int, lower: int, upper: int) -> int:
        """
        ret: Optional[int] = None
        lower: int = 0
        upper: int = 0

        if len(args) == 1:
            upper = args[0]
            ret = self.mat.rect_sum(l, r, self.get(upper))
            return ret

        elif len(args) == 2:
            lower = args[0]
            upper = args[1]
            ret = self.mat.rect_sum(l, r, self.get(lower), self.get(upper))
            return ret

        assert (
            ret is not None), f"ERROR: rect_sumの引数の個数がおかしい\nExpected 3 or 4, Actual {2+len(args)}"
        return ret


def compress(A: List[int]) -> List[int]:
    nums: List[int] = list(dict.fromkeys(A).keys())
    nums.sort()
    ret: List[int] = []
    for e in A:
        ret.append(bisect.bisect_left(nums, e))
    return ret


def main() -> None:
    N: int = int(input())
    A: List[int] = [int(input()) for _ in range(N)]
    B: List[int] = [int(input()) for _ in range(N)]

    A = compress(A)
    B = compress(B)

    p: List[Tuple[int, int]] = [(A[i], B[i]) for i in range(N)]
    p.sort()
    for i in range(N):
        A[i], B[i] = p[i]

    wm: CompressedWaveletMatrix = CompressedWaveletMatrix(20, B)
    ans: int = 0
    for i in range(N):
        pp: int = bisect.bisect_left(A, A[i])
        ans += wm.range_freq(0, pp, B[i], float('inf'))
    print(ans)


if __name__ == '__main__':
    main()
