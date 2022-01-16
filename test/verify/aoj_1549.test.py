# verification-helper: PROBLEM http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=1549
# WaveletMatrix
import sys
input = sys.stdin.buffer.readline
from typing import *
from template.template import *
from structure.wavelet.wavelet_matrix import *


def solve(n: int, a: List[int], q: int, l: List[int], r: List[int], d: List[int]):
    OFS = int(1e6)
    a = list(map(lambda x: x + OFS, a))
    bitlen = len(bin(10**5)) - 2

    wm = CompressedWaveletMatrix((int, bitlen), a)

    for i in range(q):
        li, ri, di = l[i], r[i], d[i]
        ri += 1
        di += OFS
        ans = OFS * 2
        if(wm.rank(di, li) < wm.rank(di, ri)):
            ans = 0
        else:
            succ = wm.next_value(li, ri, di)
            if(~succ):
                ans, _ = chmin(ans, abs(succ - di))
            pred = wm.prev_value(li, ri, di)
            if(~pred):
                ans, _ = chmin(ans, abs(pred - di))
        print(ans)


def main():
    n: int = int(input())
    a: List[int] = list(map(int, input().split()))
    q: int = int(input())
    l: List[int] = [0] * q
    r: List[int] = [0] * q
    d: List[int] = [0] * q
    for i in range(q):
        l[i], r[i], d[i] = map(int, input().split())
    solve(n, a, q, l, r, d)


if __name__ == '__main__':
    main()
