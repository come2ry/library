# verification-helper: PROBLEM http://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=1549
# WaveletMatrix
import sys
input = sys.stdin.buffer.readline
from typing import *
from template.template import *
from structure.wavelet.wavelet_matrix import *

def main():
    n: int = int(input())
    OFS = int(1e6)
    a: List[int] = list(map(lambda x: int(x) + OFS, input().split()))

    wm: CompressedWaveletMatrix[int] = CompressedWaveletMatrix[int](
        (int, 17), a)

    q: int = int(input())
    for _ in range(q):
        l, r, d = map(int, input().split())
        r += 1
        d += OFS
        ans: int = OFS * 2
        if(wm.rank(d, l) < wm.rank(d, r)):
            ans = 0
        else:
            succ: int = wm.next_value(l, r, d)
            if(~succ):
                ans, _ = chmin(ans, abs(succ - d))
            pred: int = wm.prev_value(l, r, d)
            if(~pred):
                ans, _ = chmin(ans, abs(pred - d))
        print(ans)


if __name__ == '__main__':
    main()
