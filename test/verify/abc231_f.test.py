# verification-helper: PROBLEM https://atcoder.jp/contests/abc231/tasks/abc231_F
# WaveletMatrix
import sys
input = sys.stdin.buffer.readline
from typing import *
from template.template import *
from structure.wavelet.wavelet_matrix import *


def solve(N: int, A: "List[int]", B: "List[int]"):
    """
    Aにi, Bにj(i, j \\in {1..N})をあげた時
    A[i]>A[j] and B[i]<B[j]であるバターン
    """
    A = compress(A)
    B = compress(B)
    # print(A)
    # print(B)

    AB = list(zip(A, B))
    AB = sorted(AB, key=lambda x: -100000 * x[0] + x[1])
    dup_items = defaultdict(int)
    for i in range(N):
        A[i], B[i] = AB[i]
        dup_items[(A[i], B[i])] += 1

    ans = 0
    for v in dup_items.values():
        ans += (1 + (v - 1)) * (v - 1) // 2

    # bitlen = len(bin(10**9)) - 2
    bitlen = len(bin(N)) - 2
    wm = CompressedWaveletMatrix(bitlen, B.copy())
    # wm = WaveletMatrix(31, B.copy())

    for i in range(N):
        tmp = wm.range_freq(i, N, B[i], 10**12)
        # print(f"{i:<3}: B[i]={B[i]}, +{tmp}")
        ans += tmp

    print(ans)
    return


def main():
    def iterate_tokens():
        for line in sys.stdin:
            for word in line.split():
                yield word
    tokens = iterate_tokens()
    N = int(next(tokens))  # type: int
    A = [int(next(tokens)) for _ in range(N)]  # type: "List[int]"
    B = [int(next(tokens)) for _ in range(N)]  # type: "List[int]"
    solve(N, A, B)


if __name__ == '__main__':
    main()
