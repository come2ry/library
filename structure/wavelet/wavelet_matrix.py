
#
# Copyright (c) 2012 Hiroshi Manabe (manabe.hiroshi@gmail.com)
# Based on wat-array (2010 Daisuke Okanohara)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above Copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above Copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the authors nor the names of its contributors
#    may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
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

- kth_smallest(l, r, k): 区間 [l, r) に含まれる要素のうち k 番目(0-indexed) に小さいものを返す.
計算量
構築: O( N log V )
クエリ: O( log V )
"""
from typing import *
import bisect


class BitVectorMock(object):
    def __init__(self, size):
        self._bytearray = bytearray(size + 7 / 8)
        self._bit_len = size

    def set(self, pos, bit):
        if pos >= self._bit_len:
            raise ValueError

        if bit not in (0, 1):
            raise ValueError

        self._bytearray[pos / 8] |= bit << (pos % 8)

    def peek(self, pos):
        if pos < 0 or pos >= self._bit_len:
            raise ValueError

        return 1 if (self._bytearray[(pos / 8)] & (1 << pos % 8)) else 0

    def get_length(self):
        return self._bit_len

    def rank(self, bit, pos):
        if bit not in (0, 1):
            raise ValueError

        if pos < 0:
            raise ValueError

        rank = 0

        for i in range(pos):
            if self.peek(i) == bit:
                rank += 1

        return rank

    def select(self, bit, rank):
        if bit not in (0, 1):
            raise ValueError
        if rank <= 0:
            return -1
        r = 0
        for i in range(self._bit_len):
            if self.peek(i) == bit:
                r += 1
                if rank == r:
                    return i + 1

        return -1


class WaveletMatrix(object):
    def __init__(self, bits, array, create_cache=True):
        def get_reversed_first_bits(num, max_bit, bit_num, bit_reverse_table):
            return bit_reverse_table[num & ((1 << max_bit) -
                                            (1 << (max_bit - bit_num)))]

        if bits < 0:
            raise ValueError

        self._bits = bits

        max_value = 1 << bits

        self._has_cache = create_cache
        if create_cache:
            self._bit_reverse_table = []

            for i in range(max_value):
                rev = 0
                for j in range(bits):
                    rev |= ((i & (1 << j)) >> j) << (bits - j - 1)

                self._bit_reverse_table.append(rev)

        for n in array:
            if n >= max_value:
                raise ValueError

        self._length = len(array)

        cur_array = array
        self._wavelet_matrix = []
        self._zero_counts = []

        if create_cache:
            self._node_begin_pos = []
            prev_begin_pos = [0, self._length]

            for i in range(bits):
                self._wavelet_matrix.append(BitVectorMock(self._length))
                self._node_begin_pos.append([0] * ((1 << (i + 1)) + 1))

                for n in array:
                    bit = 1 if (n & (1 << bits - i - 1)) else 0
                    subscript = get_reversed_first_bits(n, bits, i,
                                                        self._bit_reverse_table)
                    self._wavelet_matrix[i].set(prev_begin_pos[subscript], bit)
                    prev_begin_pos[subscript] += 1
                    self._node_begin_pos[i][subscript + (bit << i) + 1] += 1

                for n in reversed(range(1 << i)):
                    prev_begin_pos[n + 1] = prev_begin_pos[n]

                prev_begin_pos[0] = 0

                for j in range(1 << (i + 1)):
                    self._node_begin_pos[i][j + 1] += self._node_begin_pos[i][j]

                self._zero_counts.append(self._node_begin_pos[i][1 << i])

                prev_begin_pos = self._node_begin_pos[i]

        else:
            for i in range(bits):
                test_bit = 1 << (bits - i - 1)
                next_array = [[], []]
                self._wavelet_matrix.append(BitVectorMock(len(array)))

                for j in range(len(cur_array)):
                    n = cur_array[j]
                    bit = 1 if (n & test_bit) else 0
                    self._wavelet_matrix[i].set(j, bit)
                    next_array[bit].append(n)

                self._zero_counts.append(len(next_array[0]))

                cur_array = next_array[0] + next_array[1]

    def access(self, pos):
        if pos < 0 or pos >= self._length:
            raise ValueError

        index = pos
        num = 0

        for i in range(self._bits):
            bit = self._wavelet_matrix[i].peek(index)
            num <<= 1
            num |= bit

            index = self._wavelet_matrix[i].rank(bit, index)

            if bit:
                index += self._zero_counts[i]

        return num

    def rank(self, num, pos):
        (less, more, rank) = self.rank_all(num, 0, pos)
        return rank

    def rank_lt(self, num, pos):
        (less, more, rank) = self.rank_all(num, 0, pos)
        return less

    def rank_gt(self, num, pos):
        (less, more, rank) = self.rank_all(num, 0, pos)
        return more

    def rank_all(self, num, begin_pos, end_pos):
        if num < 0 or num >= (1 << self._bits):
            raise ValueError

        if (begin_pos < 0 or begin_pos > self._length or
                end_pos < 0 or end_pos > self._length):
            raise ValueError

        if begin_pos >= end_pos:
            return (0, 0, 0)

        more_and_less = [0, 0]
        node_num = 0

        from_zero = True if begin_pos == 0 and self._has_cache else False
        to_end = True if end_pos == self._length and self._has_cache else False

        for i in range(self._bits):
            bit = 1 if num & (1 << (self._bits - i - 1)) else 0
            range_bits = end_pos - begin_pos

            if from_zero:
                begin_zero = self._node_begin_pos[i][node_num]
            else:
                begin_zero = self._wavelet_matrix[i].rank(0, begin_pos)

            if to_end:
                end_zero = self._node_begin_pos[i][node_num + 1]
            else:
                end_zero = self._wavelet_matrix[i].rank(0, end_pos)

            if bit:
                begin_pos += self._zero_counts[i] - begin_zero
                end_pos += self._zero_counts[i] - end_zero
            else:
                begin_pos = begin_zero
                end_pos = end_zero

            non_match_bits = range_bits - (end_pos - begin_pos)
            node_num |= bit << i
            more_and_less[bit] += non_match_bits

        return (more_and_less[1], more_and_less[0],
                end_pos - begin_pos)

    def select(self, num, rank):
        return self.select_from_pos(num, 0, rank)

    def select_from_pos(self, num, pos, rank):
        if num < 0 or num >= (1 << self._bits):
            raise ValueError

        if rank <= 0:
            raise ValueError

        if pos < 0 or pos >= self._length:
            raise ValueError

        if pos == 0 and self._has_cache:
            num_rev = self._bit_reverse_table[num]
            index = self._node_begin_pos[-1][num_rev]
        else:
            index = pos
            for i in range(self._bits):
                bit = 1 if num & (1 << (self._bits - i - 1)) else 0
                index = self._wavelet_matrix[i].rank(bit, index)

                if bit:
                    index += self._zero_counts[i]

        index += rank

        for i in reversed(range(self._bits)):
            bit = 1 if num & (1 << (self._bits - i - 1)) else 0

            if bit:
                index -= self._zero_counts[i]

            index = self._wavelet_matrix[i].select(bit, index)

            if index == -1:
                return -1

        return index

    def kth_smallest(self, begin_pos, end_pos, k):
        if (begin_pos < 0 or begin_pos > self._length or end_pos < 0 or end_pos > self._length):
            raise ValueError

        if k < 0 or k >= end_pos - begin_pos:
            raise ValueError

        orig_begin_pos = begin_pos
        num = 0
        from_zero = True if begin_pos == 0 and self._has_cache else False
        to_end = True if end_pos == self._length and self._has_cache else False
        node_num = 0

        for i in range(self._bits):
            if from_zero:
                begin_zero = self._node_begin_pos[i][node_num]
            else:
                begin_zero = self._wavelet_matrix[i].rank(0, begin_pos)

            if to_end:
                end_zero = self._node_begin_pos[i][node_num + 1]
            else:
                end_zero = self._wavelet_matrix[i].rank(0, end_pos)

            zero_bits = end_zero - begin_zero

            bit = 0 if k < zero_bits else 1

            if bit:
                k -= zero_bits
                begin_pos += self._zero_counts[i] - begin_zero
                end_pos += self._zero_counts[i] - end_zero
            else:
                begin_pos = begin_zero
                end_pos = end_zero

            node_num |= bit << i
            num <<= 1
            num |= bit

        if self._has_cache:
            return (num, self.select(
                    num, begin_pos + k -
                    self._node_begin_pos[-1][
                        self._bit_reverse_table[num]] + 1) - 1)
        else:
            return (num, self.select_from_pos(num, orig_begin_pos, k + 1) - 1)


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
