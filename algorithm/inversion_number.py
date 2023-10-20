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


def inversion_number(ls):
    "転倒数"
    encode = {e: i for i, e in enumerate(sorted(set(ls)))}
    ls = [encode[e] for e in ls]
    fwt = fenwick_tree(len(encode))
    ret = 0
    for elem in reversed(ls):
        fwt.add(elem, 1)
        ret += fwt._sum(elem)
    return ret
