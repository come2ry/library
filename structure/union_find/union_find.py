# union-find (rank)
class UF_tree:
    def __init__(self, n):
        self.root = [-1] * (n + 1)
        self.rank = [0] * (n + 1)

    def find(self, x):
        stack = []
        while self.root[x] >= 0:
            stack.append(x)
            x = self.root[x]
        for i in stack:
            self.root[i] = x
        return x

    def same(self, x, y):
        return self.find(x) == self.find(y)

    def unite(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x == y:
            return False
        if self.rank[x] < self.rank[y]:
            self.root[y] += self.root[x]
            self.root[x] = y
        else:
            self.root[x] += self.root[y]
            self.root[y] = x
            if self.rank[x] == self.rank[y]:
                self.rank[x] += 1
        return True

    def size(self, x):
        return -self.root[self.find(x)]
