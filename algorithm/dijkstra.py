# https://atcoder.jp/contests/abc231/submissions/27832918
class Dijkstra():
    class Edge():
        def __init__(self, _to, _cost):
            self.to = _to
            self.cost = _cost

    def __init__(self, V):
        self.G = [[] for i in range(V)]
        self._E = 0
        self._V = V

    @property
    def E(self):
        return self._E

    @property
    def V(self):
        return self._V

    def add_edge(self, _from, _to, _cost):
        self.G[_from].append(self.Edge(_to, _cost))
        self._E += 1

    def shortest_path(self, s):
        import heapq
        que = []
        d = [10**15] * self.V
        d[s] = 0
        heapq.heappush(que, (0, s))

        while len(que) != 0:
            cost, v = heapq.heappop(que)
            if d[v] < cost:
                continue

            for i in range(len(self.G[v])):
                e = self.G[v][i]
                if d[e.to] > d[v] + e.cost:
                    d[e.to] = d[v] + e.cost
                    heapq.heappush(que, (d[e.to], e.to))
        return d
