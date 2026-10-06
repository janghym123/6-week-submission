#!/usr/bin/env python3
"""Task 3 router implementation; pair with the supplied bench.py.

Copy this file beside bench.py and name it task3_reconverge.py. The supplied
dijkstra_table() remains the authority for forwarding tables and SPF counting.
"""
import heapq

SPF_RUNS = 0
SPF_DISTANCES = {}


def dijkstra_table(graph, source):
    """Reference SPF. Returns {destination: first_hop}."""
    global SPF_RUNS, SPF_DISTANCES
    SPF_RUNS += 1
    best = {source: (0, None)}
    pq, done = [(0, source, None)], set()
    while pq:
        cost, node, first_hop = heapq.heappop(pq)
        if node in done:
            continue
        done.add(node)
        best[node] = (cost, first_hop)
        for nbr, weight in sorted(graph[node].items()):
            if nbr in done:
                continue
            hop = nbr if node == source else first_hop
            if cost + weight < best.get(nbr, (float("inf"), None))[0]:
                best[nbr] = (cost + weight, hop)
                heapq.heappush(pq, (cost + weight, nbr, hop))
    SPF_DISTANCES[source] = {d: value[0] for d, value in best.items()}
    return {d: h for d, (_, h) in best.items() if d != source and h}


class FullRecompute:
    """Reference implementation included for compatibility with bench.py."""
    def __init__(self, graph, source):
        self.graph = {n: dict(e) for n, e in graph.items()}
        self.source = source
        self.table = dijkstra_table(self.graph, source)

    def link_change(self, a, b, cost):
        if cost is None:
            self.graph[a].pop(b, None)
            self.graph[b].pop(a, None)
        else:
            self.graph[a][b] = cost
            self.graph[b][a] = cost
        self.table = dijkstra_table(self.graph, self.source)


class YourRouter:
    """Incremental invalidation: run table SPF only if the changed edge can matter."""
    def __init__(self, graph, source):
        self.graph = {n: dict(e) for n, e in graph.items()}
        self.source = source
        self.table = dijkstra_table(self.graph, source)
        # The same SPF pass that built the table also retains distances.
        self.dist = dict(SPF_DISTANCES[source])

    def link_change(self, a, b, cost):
        old = self.graph[a].get(b)
        da = self.dist.get(a, float("inf"))
        db = self.dist.get(b, float("inf"))

        # In an undirected graph, an edge can change source distances or a
        # shortest-path tie only when its weight is <= the distance difference
        # between its endpoints. Equality is included for deterministic ties.
        # Be deliberately conservative on deletions and cost increases: they
        # can invalidate an existing equal-cost route, so always run SPF.
        # For an added or cheaper edge, it can matter only if it creates a
        # shorter route or an equal-cost tie between its endpoints.
        if cost is None or (old is not None and cost >= old):
            relevant = True
        elif da == float("inf") or db == float("inf"):
            relevant = False
        else:
            relevant = cost <= abs(da - db)

        if cost is None:
            self.graph[a].pop(b, None)
            self.graph[b].pop(a, None)
        else:
            self.graph[a][b] = cost
            self.graph[b][a] = cost

        if relevant:
            self.table = dijkstra_table(self.graph, self.source)
            self.dist = dict(SPF_DISTANCES[self.source])
