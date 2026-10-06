import argparse
import heapq
import argparse

# Undirected weighted graph: node -> {neighbour: cost}
TOPOLOGY = {
    "u": {"v": 2, "w": 5, "x": 1},
    "v": {"u": 2, "w": 3, "x": 2},
    "w": {"u": 5, "v": 3, "x": 3, "y": 1, "z": 5},
    "x": {"u": 1, "v": 2, "w": 3, "y": 1},
    "y": {"w": 1, "x": 1, "z": 2},
    "z": {"w": 5, "y": 2},
}


def dijkstra(graph, source):
    costs = {source: 0}
    queue = [(0, source)]

    while queue:
        cost, node = heapq.heappop(queue)

        # 더 나은 경로를 이미 찾았다면 무시
        if cost > costs[node]:
            continue

        for neighbour, edge_cost in graph[node].items():
            new_cost = cost + edge_cost
            if neighbour not in costs or new_cost < costs[neighbour]:
                costs[neighbour] = new_cost
                heapq.heappush(queue, (new_cost, neighbour))

    # 과제 조건상 시작점은 자기 자신까지의 비용 0으로 결과에 포함
    return costs


def forwarding_table(graph, source):
    source_costs = dijkstra(graph, source)
    table = {}

    for destination, best_cost in source_costs.items():
        if destination == source:
            continue

        for first_hop in sorted(graph[source]):
            remaining_costs = dijkstra(graph, first_hop)
            total_cost = graph[source][first_hop] + remaining_costs.get(
                destination, float("inf")
            )
            if total_cost == best_cost:
                table[destination] = first_hop
                break

    return table


def link_down(graph, a, b):
    """A copy of `graph` with the link a-b removed, in both directions."""
    g = {n: dict(e) for n, e in graph.items()}
    g[a].pop(b, None)
    g[b].pop(a, None)
    return g


# ------------------------------------------------------------------- harness
# Costs from the textbook's worked example, §5.2.1
EXPECTED_COST_U = {"v": 2, "w": 3, "x": 1, "y": 2, "z": 4}
EXPECTED_TABLE_U = {"v": "v", "w": "x", "x": "x", "y": "x", "z": "x"}


def verify():
    fails = 0
    try:
        cost = dijkstra(TOPOLOGY, "u")
    except NotImplementedError:
        print("  dijkstra is still a stub"); return 1
    ok = cost == EXPECTED_COST_U
    print(f"  {'ok  ' if ok else 'FAIL'}  costs from u: {cost}")
    if not ok:
        print(f"        expected {EXPECTED_COST_U}")
    fails += not ok

    try:
        table = forwarding_table(TOPOLOGY, "u")
    except NotImplementedError:
        print("  forwarding_table is still a stub"); return 1
    ok = table == EXPECTED_TABLE_U
    print(f"  {'ok  ' if ok else 'FAIL'}  table at u:  {table}")
    if not ok:
        print(f"        expected {EXPECTED_TABLE_U}")
    fails += not ok

    # every node should be able to reach every other
    for n in TOPOLOGY:
        t = forwarding_table(TOPOLOGY, n)
        missing = set(TOPOLOGY) - {n} - set(t)
        bad = [d for d, h in t.items() if h not in TOPOLOGY[n]]
        ok = not missing and not bad
        print(f"  {'ok  ' if ok else 'FAIL'}  table at {n} covers all, hops are neighbours"
              + (f"  missing={missing} bad={bad}" if not ok else ""))
        fails += not ok

    # cutting a link must change somebody's mind
    cut = link_down(TOPOLOGY, "u", "x")
    after = forwarding_table(cut, "u")
    ok = after != table
    print(f"  {'ok  ' if ok else 'FAIL'}  u reroutes when u-x goes down: {after}")
    fails += not ok

    print(f"\n  {'all ok' if not fails else str(fails) + ' failed'}")
    return 1 if fails else 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    raise SystemExit(verify() if a.verify else p.print_help())