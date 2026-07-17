import heapq
import time
import random

# ── Graph Builder ─────────────────────────────────────────────────────
def build_graph(num_vertices, num_edges, allow_negative=False):
    graph = {i: [] for i in range(num_vertices)}
    edges_added = 0
    while edges_added < num_edges:
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        if u != v:
            w = random.randint(-5, 20) if allow_negative else random.randint(1, 20)
            graph[u].append((v, w))
            edges_added += 1
    return graph

# ── Dijkstra's Algorithm ──────────────────────────────────────────────
def dijkstra(graph, source):
    dist = {v: float('inf') for v in graph}
    dist[source] = 0
    prev = {v: None for v in graph}
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
    return dist, prev

def reconstruct_path(prev, target):
    path = []
    while target is not None:
        path.append(target)
        target = prev[target]
    return path[::-1]

# ── Prim's Algorithm (MST) ────────────────────────────────────────────
def prim_mst(graph, start):
    visited = {start}
    edges = [(w, start, v) for v, w in graph[start] if w > 0]
    heapq.heapify(edges)
    mst = []
    total_weight = 0

    while edges and len(visited) < len(graph):
        w, u, v = heapq.heappop(edges)
        if v in visited:
            continue
        visited.add(v)
        mst.append((u, v, w))
        total_weight += w
        for next_v, next_w in graph[v]:
            if next_v not in visited and next_w > 0:
                heapq.heappush(edges, (next_w, v, next_v))
    return mst, total_weight

# ── Bellman-Ford Algorithm ─────────────────────────────────────────────
def bellman_ford(graph, source):
    dist = {v: float('inf') for v in graph}
    dist[source] = 0
    prev = {v: None for v in graph}
    V = len(graph)

    for iteration in range(V - 1):
        updated = False
        for u in graph:
            for v, w in graph[u]:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev[v] = u
                    updated = True
        if not updated:
            break  # Early termination

    # Detect negative cycles
    for u in graph:
        for v, w in graph[u]:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                return None, None, True  # Negative cycle detected

    return dist, prev, False

# ── Demo with small named graph ───────────────────────────────────────
def demo_small_graph():
    print("\n" + "="*60)
    print("  DEMO: Small Named City Graph")
    print("="*60)

    graph = {
        'Kathmandu': [('Pokhara', 200), ('Chitwan', 150), ('Bhaktapur', 13)],
        'Pokhara':   [('Kathmandu', 200), ('Chitwan', 110), ('Lumbini', 180)],
        'Chitwan':   [('Kathmandu', 150), ('Pokhara', 110), ('Lumbini', 120)],
        'Bhaktapur': [('Kathmandu', 13), ('Banepa', 20)],
        'Banepa':    [('Bhaktapur', 20), ('Chitwan', 130)],
        'Lumbini':   [('Pokhara', 180), ('Chitwan', 120)],
    }

    print("\n[Dijkstra] Shortest paths from Kathmandu:")
    dist, prev = dijkstra(graph, 'Kathmandu')
    for city, d in sorted(dist.items()):
        path = reconstruct_path(prev, city)
        print(f"  {city:<12}: {d:>4} km  |  Path: {' -> '.join(path)}")

    print("\n[Prim's MST] Minimum Spanning Tree from Kathmandu:")
    mst, total = prim_mst(graph, 'Kathmandu')
    for u, v, w in mst:
        print(f"  {u} -- {v}  (weight: {w})")
    print(f"  Total MST weight: {total} km")

    # Graph with negative edges for Bellman-Ford
    graph_neg = {
        'A': [('B', 4), ('C', 2)],
        'B': [('D', 5), ('C', -1)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2)],
        'E': []
    }
    print("\n[Bellman-Ford] Shortest paths with negative edge (B->C: -1):")
    dist, prev, neg_cycle = bellman_ford(graph_neg, 'A')
    if neg_cycle:
        print("  Negative cycle detected!")
    else:
        for city, d in dist.items():
            path = reconstruct_path(prev, city)
            print(f"  {city}: distance={d}  path={' -> '.join(path)}")

    # Test negative cycle detection
    graph_neg_cycle = {
        'X': [('Y', 1)],
        'Y': [('Z', -3)],
        'Z': [('X', 1)],  # X->Y->Z->X = 1-3+1 = -1 (negative cycle)
    }
    print("\n[Bellman-Ford] Negative cycle detection test:")
    _, _, neg_cycle = bellman_ford(graph_neg_cycle, 'X')
    print(f"  Negative cycle found: {neg_cycle}  (expected: True)")

# ── Benchmark ─────────────────────────────────────────────────────────
def benchmark():
    print("\n" + "="*65)
    print(f"{'Task 2: Graph Algorithm Benchmark':^65}")
    print("="*65)
    configs = [(100, 300), (1000, 3000), (5000, 15000)]
    print(f"{'V':<6} {'E':<7} {'Dijkstra(ms)':>14} {'BellmanFord(ms)':>16} {'Prim(ms)':>10}")
    print("-"*60)

    for V, E in configs:
        g = build_graph(V, E, allow_negative=False)

        t = time.perf_counter()
        dijkstra(g, 0)
        t_dijk = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        bellman_ford(g, 0)
        t_bf = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        prim_mst(g, 0)
        t_prim = (time.perf_counter() - t) * 1000

        print(f"{V:<6} {E:<7} {t_dijk:>13.2f}ms {t_bf:>15.2f}ms {t_prim:>9.2f}ms")

if __name__ == "__main__":
    demo_small_graph()
    benchmark()
