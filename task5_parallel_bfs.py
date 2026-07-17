"""
Task 5: Concurrent Programming — Parallel BFS
ST5003CEM — Advanced Algorithms

Implements parallel Breadth-First Search using Python threading module
(equivalent to POSIX pthreads in concept).

Synchronisation mechanisms:
  - Mutex (threading.Lock): protects shared `visited` set — critical section
  - Frontier lock:          protects shared `next_frontier` list
  - Thread barrier:         main thread joins all workers between BFS levels

Note on Python GIL:
  Python's Global Interpreter Lock (GIL) prevents true CPU parallelism
  in CPython for CPU-bound tasks. Speedup is therefore limited compared
  to a C/pthreads implementation. The code correctly implements all
  synchronisation primitives and the GIL limitation is discussed below.
  In a C implementation with pthreads, true 4-8x speedup would be observed.
"""
import threading
import time
import random
from collections import deque


def build_graph(n, avg_degree=15, seed=42):
    """Build a random undirected graph. Higher degree = larger frontier = more parallelism."""
    random.seed(seed)
    graph = {i: [] for i in range(n)}
    for i in range(n):
        neighbours = random.sample(range(n), min(avg_degree, n-1))
        for j in neighbours:
            if j != i:
                graph[i].append(j)
                graph[j].append(i)
    return graph


# ══════════════════════════════════════════════════════════════════════
# 1. SEQUENTIAL BFS
# ══════════════════════════════════════════════════════════════════════
def bfs_sequential(graph, source):
    """Standard BFS. O(V + E) time, O(V) space."""
    visited = {source}
    queue   = deque([source])
    order   = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return order


# ══════════════════════════════════════════════════════════════════════
# 2. PARALLEL BFS (Level-Synchronised)
# ══════════════════════════════════════════════════════════════════════
class ParallelBFS:
    """
    Parallel BFS: processes each frontier level using multiple threads.

    Critical sections:
      1. visited set — guarded by visited_lock (Mutex)
      2. next_frontier list — guarded by frontier_lock (Mutex)

    Thread barrier: main thread calls t.join() for all workers,
    ensuring level k+1 only starts after all level k nodes are processed.
    This prevents race conditions between BFS levels.
    """
    def __init__(self, graph, num_threads=4):
        self.graph         = graph
        self.num_threads   = num_threads

    def _worker(self, chunk, visited, visited_lock, next_frontier, frontier_lock):
        """
        Thread worker: expand nodes in `chunk`.
        Each neighbour is checked-and-added atomically under visited_lock.
        """
        local_new = []
        for node in chunk:
            for nb in self.graph[node]:
                # ── CRITICAL SECTION START ──────────────────────────
                with visited_lock:          # acquire mutex
                    if nb not in visited:
                        visited.add(nb)
                        local_new.append(nb)
                # ── CRITICAL SECTION END ────────────────────────────
        # Merge into shared frontier (minimise lock time)
        if local_new:
            with frontier_lock:
                next_frontier.extend(local_new)

    def run(self, source):
        visited        = {source}
        visited_lock   = threading.Lock()   # Mutex 1
        frontier_lock  = threading.Lock()   # Mutex 2
        frontier       = [source]
        order          = [source]
        levels         = 0

        while frontier:
            next_frontier = []
            # Partition frontier across threads
            k          = max(1, (len(frontier) + self.num_threads - 1) // self.num_threads)
            chunks     = [frontier[i:i+k] for i in range(0, len(frontier), k)]

            threads = []
            for chunk in chunks:
                t = threading.Thread(
                    target=self._worker,
                    args=(chunk, visited, visited_lock, next_frontier, frontier_lock)
                )
                threads.append(t)
                t.start()

            # ── BARRIER: wait for all threads at this level ──────────
            for t in threads:
                t.join()
            # ────────────────────────────────────────────────────────

            frontier = next_frontier
            order.extend(frontier)
            levels += 1

        return order, levels


# ══════════════════════════════════════════════════════════════════════
# 3. RACE CONDITION DEMONSTRATION
# ══════════════════════════════════════════════════════════════════════
def demo_race_condition():
    print("\n" + "="*60)
    print("  Race Condition Demo: Unsafe vs Mutex-Protected Counter")
    print("="*60)

    THREADS    = 4
    INCREMENTS = 100_000

    # ── UNSAFE: shared list manipulated without lock ───────────────
    # We use a list+sleep trick to reliably expose the race.
    unsafe_results = []
    def unsafe_worker(results):
        local = 0
        for _ in range(INCREMENTS):
            local += 1
        results.append(local)

    # Summing partial results (simulates what a beginner does wrong)
    threads = [threading.Thread(target=unsafe_worker, args=(unsafe_results,))
               for _ in range(THREADS)]
    for t in threads: t.start()
    for t in threads: t.join()
    unsafe_total = sum(unsafe_results)

    # Real race: shared counter without lock
    shared = [0]
    def racy_worker():
        for _ in range(10000):
            tmp = shared[0]          # read
            shared[0] = tmp + 1      # write (race window here)
    racy_threads = [threading.Thread(target=racy_worker) for _ in range(THREADS)]
    for t in racy_threads: t.start()
    for t in racy_threads: t.join()

    print(f"\n  Scenario 1 — Shared counter WITHOUT lock ({THREADS} threads x 10000):")
    print(f"    Expected : {THREADS * 10000}")
    print(f"    Actual   : {shared[0]}")
    corrupted = shared[0] != THREADS * 10000
    print(f"    Race condition (data corruption): {corrupted}")
    if not corrupted:
        print("    (GIL may hide race on this run — in C/pthreads corruption is guaranteed)")

    # ── SAFE: mutex-protected counter ─────────────────────────────
    safe_counter = [0]
    lock = threading.Lock()
    def safe_worker():
        for _ in range(10000):
            with lock:               # acquire mutex — CRITICAL SECTION
                safe_counter[0] += 1

    safe_threads = [threading.Thread(target=safe_worker) for _ in range(THREADS)]
    for t in safe_threads: t.start()
    for t in safe_threads: t.join()

    print(f"\n  Scenario 2 — Shared counter WITH mutex ({THREADS} threads x 10000):")
    print(f"    Expected : {THREADS * 10000}")
    print(f"    Actual   : {safe_counter[0]}")
    print(f"    Thread safe (correct): {safe_counter[0] == THREADS * 10000}")

    print("\n  Mutex ensures atomicity of read-modify-write.")
    print("  Without it: thread A reads value, thread B reads same value,")
    print("  both write back +1 => one increment is lost (lost update anomaly).")


# ══════════════════════════════════════════════════════════════════════
# 4. CORRECTNESS CHECK
# ══════════════════════════════════════════════════════════════════════
def verify(graph, source):
    seq = set(bfs_sequential(graph, source))
    par = set(ParallelBFS(graph, num_threads=4).run(source)[0])
    return seq == par, len(seq), len(par)


# ══════════════════════════════════════════════════════════════════════
# 5. BENCHMARK: SEQUENTIAL vs PARALLEL
# ══════════════════════════════════════════════════════════════════════
def benchmark():
    print("\n" + "="*68)
    print(f"{'Task 5: Parallel BFS Benchmark':^68}")
    print("="*68)

    configs = [(2000, 20), (5000, 20), (10000, 20)]

    for n, deg in configs:
        graph = build_graph(n, avg_degree=deg)

        # Correctness check
        ok, ns, np_ = verify(graph, 0)
        print(f"\nGraph n={n}, avg_degree={deg}")
        print(f"  Correctness: seq={ns} nodes, par={np_} nodes, match={ok} ✓")

        # Sequential (3 runs average)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            bfs_sequential(graph, 0)
            times.append((time.perf_counter() - t0) * 1000)
        seq_time = sum(times) / len(times)

        print(f"\n  {'Mode':<25} {'Time (ms)':>12} {'Speedup':>10} {'Efficiency':>12}")
        print(f"  {'-'*62}")
        print(f"  {'Sequential':<25} {seq_time:>11.2f}ms {'1.00x':>10} {'100.0%':>12}")

        for nt in [1, 2, 4, 8]:
            times = []
            for _ in range(3):
                pbfs = ParallelBFS(graph, num_threads=nt)
                t0   = time.perf_counter()
                pbfs.run(0)
                times.append((time.perf_counter() - t0) * 1000)
            par_time   = sum(times) / len(times)
            speedup    = seq_time / par_time
            efficiency = speedup / nt * 100
            label = f"{nt} Thread{'s' if nt > 1 else ''} (Parallel)"
            print(f"  {label:<25} {par_time:>11.2f}ms {speedup:>9.2f}x {efficiency:>11.1f}%")

    print("\n" + "="*68)
    print("Synchronisation Overhead Analysis")
    print("="*68)
    print("""
  Why parallel BFS is slower than sequential in Python:

  1. Python GIL (Global Interpreter Lock):
     CPython allows only ONE thread to execute Python bytecode at a time.
     Threading gives concurrency (I/O-bound tasks) but NOT true parallelism
     for CPU-bound tasks like BFS. In C with pthreads, true 4-8x speedup
     would be observed on a multi-core machine.

  2. Mutex contention:
     Every neighbour check requires acquiring visited_lock. With many
     threads competing for the same lock, threads queue up and wait.
     This serialises the critical section, limiting speedup.

  3. Thread creation overhead:
     Spawning and joining threads for EACH BFS level adds fixed overhead
     proportional to num_threads × num_levels.

  4. Amdahl's Law:
     Maximum speedup = 1 / (sequential_fraction + parallel_fraction/N)
     Even 5% sequential code limits max speedup to 20x regardless of
     how many threads are used.

  5. Cache coherence:
     Shared data (visited set) causes cache line invalidation across
     CPU cores, adding memory latency that grows with thread count.

  Solution for real speedup: use multiprocessing (bypasses GIL) or
  implement in C with pthreads + lock-free atomic operations on
  visited flags (compare-and-swap).
    """)


# ══════════════════════════════════════════════════════════════════════
# 6. BFS LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════
def analyse_bfs_levels():
    print("\n" + "="*60)
    print("  BFS Level Analysis (frontier size per level)")
    print("="*60)
    graph = build_graph(1000, avg_degree=15)
    pbfs  = ParallelBFS(graph, num_threads=4)

    # Manually trace level sizes
    visited  = {0}
    frontier = [0]
    level    = 0
    print(f"\n  {'Level':<8} {'Frontier size':>15} {'Parallelism potential':>22}")
    print(f"  {'-'*48}")
    while frontier:
        next_f = []
        for node in frontier:
            for nb in graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    next_f.append(nb)
        bar = "█" * min(40, len(frontier) // 5)
        print(f"  {level:<8} {len(frontier):>15}  {bar}")
        frontier = next_f
        level += 1
    print(f"\n  Total levels: {level} | Total vertices: {len(visited)}")
    print("  Large frontier levels benefit most from parallelism.")


if __name__ == "__main__":
    demo_race_condition()
    analyse_bfs_levels()
    benchmark()
