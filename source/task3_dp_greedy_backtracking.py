import time
from bisect import bisect_right
import random

# ══════════════════════════════════════════════════════════════════════
# TASK 3A — Dynamic Programming: Weighted Job Scheduling
# ══════════════════════════════════════════════════════════════════════

def weighted_job_scheduling(jobs):
    """
    jobs: list of (start, finish, profit)
    Returns: (max_profit, selected_jobs)
    O(n log n) time, O(n) space
    """
    jobs = sorted(jobs, key=lambda x: x[1])  # sort by finish time
    n = len(jobs)
    finishes = [j[1] for j in jobs]
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        s, f, p = jobs[i - 1]
        # Latest job that finishes <= start of job i
        idx = bisect_right(finishes, s, 0, i - 1)
        include = p + dp[idx]
        exclude = dp[i - 1]
        dp[i] = max(include, exclude)

    # Backtrack to find selected jobs
    selected = []
    i = n
    while i > 0:
        s, f, p = jobs[i - 1]
        idx = bisect_right(finishes, s, 0, i - 1)
        if p + dp[idx] > dp[i - 1]:
            selected.append(jobs[i - 1])
            i = idx
        else:
            i -= 1

    return dp[n], selected[::-1]


def run_dp():
    print("\n" + "="*60)
    print("  TASK 3A: Weighted Job Scheduling (Dynamic Programming)")
    print("="*60)

    # Example from coursework
    jobs = [
        (1, 3, 50),   # start=1, finish=3, profit=50
        (2, 5, 10),
        (4, 6, 70),
        (6, 7, 60),
        (5, 8, 20),
        (1, 4, 30),
    ]
    profit, selected = weighted_job_scheduling(jobs)
    print(f"\nExample jobs: {jobs}")
    print(f"Max profit  : {profit}")
    print(f"Selected    : {selected}")

    # Benchmark
    print(f"\n{'n':<8} {'Time (ms)':>12} {'Max Profit':>12}")
    print("-"*35)
    for n in [100, 1000, 10000]:
        random_jobs = [
            (random.randint(0, 100),
             random.randint(50, 200),
             random.randint(1, 100))
            for _ in range(n)
        ]
        # Ensure finish > start
        random_jobs = [(s, s + random.randint(1, 50), p) for s, _, p in random_jobs]
        t = time.perf_counter()
        profit, _ = weighted_job_scheduling(random_jobs)
        elapsed = (time.perf_counter() - t) * 1000
        print(f"{n:<8} {elapsed:>11.3f}ms {profit:>12}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3B — Greedy: Minimum Number of Platforms
# ══════════════════════════════════════════════════════════════════════

def min_platforms(arrivals, departures):
    """
    O(n log n) — sort then two-pointer sweep.
    Provably optimal: computes maximum simultaneous overlap.
    """
    arrivals = sorted(arrivals)
    departures = sorted(departures)
    n = len(arrivals)
    platforms = 0
    max_platforms = 0
    i = j = 0

    while i < n and j < n:
        if arrivals[i] <= departures[j]:
            platforms += 1
            max_platforms = max(max_platforms, platforms)
            i += 1
        else:
            platforms -= 1
            j += 1
    return max_platforms


def run_greedy():
    print("\n" + "="*60)
    print("  TASK 3B: Minimum Number of Platforms (Greedy)")
    print("="*60)

    # Example
    arr = [900, 940, 950, 1100, 1500, 1800]
    dep = [910, 1200, 1120, 1130, 1900, 2000]
    result = min_platforms(arr, dep)
    print(f"\nArrivals  : {arr}")
    print(f"Departures: {dep}")
    print(f"Min platforms needed: {result}  (expected: 3)")

    # Edge cases
    print("\nEdge cases:")
    print(f"  No overlap    : {min_platforms([100, 300, 500], [200, 400, 600])} (expected: 1)")
    print(f"  All overlap   : {min_platforms([100, 100, 100], [500, 500, 500])} (expected: 3)")
    print(f"  Single train  : {min_platforms([800], [900])} (expected: 1)")

    # Comparison: greedy vs brute-force (small input)
    print("\nGreedy vs Brute-Force verification (n=6):")
    print(f"  Greedy result : {result}")
    # Brute force counts max overlap at each time point
    times = sorted([(t, 'A') for t in arr] + [(t, 'D') for t in dep])
    curr = peak = 0
    for _, kind in times:
        if kind == 'A':
            curr += 1
            peak = max(peak, curr)
        else:
            curr -= 1
    print(f"  Brute force   : {peak}")
    print(f"  Match: {result == peak}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3C — Backtracking: Knight's Tour (Warnsdorff's Heuristic)
# ══════════════════════════════════════════════════════════════════════

MOVES = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)]

def knight_tour(n, start_x=0, start_y=0):
    """
    Knight's Tour using Warnsdorff's heuristic as pruning.
    Returns board with move numbers, or None if no solution found.
    O(n^2) in practice with heuristic (O((n^2)!) worst case without).
    """
    board = [[-1] * n for _ in range(n)]

    def valid(x, y):
        return 0 <= x < n and 0 <= y < n and board[x][y] == -1

    def degree(x, y):
        return sum(1 for dx, dy in MOVES if valid(x+dx, y+dy))

    def solve(x, y, move_num):
        board[x][y] = move_num
        if move_num == n * n - 1:
            return True
        # Warnsdorff: try moves with fewest onward options first
        candidates = []
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if valid(nx, ny):
                candidates.append((degree(nx, ny), nx, ny))
        candidates.sort()  # ascending by degree (Warnsdorff's rule)
        for _, nx, ny in candidates:
            if solve(nx, ny, move_num + 1):
                return True
        board[x][y] = -1  # backtrack
        return False

    if solve(start_x, start_y, 0):
        return board
    return None


def print_board(board):
    n = len(board)
    width = len(str(n * n))
    for row in board:
        print("  " + " ".join(f"{cell:{width}}" for cell in row))


def run_backtracking():
    print("\n" + "="*60)
    print("  TASK 3C: Knight's Tour (Backtracking + Warnsdorff)")
    print("="*60)

    for size in [5, 6, 8]:
        print(f"\n{size}x{size} board (starting at 0,0):")
        t = time.perf_counter()
        board = knight_tour(size)
        elapsed = (time.perf_counter() - t) * 1000
        if board:
            print(f"  Solution found in {elapsed:.2f}ms")
            print_board(board)
            # Verify: check all squares visited exactly once
            flat = [board[r][c] for r in range(size) for c in range(size)]
            assert sorted(flat) == list(range(size*size)), "Invalid tour!"
            print(f"  Verified: all {size*size} squares visited exactly once ✓")
        else:
            print(f"  No solution found ({elapsed:.2f}ms)")

    # Performance comparison across board sizes
    print(f"\n{'Size':<6} {'Time (ms)':>12} {'Squares':>10}")
    print("-"*30)
    for n in [5, 6, 7, 8, 10, 12]:
        t = time.perf_counter()
        board = knight_tour(n)
        elapsed = (time.perf_counter() - t) * 1000
        status = f"{n*n}" if board else "No solution"
        print(f"{n}x{n:<3} {elapsed:>11.2f}ms {status:>10}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_dp()
    run_greedy()
    run_backtracking()
