"""
ST5003CEM — Advanced Algorithms Coursework
Run all 5 tasks in sequence.

Usage:
    python run_all_tasks.py          # run everything
    python run_all_tasks.py --task 1 # run specific task
"""
import sys
import time

def separator(title):
    print("\n")
    print("█"*65)
    print(f"  {title}")
    print("█"*65)

def main():
    task_filter = None
    if "--task" in sys.argv:
        idx = sys.argv.index("--task")
        task_filter = int(sys.argv[idx + 1])

    start_total = time.perf_counter()

    if task_filter in (None, 1):
        separator("TASK 1 — Advanced Data Structures (25 marks)")
        from task1_data_structures import benchmark
        benchmark()

    if task_filter in (None, 2):
        separator("TASK 2 — Graph Algorithms & Pathfinding (30 marks)")
        from task2_graph_algorithms import demo_small_graph, benchmark
        demo_small_graph()
        benchmark()

    if task_filter in (None, 3):
        separator("TASK 3 — Algorithmic Strategies (25 marks)")
        from task3_dp_greedy_backtracking import run_dp, run_greedy, run_backtracking
        run_dp()
        run_greedy()
        run_backtracking()

    if task_filter in (None, 4):
        separator("TASK 4 — NP-Hard: VRPTW Heuristics (10 marks)")
        from task4_vrptw import run_vrptw
        run_vrptw()

    if task_filter in (None, 5):
        separator("TASK 5 — Concurrent Programming (10 marks)")
        from task5_parallel_bfs import demo_race_condition, analyse_bfs_levels, benchmark
        demo_race_condition()
        analyse_bfs_levels()
        benchmark()

    total = time.perf_counter() - start_total
    print(f"\n{'='*65}")
    print(f"  All tasks completed in {total:.2f}s")
    print(f"{'='*65}\n")

if __name__ == "__main__":
    main()
