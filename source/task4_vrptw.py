"""
Task 4: NP-Hard Problem — Vehicle Routing Problem with Time Windows (VRPTW)
ST5003CEM — Advanced Algorithms

NP-Hardness Proof:
  VRPTW generalises TSP. Setting 1 vehicle, unlimited capacity, and
  no time windows reduces VRPTW to TSP, which is NP-Hard (Karp 1972).
  Therefore VRPTW is NP-Hard via polynomial-time reduction from TSP.

Heuristics implemented:
  1. Greedy Nearest-Neighbour — O(n² · v) construction heuristic
  2. 2-opt Local Search        — O(n²) improvement per route
"""
import math
import random
import time
import copy

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def generate_instance(n=30, seed=42, tw_width=60):
    """
    Generate VRPTW instance.
    tw_width controls how wide time windows are — wider = more feasible routes.
    Customer: (x, y, demand, tw_open, tw_close, service_time)
    """
    random.seed(seed)
    depot = (50, 50)
    customers = []
    for _ in range(n):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        demand = random.randint(5, 15)
        tw_open = random.uniform(0, 100 - tw_width)
        tw_close = tw_open + tw_width
        service_time = random.uniform(1, 5)
        customers.append((x, y, demand, tw_open, tw_close, service_time))
    return depot, customers


def route_cost(route, customers, depot):
    if not route: return 0.0
    cost  = euclidean(depot, (customers[route[0]][0], customers[route[0]][1]))
    for i in range(len(route) - 1):
        c1, c2 = customers[route[i]], customers[route[i+1]]
        cost += euclidean((c1[0],c1[1]), (c2[0],c2[1]))
    cost += euclidean((customers[route[-1]][0], customers[route[-1]][1]), depot)
    return cost


def is_feasible(route, customers, depot, capacity):
    if not route: return True
    if sum(customers[i][2] for i in route) > capacity: return False
    time_now = 0.0
    pos = depot
    for idx in route:
        c = customers[idx]
        time_now += euclidean(pos, (c[0], c[1]))
        if time_now > c[4]: return False       # missed time window
        time_now = max(time_now, c[3]) + c[5]  # wait + service
        pos = (c[0], c[1])
    return True


def total_cost(routes, customers, depot):
    return sum(route_cost(r, customers, depot) for r in routes)


# ── Heuristic 1: Greedy Nearest-Neighbour ────────────────────────────
def greedy_vrptw(customers, depot, capacity, num_vehicles):
    """
    Greedily assign the closest feasible unvisited customer to each vehicle.
    Time:  O(n² · v)
    Quality: fast but typically 15-25% above optimal.
    """
    unvisited = list(range(len(customers)))
    routes = []

    for _ in range(num_vehicles):
        if not unvisited: break
        route = []
        load  = 0
        pos   = depot
        t_now = 0.0

        while unvisited:
            best_idx   = None
            best_dist  = float('inf')
            best_t     = 0.0
            for idx in unvisited:
                c = customers[idx]
                d = euclidean(pos, (c[0], c[1]))
                arrival = t_now + d
                if arrival <= c[4] and load + c[2] <= capacity:
                    if d < best_dist:
                        best_dist = d
                        best_idx  = idx
                        best_t    = max(arrival, c[3]) + c[5]
            if best_idx is None: break
            route.append(best_idx)
            load  += customers[best_idx][2]
            t_now  = best_t
            pos    = (customers[best_idx][0], customers[best_idx][1])
            unvisited.remove(best_idx)

        routes.append(route)

    return routes, total_cost(routes, customers, depot)


# ── Heuristic 2: 2-opt Local Search ──────────────────────────────────
def two_opt(route, customers, depot, capacity):
    """
    Reverse sub-segments of route if it reduces distance and stays feasible.
    Time: O(n²) per pass, multiple passes until no improvement.
    """
    best = route[:]
    improved = True
    iterations = 0
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best)):
                candidate = best[:i] + best[i:j+1][::-1] + best[j+1:]
                old_cost  = route_cost(best,      customers, depot)
                new_cost  = route_cost(candidate, customers, depot)
                if new_cost < old_cost - 1e-9:
                    if is_feasible(candidate, customers, depot, capacity):
                        best = candidate
                        improved = True
        iterations += 1
    return best, iterations


def vrptw_2opt(customers, depot, capacity, num_vehicles):
    """Full pipeline: greedy construction + 2-opt improvement."""
    routes, _ = greedy_vrptw(customers, depot, capacity, num_vehicles)
    improved_routes = []
    total_iters = 0
    for r in routes:
        improved_r, iters = two_opt(r, customers, depot, capacity)
        improved_routes.append(improved_r)
        total_iters += iters
    return improved_routes, total_cost(improved_routes, customers, depot), total_iters


# ── Run & Evaluate ────────────────────────────────────────────────────
def run_vrptw():
    print("\n" + "="*65)
    print("  TASK 4: VRPTW — NP-Hard Problem + Heuristics")
    print("="*65)

    print("\nWhy VRPTW is NP-Hard:")
    print("  - Generalises TSP: 1 vehicle + no capacity + no time windows = TSP")
    print("  - TSP is NP-Hard (Karp, 1972)")
    print("  - Reduction is polynomial => VRPTW is NP-Hard")
    print("  - No known polynomial-time exact algorithm (unless P = NP)")

    CAPACITY    = 50
    NUM_VEHICLES = 5

    # Use wider time windows so 2-opt can show improvement
    depot, customers = generate_instance(n=25, seed=7, tw_width=80)
    print(f"\nInstance: {len(customers)} customers | capacity={CAPACITY} | vehicles={NUM_VEHICLES}")

    # Heuristic 1
    t0 = time.perf_counter()
    routes_g, cost_g = greedy_vrptw(customers, depot, CAPACITY, NUM_VEHICLES)
    t_g = (time.perf_counter() - t0) * 1000

    # Heuristic 2
    t0 = time.perf_counter()
    routes_o, cost_o, iters = vrptw_2opt(customers, depot, CAPACITY, NUM_VEHICLES)
    t_o = (time.perf_counter() - t0) * 1000

    pct = ((cost_g - cost_o) / cost_g * 100) if cost_g > 0 else 0

    served_g = sum(len(r) for r in routes_g)
    served_o = sum(len(r) for r in routes_o)

    print(f"\n{'Heuristic':<30} {'Cost':>10} {'Time':>10} {'Served':>8}")
    print("-"*62)
    print(f"{'1. Greedy Nearest-Neighbour':<30} {cost_g:>10.2f} {t_g:>9.2f}ms {served_g:>8}")
    print(f"{'2. Greedy + 2-opt':<30} {cost_o:>10.2f} {t_o:>9.2f}ms {served_o:>8}")
    print(f"\n  2-opt improvement: {pct:.1f}%  ({iters} improvement passes)")
    print(f"  Runtime trade-off: 2-opt is {t_o/t_g:.1f}x slower than greedy")

    print("\nGreedy route details:")
    for i, route in enumerate(routes_g):
        if route:
            rc   = route_cost(route, customers, depot)
            load = sum(customers[j][2] for j in route)
            feas = is_feasible(route, customers, depot, CAPACITY)
            print(f"  Vehicle {i+1}: {len(route):2d} stops | cost={rc:6.1f} | "
                  f"load={load}/{CAPACITY} | feasible={feas}")

    print("\n2-opt route details:")
    for i, route in enumerate(routes_o):
        if route:
            rc   = route_cost(route, customers, depot)
            load = sum(customers[j][2] for j in route)
            feas = is_feasible(route, customers, depot, CAPACITY)
            print(f"  Vehicle {i+1}: {len(route):2d} stops | cost={rc:6.1f} | "
                  f"load={load}/{CAPACITY} | feasible={feas}")

    print("\n" + "="*65)
    print("Scalability: Greedy vs 2-opt")
    print("="*65)
    print(f"{'n':<6} {'Greedy cost':>13} {'2-opt cost':>12} {'Improvement':>13} {'Greedy ms':>11} {'2opt ms':>9}")
    print("-"*65)
    for n, seed in [(10,1),(20,2),(30,3),(40,4),(50,5)]:
        _, custs = generate_instance(n=n, seed=seed, tw_width=80)
        _, cg    = greedy_vrptw(custs, depot, CAPACITY, NUM_VEHICLES)
        t0 = time.perf_counter()
        _, cg = greedy_vrptw(custs, depot, CAPACITY, NUM_VEHICLES)
        tg = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        _, co, _ = vrptw_2opt(custs, depot, CAPACITY, NUM_VEHICLES)
        to_ = (time.perf_counter() - t0) * 1000
        imp = ((cg - co) / cg * 100) if cg > 0 else 0
        print(f"{n:<6} {cg:>13.2f} {co:>12.2f} {imp:>12.1f}% {tg:>10.2f}ms {to_:>8.2f}ms")

    print("\nKey insight:")
    print("  - Greedy: fast O(n²v) but suboptimal — misses global improvements")
    print("  - 2-opt:  slower O(n²) per route but finds locally optimal routes")
    print("  - For NP-Hard problems: no heuristic guarantees global optimality")
    print("  - Trade-off: better quality costs more computation time")


if __name__ == "__main__":
    run_vrptw()
