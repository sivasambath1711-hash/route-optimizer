"""
Step 4b: Iterated Local Search (ILS) - the full algorithm.

  1. Build a start with greedy construction, then local-search it.
  2. Loop until the time limit:
       a. PERTURB the best solution (ruin-and-recreate kick of strength k).
       b. LOCAL-SEARCH the kicked solution back to a local optimum.
       c. If it beats the best, keep it and reset k to gentle (k_min).
          Otherwise count a failure; every 'patience' failures, kick harder
          (k += 1, up to k_max) to escape a stubborn region.
  3. Return the best solution ever seen, plus a history of improvements
     (for the convergence graph and the executable's console output).

Intensification = local search + kicking from the best (small k).
Diversification  = the kick itself, growing when the search stalls.
"""

import time
import random
from construction import construct, read_instance, INSTANCES
from local_search import local_search
from evaluator import route_time


def perturb(inst, route, k, rng):
    """Ruin-and-recreate: remove k random customers, insert k random ones (feasibly)."""
    route = list(route)
    for _ in range(min(k, len(route))):
        route.pop(rng.randrange(len(route)))

    placed = set(route)
    unplaced = [c for c in range(1, inst["n"]) if c not in placed]
    rng.shuffle(unplaced)

    inserted = 0
    for u in unplaced:
        if inserted >= k:
            break
        pos = rng.randrange(len(route) + 1)
        cand = route[:pos] + [u] + route[pos:]
        if route_time(cand, inst) <= inst["L"] + 1e-9:   # keep it time-feasible
            route = cand
            inserted += 1
    return route


def ils(inst, time_limit, seed=0, k_min=1, k_max=4, patience=15,
        first_improvement=True, log=False, on_improve=None):
    rng = random.Random(seed)
    start = time.time()

    route, _, _ = construct(inst)
    route, obj, rtime = local_search(inst, route, first_improvement=first_improvement)
    best = (route, obj, rtime)

    history = [(0.0, obj, list(route))]        # (elapsed_seconds, objective, sequence)
    if log:
        print(f"  t={0.0:6.1f}s  obj={obj:<3}  seq={route}")
    if on_improve:
        on_improve(0.0, obj, list(route), rtime)

    k = k_min
    fails = 0
    iters = 0

    while time.time() - start < time_limit:
        iters += 1
        cand = perturb(inst, best[0], k, rng)
        cr, co, ct = local_search(inst, cand, first_improvement=first_improvement)

        if (co, -ct) > (best[1], -best[2]):     # strictly better -> keep
            best = (cr, co, ct)
            elapsed = time.time() - start
            history.append((elapsed, co, list(cr)))
            if log:
                print(f"  t={elapsed:6.1f}s  obj={co:<3}  seq={cr}")
            if on_improve:
                on_improve(elapsed, co, list(cr), ct)
            k = k_min
            fails = 0
        else:
            fails += 1
            if fails % patience == 0:
                k = min(k + 1, k_max)            # stuck -> kick harder

    return best, history, iters


if __name__ == "__main__":
    TIME_LIMIT = 20.0        # short budget for validation; final runs use 300s
    print(f"ILS validation run ({TIME_LIMIT:.0f}s per instance; final spec = 300s)\n")
    print(f"{'instance':<18}{'obj':>5}{'UB':>5}{'gap':>9}{'time_h':>9}{'iters':>8}")
    print("-" * 54)
    for name, (path, ub) in INSTANCES.items():
        inst = read_instance(path)
        (route, obj, rtime), hist, iters = ils(inst, TIME_LIMIT, seed=0)
        gap = (ub - obj) / obj if obj > 0 else float("inf")
        print(f"{name:<18}{obj:>5}{ub:>5}{gap:>8.1%}{rtime:>9.2f}{iters:>8}")
