"""
Step 4a: Local search (the intensification engine).

Given a starting route, repeatedly apply an improving EDIT until no edit helps.
Four move types (the four your professor used):

  remove  : drop a customer  -> frees driving time (enables future adds)
  add     : insert an unplaced customer at some position -> more deliveries
  swap    : exchange the positions of two visited customers -> fix ordering/time
  reverse : flip a segment of the route (classic 2-opt) -> untangle, cut time

IMPROVEMENT RULE:
  Score a solution by key = (objective, -time). Higher is better: deliver more
  first; on ties, take less time (freed time lets a later 'add' squeeze in more).

Two strategies:
  best_improvement  : scan the WHOLE neighbourhood, apply the single best edit.
  first_improvement : apply the FIRST improving edit found, then rescan.
                      Faster per climb -> more kick-climb cycles inside ILS.
"""

from evaluator import evaluate


def _key(res):
    return (res["objective"], -res["time"])


def local_search(inst, route, first_improvement=False):
    """Climb to a local optimum from the given route. Returns (route, obj, time)."""
    route = list(route)
    cur = evaluate(route, inst)

    while True:
        cur_key = _key(cur)
        n_customers = inst["n"]
        in_route = set(route)
        L = len(route)

        best_route = None
        best_res = None
        best_key = cur_key

        def consider(cand):
            """Return (applied_now, res) for first-improvement; track best otherwise."""
            nonlocal best_route, best_res, best_key
            r = evaluate(cand, inst)
            if not r["feasible_time"]:
                return False, None
            if _key(r) > best_key:
                best_key, best_route, best_res = _key(r), cand, r
                if first_improvement:
                    return True, r
            return False, None

        applied = False

        # ---- add (most likely to raise objective -> try first) ----
        for u in range(1, n_customers):
            if u in in_route:
                continue
            for pos in range(L + 1):
                done, _ = consider(route[:pos] + [u] + route[pos:])
                if done:
                    applied = True; break
            if applied: break

        # ---- swap ----
        if not applied:
            for i in range(L):
                for j in range(i + 1, L):
                    cand = route[:]
                    cand[i], cand[j] = cand[j], cand[i]
                    done, _ = consider(cand)
                    if done:
                        applied = True; break
                if applied: break

        # ---- reverse (2-opt) ----
        if not applied:
            for i in range(L):
                for j in range(i + 1, L):
                    done, _ = consider(route[:i] + route[i:j + 1][::-1] + route[j + 1:])
                    if done:
                        applied = True; break
                if applied: break

        # ---- remove ----
        if not applied:
            for i in range(L):
                done, _ = consider(route[:i] + route[i + 1:])
                if done:
                    applied = True; break

        if best_route is None:                 # no improving move anywhere -> local optimum
            return route, cur["objective"], cur["time"]

        route, cur = best_route, best_res


if __name__ == "__main__":
    import time as _time
    from construction import construct, read_instance, INSTANCES
    print(f"{'instance':<18}{'constr':>8}{'+LS':>6}{'UB':>5}{'gap':>9}{'secs':>7}")
    print("-" * 55)
    for name, (path, ub) in INSTANCES.items():
        inst = read_instance(path)
        route, obj0, _ = construct(inst)
        t0 = _time.time()
        route, obj1, rtime = local_search(inst, route, first_improvement=True)
        secs = _time.time() - t0
        gap = (ub - obj1) / obj1 if obj1 > 0 else float("inf")
        print(f"{name:<18}{obj0:>8}{obj1:>6}{ub:>5}{gap:>8.1%}{secs:>7.1f}")
