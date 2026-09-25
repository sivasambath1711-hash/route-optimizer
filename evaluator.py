"""
Step 2: The route evaluator.

Given a fixed sequence of customers, answer two questions:
  1. Is the route feasible on TIME?  (total driving <= L = 8 hours)
  2. What is the MAXIMUM number of units it can deliver, respecting capacity Q?

Question 2 is itself a small optimization problem (see the "conveyor belt"
explanation). We model it as a linear program:

  For each visited customer 'p' and each commodity 'c' we have a variable y[p,c]
  = the change in on-board load of commodity c at that stop.
     - supplier of c (q>0):  y in [0, q]      (pick up 0..q, load goes UP)
     - demander of c (q<0):  y in [-|q|, 0]   (drop off 0..|q|, load goes DOWN)
     - neither            :  no variable (contributes 0)

  Load of commodity c right after stop p  =  L[p,c] = sum of y[p',c] for p' <= p.

  Constraints:
     (a) L[p,c] >= 0              -> can't drop what you never picked up / no negatives
     (b) sum_c L[p,c] <= Q        -> the conveyor belt is only Q wide at every point
     (c) L[last,c] == 0           -> van returns to depot EMPTY (all commodities)

  Objective: maximize total units delivered = sum of drop-offs = -sum(y at demanders).

Because supplies/demands are integers and this LP has network structure, the
optimum comes out integer in practice (we assert this to be safe).
"""

import numpy as np
from scipy.optimize import linprog
from parser import read_instance


def route_time(seq, inst):
    """Total driving time for depot -> seq[0] -> ... -> seq[-1] -> depot."""
    t = inst["t"]
    if not seq:
        return 0.0
    total = t[0][seq[0]]                       # depot to first customer
    for a, b in zip(seq, seq[1:]):
        total += t[a][b]                        # customer to customer
    total += t[seq[-1]][0]                      # last customer back to depot
    return total


def evaluate(seq, inst):
    """
    Return a dict:
      feasible_time : bool  (route fits within L hours)
      time          : float (hours)
      objective     : int   (max units deliverable; 0 if time-infeasible)
      plan          : per-stop (pickup/dropoff) amounts, for the console output later
    """
    time = route_time(seq, inst)
    if time > inst["L"] + 1e-9:
        return {"feasible_time": False, "time": time, "objective": 0, "plan": None}

    demand = inst["demand"]
    Q = inst["capacity"]
    C = inst["num_commodities"]
    k = len(seq)
    if k == 0:
        return {"feasible_time": True, "time": time, "objective": 0, "plan": []}

    # --- Build the list of decision variables y[p,c] that actually exist ---
    var_index = {}          # (p, c) -> column number in the LP
    lb, ub = [], []         # bounds for each variable
    is_dropoff = []         # True if this variable is a drop-off (used for objective)
    for p in range(k):
        node = seq[p]
        for c in range(C):
            q = demand[node][c]
            if q > 0:                          # supplier: pickup in [0, q]
                var_index[(p, c)] = len(lb)
                lb.append(0.0); ub.append(float(q)); is_dropoff.append(False)
            elif q < 0:                        # demander: dropoff in [-|q|, 0]
                var_index[(p, c)] = len(lb)
                lb.append(float(q)); ub.append(0.0); is_dropoff.append(True)
            # q == 0 -> no variable

    nvars = len(lb)
    if nvars == 0:
        return {"feasible_time": True, "time": time, "objective": 0, "plan": []}

    # --- Objective: maximize units delivered = -sum(y at demanders) ---
    # linprog MINIMIZES, so minimize sum(y at demanders) (they are <= 0),
    # which maximizes total drop-off magnitude.
    cost = np.zeros(nvars)
    for j in range(nvars):
        if is_dropoff[j]:
            cost[j] = 1.0        # minimizing sum of (negative) dropoff vars => maximize delivered

    A_ub, b_ub = [], []          # inequality rows:  A_ub x <= b_ub
    A_eq, b_eq = [], []          # equality rows:    A_eq x  = b_eq

    # Helper: coefficient row for L[p,c] = sum_{p'<=p} y[p',c]
    def load_row(p, c):
        row = np.zeros(nvars)
        for pp in range(p + 1):
            if (pp, c) in var_index:
                row[var_index[(pp, c)]] = 1.0
        return row

    for p in range(k):
        # (a) L[p,c] >= 0   <=>   -L[p,c] <= 0
        for c in range(C):
            A_ub.append(-load_row(p, c)); b_ub.append(0.0)
        # (b) sum_c L[p,c] <= Q
        cap_row = np.zeros(nvars)
        for c in range(C):
            cap_row += load_row(p, c)
        A_ub.append(cap_row); b_ub.append(float(Q))

    # (c) end empty: L[k-1, c] == 0
    for c in range(C):
        A_eq.append(load_row(k - 1, c)); b_eq.append(0.0)

    res = linprog(
        c=cost,
        A_ub=np.array(A_ub), b_ub=np.array(b_ub),
        A_eq=np.array(A_eq), b_eq=np.array(b_eq),
        bounds=list(zip(lb, ub)),
        method="highs",
    )

    if not res.success:
        return {"feasible_time": True, "time": time, "objective": 0, "plan": None}

    delivered = -res.fun                       # flip sign back to a positive count
    objective = int(round(delivered))

    # sanity: LP optimum should be (essentially) integer for these instances
    assert abs(delivered - objective) < 1e-4, f"non-integer LP optimum: {delivered}"

    # Build a readable plan: per stop, pickups (+) and dropoffs (-) per commodity
    plan = []
    for p in range(k):
        entry = {"node": seq[p], "pickup": {}, "dropoff": {}}
        for c in range(C):
            if (p, c) in var_index:
                val = res.x[var_index[(p, c)]]
                amt = int(round(val))
                if amt > 0:
                    entry["pickup"][c] = amt
                elif amt < 0:
                    entry["dropoff"][c] = -amt
        plan.append(entry)

    return {"feasible_time": True, "time": time, "objective": objective, "plan": plan}


# ----------------------------- TESTS -----------------------------
if __name__ == "__main__":

    # A tiny fake instance so we can test the LP logic in isolation.
    def fake(coords, demand, Q):
        import math
        n = len(coords)
        t = [[math.dist(coords[i], coords[j]) for j in range(n)] for i in range(n)]
        return {"n": n, "capacity": Q, "coords": coords, "demand": demand,
                "t": t, "L": 999.0, "num_commodities": len(demand[0])}

    print("TEST 1 - apple example (single commodity, expect 9)")
    # depot + A(+4) B(-3) C(+5) D(-6); coords near origin so time is trivial
    inst1 = fake(
        coords=[(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        demand=[[0], [4], [-3], [5], [-6]],
        Q=10,
    )
    r1 = evaluate([1, 2, 3, 4], inst1)   # A,B,C,D
    print("   objective:", r1["objective"], " (expected 9)")

    print("\nTEST 2 - the capacity trap (Q=2, expect 2, NOT 0)")
    # A supplies 2 of product1; B supplies 2 of product2; C demands 2 of product2
    inst2 = fake(
        coords=[(0, 0), (1, 0), (2, 0), (3, 0)],
        demand=[[0, 0], [2, 0], [0, 2], [0, -2]],
        Q=2,
    )
    r2 = evaluate([1, 2, 3], inst2)      # A,B,C
    print("   objective:", r2["objective"], " (expected 2 - smart plan skips product 1)")
    print("   plan:", r2["plan"])

    print("\nTEST 3 - real instance, a hand-picked route")
    inst = read_instance("/mnt/user-data/uploads/m2n20Q10s000c2__2_.tsp")
    seq = [5, 4, 3, 6]   # arbitrary customers (0-based indices)
    r3 = evaluate(seq, inst)
    print("   sequence (0-based):", seq)
    print("   time:", round(r3["time"], 3), "h   feasible:", r3["feasible_time"])
    print("   objective:", r3["objective"])
