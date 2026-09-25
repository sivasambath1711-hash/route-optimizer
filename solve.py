"""
MIS41480 group project - selective m-PDTSP solver (executable).

Usage:
    python solve.py

It asks for an instance file name (e.g. m2n20Q10s000c2.tsp), then runs the
Iterated Local Search algorithm for 5 minutes. For every solution generated
during the iterations of the algorithm it prints, to the console:
    - the sequence of nodes visited (depot 1 -> customers -> depot 1)
    - the objective function value (total units delivered)
    - the time (seconds since start) at which that solution was generated
Solutions that improve on all previous ones are additionally marked NEW BEST.

Node numbering in the printed sequence is 1-based, matching the instance file
(node 1 = depot). Internally the code is 0-based.
"""

import os
import sys
import time

from parser import read_instance
from ils import ils

TIME_LIMIT = 300.0        # 5 minutes, as required by the brief
SEED = 0                  # fixed seed -> reproducible; change for replications
ACCEPTANCE = "annealing"  # main method (see report Section 4.6)


def find_instance(name):
    """Locate the instance file: try as given, then a few common locations."""
    candidates = [
        name,
        os.path.join(os.getcwd(), name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        os.path.join("/mnt/user-data/uploads", name),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def format_sequence(seq):
    """depot -> customers -> depot, in 1-based file numbering."""
    nodes = [1] + [i + 1 for i in seq] + [1]
    return " -> ".join(str(x) for x in nodes)


def main():
    name = input("Enter instance file name (e.g. m2n20Q10s000c2.tsp): ").strip()
    path = find_instance(name)
    if path is None:
        print(f"ERROR: could not find instance file '{name}'.")
        sys.exit(1)

    inst = read_instance(path)
    print(f"\nLoaded {name}: {inst['n'] - 1} customers, capacity Q={inst['capacity']}, "
          f"limit L={inst['L']} h")
    print(f"Running Iterated Local Search for {TIME_LIMIT:.0f} seconds...\n")
    print("Every solution generated during the iterations is listed below.")
    print("Lines marked *** NEW BEST *** improve on all previous solutions.")
    print("-" * 70)

    def report_candidate(elapsed, obj, seq, rtime):
        # printed for EVERY solution the algorithm generates
        print(f"[t = {elapsed:7.2f}s]  objective = {obj:<3}  "
              f"seq = {format_sequence(seq)}")

    def report_best(elapsed, obj, seq, rtime):
        # printed additionally when a new best is found
        print(f"    *** NEW BEST ***  objective = {obj}  at t = {elapsed:.2f}s")

    (best_seq, best_obj, best_time), history, iters = ils(
        inst, TIME_LIMIT, seed=SEED, acceptance=ACCEPTANCE,
        on_improve=report_best, on_candidate=report_candidate
    )

    print("-" * 70)
    print(f"\nFINAL best objective = {best_obj}")
    print(f"FINAL sequence       = {format_sequence(best_seq)}")
    print(f"FINAL route time     = {best_time:.3f} h")
    print(f"(perturbation cycles completed: {iters})")


if __name__ == "__main__":
    main()
