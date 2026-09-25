"""
Automated experiment runner for the MIS41480 project.

Runs the algorithm on every instance with 5 different random seeds, each for the
full 5-minute budget, then writes:

  results_per_run.csv   - one row per (instance, seed): objective, gap, time, iters
  results_summary.csv   - per instance: min / max / mean objective and gap
  convergence.png       - best-so-far vs time, one panel per instance (seed 0)
  console output        - a readable summary table

Just run:   python run_experiments.py
Then leave it; ~100 minutes total (20 runs x 5 minutes).

To do a quick dry-run first, set TIME_LIMIT below to e.g. 20 and SEEDS to [0, 1].
"""

import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parser import read_instance
from ils import ils

# ---- experiment configuration ----
TIME_LIMIT = 300.0                 # 5 minutes per run (the brief's requirement)
SEEDS = [0, 1, 2, 3, 4]            # 5 replications

INSTANCES = {
    "m2n20Q10s000c2": ("m2n20Q10s000c2.tsp", 33),
    "m2n20Q20s000c2": ("m2n20Q20s000c2.tsp", 37),
    "m2n30Q10s000c2": ("m2n30Q10s000c2.tsp", 60),
    "m2n30Q20s000c2": ("m2n30Q20s000c2.tsp", 61),
}


def main():
    per_run = []                    # rows for results_per_run.csv
    summary = []                    # rows for results_summary.csv
    convergence = {}                # instance -> history (seed 0) for the plot

    grand_start = time.time()

    for name, (filename, ub) in INSTANCES.items():
        inst = read_instance(filename)
        objs, gaps = [], []
        print(f"\n=== {name}  (upper bound {ub}) ===")

        for seed in SEEDS:
            t0 = time.time()
            (seq, obj, rtime), history, iters = ils(inst, TIME_LIMIT, seed=seed)
            gap = (ub - obj) / obj
            objs.append(obj); gaps.append(gap)
            if seed == SEEDS[0]:
                convergence[name] = (ub, history)

            per_run.append({
                "instance": name, "seed": seed, "objective": obj,
                "upper_bound": ub, "gap": round(gap, 4),
                "route_time_h": round(rtime, 3), "iterations": iters,
                "wall_seconds": round(time.time() - t0, 1),
            })
            print(f"  seed {seed}: obj={obj:>3}  gap={gap:6.1%}  "
                  f"time={rtime:5.2f}h  iters={iters}")

        summary.append({
            "instance": name, "upper_bound": ub,
            "min_obj": min(objs), "max_obj": max(objs),
            "mean_obj": round(sum(objs) / len(objs), 2),
            "min_gap": round(min(gaps), 4), "max_gap": round(max(gaps), 4),
            "mean_gap": round(sum(gaps) / len(gaps), 4),
        })
        print(f"  --> min={min(objs)} max={max(objs)} "
              f"mean={sum(objs)/len(objs):.1f}  mean gap={sum(gaps)/len(gaps):.1%}")

    # ---- write CSVs ----
    with open("results_per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        w.writeheader(); w.writerows(per_run)
    with open("results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    # ---- convergence figure (seed 0) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (name, (ub, hist)) in zip(axes.flat, convergence.items()):
        ts = [h[0] for h in hist] + [TIME_LIMIT]
        ys = [h[1] for h in hist] + [hist[-1][1]]
        ax.step(ts, ys, where="post", color="#3778C2", lw=1.8, label="Best found")
        ax.axhline(ub, ls="--", color="#A32D2D", lw=1.2, label=f"Upper bound = {ub}")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("time (seconds)"); ax.set_ylabel("objective")
        ax.grid(True, linewidth=0.3, alpha=0.4); ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Convergence over the 5-minute budget (seed 0)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("convergence.png", dpi=140, bbox_inches="tight")

    total_min = (time.time() - grand_start) / 60
    print(f"\nDone in {total_min:.1f} minutes.")
    print("Wrote results_per_run.csv, results_summary.csv, convergence.png")


if __name__ == "__main__":
    main()
