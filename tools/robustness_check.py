"""Robustness check: does the SHAP ranking pick the best first fix across many pipelines?

Builds 16 graphs under test with randomly scaled defect rates (each rate between 0.25x and
4x the notebook's) and different random draws. For each graph it compares three ways of
choosing the first data-quality control to apply:

  shap       the notebook's attribution ranking
  heuristic  the control whose defect signals fire on the largest share of wrong answers
  oracle     the control that actually recovers the most answers when applied (measured)

Run from the repository root (about 12 minutes on 2 CPUs):

    python tools/robustness_check.py

Writes docs/robustness_results.csv and prints a summary.
"""

import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_notebook  # noqa: E402

N_PIPELINES = 16
BASE_RATES = dict(attribute=0.04, duplicate=0.004, flip=0.002, delete=0.008)


def load_notebook_namespace():
    # Run the notebook's code cells up to, but not including, the first fix.
    ns = {"__name__": "robustness"}
    for kind, src, _ in build_notebook.CELLS:
        if kind != "code":
            continue
        if "choice = next_fix(D0, [])" in src:
            break
        exec(compile(src, "notebook", "exec"), ns)
    return ns


def main():
    ns = load_notebook_namespace()
    G_SOURCE, run_pipeline, evaluate, diagnose = (ns[k] for k in ("G_SOURCE", "run_pipeline", "evaluate", "diagnose"))
    FIXES, CONTROL, DEFECTS = ns["FIXES"], ns["CONTROL"], ns["DEFECTS"]

    rng = random.Random(2026)
    rows, t0 = [], time.time()
    for cfg in range(N_PIPELINES):
        rates = {k: v * np.exp(rng.uniform(np.log(0.25), np.log(4))) for k, v in BASE_RATES.items()}
        G, _ = run_pipeline(G_SOURCE, rates, seed=100 + cfg)
        res = evaluate(G)
        dev = res[res.split == "dev"]
        wrong0 = dev.wrong.mean() * 1000

        shap_rank = [c for c in diagnose(res)["table"].index if c in FIXES]
        wrong = dev[dev.wrong == 1]
        heur = {c: (wrong[[f for f in DEFECTS if CONTROL[f] == c]] > 0).any(axis=1).mean() for c in FIXES}
        heur_rank = sorted(heur, key=heur.get, reverse=True)

        recovered = {}
        for c, fix in FIXES.items():
            H = G.copy()
            fix(H)
            r = evaluate(H)
            recovered[c] = wrong0 - r[r.split == "dev"].wrong.mean() * 1000
        best = max(recovered, key=recovered.get)
        order = list(FIXES)
        truth = [-recovered[c] for c in order]

        rows.append(dict(
            pipeline=cfg, baseline_test_accuracy=1 - res[res.split == "test"].wrong.mean(),
            best_by_measurement=best, shap_top=shap_rank[0], heuristic_top=heur_rank[0],
            shap_hit=shap_rank[0] == best, heuristic_hit=heur_rank[0] == best,
            shap_regret_per_1000=recovered[best] - recovered[shap_rank[0]],
            heuristic_regret_per_1000=recovered[best] - recovered[heur_rank[0]],
            shap_rank_corr=spearmanr([shap_rank.index(c) for c in order], truth).correlation,
            heuristic_rank_corr=spearmanr([heur_rank.index(c) for c in order], truth).correlation,
            **{f"rate_{k}": round(v, 4) for k, v in rates.items()},
        ))
        print(f"pipeline {cfg:2d}  {time.time() - t0:5.0f}s  best={best:22s} shap={shap_rank[0]:22s} "
              f"heuristic={heur_rank[0]}", flush=True)

    df = pd.DataFrame(rows)
    out = Path("docs/robustness_results.csv")
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    n = len(df)
    print(f"\nBest first fix chosen: SHAP {df.shap_hit.sum()} of {n}, heuristic {df.heuristic_hit.sum()} of {n}, "
          f"random about {n / len(FIXES):.0f} of {n}")
    print(f"Mean answers lost per 1,000 versus the best fix: SHAP {df.shap_regret_per_1000.mean():.1f}, "
          f"heuristic {df.heuristic_regret_per_1000.mean():.1f}")
    print(f"Mean rank correlation with measured recovery: SHAP {df.shap_rank_corr.mean():.2f}, "
          f"heuristic {df.heuristic_rank_corr.mean():.2f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
