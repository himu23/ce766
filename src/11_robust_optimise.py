"""
Step 11 - NSGA-II again, but each policy is scored on 30 training futures (separate
LHS draw from the 1000 test futures of step 10) instead of the single 1991-2025 record.
Objectives: the same four measures averaged over the training futures.
Output: outputs/pareto_policies_robust.csv (then re-run step 10, which tests both sets)
"""
import importlib
import numpy as np
import pandas as pd
from numba import njit, prange
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import config as C
import fast

opt = importlib.import_module("08_optimise")
rb = importlib.import_module("10_robustness")
N_TRAIN, POP, GEN, SEEDS = 30, 300, 500, range(1, 7)


@njit(parallel=True, cache=True)
def eval_ens(X, I, P, E, month, smax, demand, T, a_full):
    nx, ns = X.shape[0], I.shape[0]
    out = np.zeros((nx, 4))
    for k in prange(nx):
        for j in range(ns):
            m = fast.evaluate(X[k], I[j], P[j], E[j], month, smax[j], demand[j], T, a_full)
            out[k, 0] += (1 - m[0]) / ns
            out[k, 1] += m[1] / ns
            out[k, 2] += (1 - m[2]) / ns
            out[k, 3] += m[9] / 200_000.0 / ns
    return out


class Robust(Problem):
    def __init__(self, ens, n_obj=4):
        super().__init__(n_var=fast.N_LEV, n_obj=n_obj, xl=fast.XL, xu=fast.XU)
        self.ens = ens

    def _evaluate(self, X, out, *a, **k):
        out["F"] = eval_ens(X, **self.ens)[:, :self.n_obj]


if __name__ == "__main__":
    rec, _ = opt.load_record()
    beh = pd.read_csv(C.PROC / "gr4j_behavioural.csv")
    tr = rb.sample_sow(beh, n=N_TRAIN, seed=101)
    fut = [rb.future(rec, beh, r) for r in tr.itertuples()]
    ens = dict(I=np.array([f["I"] for f in fut]), P=np.array([f["P"] for f in fut]),
               E=np.array([f["E"] for f in fut]), month=rec["month"],
               smax=np.array([f["smax"] for f in fut]),
               demand=np.array([f["demand"] for f in fut]), T=rec["T"], a_full=rec["a_full"])
    hist = pd.read_csv(C.OUT / "pareto_policies.csv", index_col=0)[opt.LEVERS].values
    Xs, Fs = [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        X0 = fast.XL + rng.random((POP, fast.N_LEV)) * (fast.XU - fast.XL)
        X0[:100] = hist[rng.permutation(len(hist))[:100]]        # seed with historical policies
        res = minimize(Robust(ens, 4 if s % 2 else 3), NSGA2(pop_size=POP, sampling=X0), ("n_gen", GEN), seed=s,
                       copy_algorithm=False)
        Xs.append(res.X); Fs.append(eval_ens(res.X, **ens))
        print("seed", s, len(res.X), flush=True)
    X, F = np.vstack(Xs), np.vstack(Fs)
    k = NonDominatedSorting().do(np.round(F, 6), only_non_dominated_front=True)
    X, F = X[k], F[k]
    _, u = np.unique(np.round(F, 3), axis=0, return_index=True)
    X = X[u]
    df = pd.DataFrame(np.c_[X, fast.evaluate_pop(X, **rec)], columns=opt.LEVERS + opt.NAMES)
    df.index = [f"R{i}" for i in range(len(df))]
    df.index.name = "policy"
    df.to_csv(C.OUT / "pareto_policies_robust.csv")
    print(len(df), "robust policies")
