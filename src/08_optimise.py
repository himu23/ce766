"""
Step 8 - NSGA-II search for hedging policies on the 1991-2025 record.

Levers    : 27 (see fast.py): 12 monthly triggers, 12 rationing factors, 3 e-flow factors
Objectives: max supply reliability, min worst 7-day supply deficit, max e-flow
            adequacy of the worse-served season (Jun-Sep or Oct-May),
            min mean annual spill
Search    : pymoo NSGA-II (Deb et al., 2002), population 200, 500 generations, 10 seeds
            (odd seeds: four objectives, even seeds: spill left out, which concentrates the
            population on the supply/e-flow trade-off), then a four-objective refinement run
            (population 2000, 500 generations) started from the
            merged non-dominated set. Hypervolume is logged per seed.

Output: outputs/pareto_policies.csv      levers + measures of the merged Pareto set
        outputs/optimisation_log.json    hypervolume by generation/seed, baselines
"""
import json
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import config as C
import fast

POP, GEN, SEEDS = 200, 500, range(1, 11)
POP2, GEN2 = 2000, 500               # refinement run started from the merged seed results
SPILL_SCALE = 200_000.0                  # ML/yr, only scales objective 4
REF = np.array([0.5, 1.0, 1.0, 1.0])     # hypervolume reference (failure, deficit, e-flow shortfall, spill)
NAMES = ["reliability", "worst7_deficit", "eflow_adequacy", "resilience", "event_deficit_ML",
         "n_events", "shortfall", "eflow_compliance", "max_daily_deficit", "spill_ML_yr",
         "adequacy_dry", "adequacy_mon"]
LEVERS = [f"k{m:02d}" for m in range(1, 13)] + [f"a{m:02d}" for m in range(1, 13)] + \
         ["b_dry", "b_mon", "g"]


def load_record(natural_cap=False):
    fz = pd.read_csv(C.PROC / "forcing_daily.csv", parse_dates=["date"], index_col="date")
    q = pd.read_csv(C.PROC / "inflow_simulated.csv", parse_dates=["date"], index_col="date")
    I = q.I_ML.values
    maf = I.mean()
    T = np.where(fz.index.month.isin(C.MONSOON_MONTHS), C.EFLOW_FRAC_MONSOON, C.EFLOW_FRAC_DRY) * maf
    if natural_cap:                       # sensitivity: never ask for more than the river carried
        T = np.minimum(T, I)
    return dict(I=I, P=fz.P_mm.values, E=fz.PET_mm.values,
                month=fz.index.month.values.astype(np.int64), smax=float(C.S_MAX_ML),
                demand=float(C.D_TARGET_MLD), T=T, a_full=float(C.A_LAKE_FULL_KM2)), fz.index


def to_F(M):
    return np.c_[1 - M[:, 0], M[:, 1], 1 - M[:, 2], M[:, 9] / SPILL_SCALE]


NOHEDGE = np.r_[np.zeros(12), np.ones(12)]
BASELINES = {"current_practice": np.r_[NOHEDGE, 0, 0, 1],
             "sop_eflow": np.r_[NOHEDGE, 1, 1, 1]}


class Tansa(Problem):
    def __init__(self, rec, n_obj=4):
        super().__init__(n_var=fast.N_LEV, n_obj=n_obj, xl=fast.XL, xu=fast.XU)
        self.rec = rec

    def _evaluate(self, X, out, *a, **k):
        out["F"] = to_F(fast.evaluate_pop(X, **self.rec))[:, :self.n_obj]


def optimise(rec, seeds=SEEDS, pop2=POP2, gen2=GEN2, tag=""):
    hv = HV(ref_point=REF)
    start = np.array(list(BASELINES.values()))     # baselines included in every initial population
    allX, log = [], {}
    for s in seeds:
        trace = []

        def cb(alg):
            if alg.n_gen % 50 == 0 or alg.n_gen == 1:
                trace.append((int(alg.n_gen), float(hv(
                    to_F(fast.evaluate_pop(alg.opt.get("X"), **rec))))))
        X0 = fast.XL + np.random.default_rng(s).random((POP, fast.N_LEV)) * (fast.XU - fast.XL)
        X0[:len(start)] = start
        res = minimize(Tansa(rec, 4 if s % 2 else 3), NSGA2(pop_size=POP, sampling=X0), ("n_gen", GEN), seed=s,
                       callback=cb, copy_algorithm=False)
        log[s] = trace
        allX.append(res.X)
        print(f"{tag} seed {s}: {len(res.X)} solutions, HV {trace[-1][1]:.4f} "
              f"(gen 250: {dict(trace)[250]:.4f})", flush=True)

    def merge(X):
        M = fast.evaluate_pop(X, **rec)
        k = NonDominatedSorting().do(np.round(to_F(M), 6), only_non_dominated_front=True)
        _, u = np.unique(np.round(to_F(M[k]), 3), axis=0, return_index=True)
        return X[k][u], M[k][u]

    X, M = merge(np.vstack(allX))
    hv1 = float(hv(to_F(M)))
    # stage 2: refine the merged set with a larger population
    rng = np.random.default_rng(0)
    X0 = X[rng.permutation(len(X))[:pop2]]
    if len(X0) < pop2:
        X0 = np.vstack([X0, fast.XL + rng.random((pop2 - len(X0), fast.N_LEV)) * (fast.XU - fast.XL)])
    res = minimize(Tansa(rec), NSGA2(pop_size=pop2, sampling=X0), ("n_gen", gen2), seed=99,
                   copy_algorithm=False)
    X, M = merge(np.vstack([X, res.X]))
    print(f"{tag} {len(X)} policies; HV merged {hv1:.4f} -> refined {hv(to_F(M)):.4f}", flush=True)
    df = pd.DataFrame(np.c_[X, M], columns=LEVERS + NAMES)
    df.index.name = "policy"
    base = {n: dict(zip(NAMES, map(float, fast.evaluate(x, *rec.values()))))
            for n, x in BASELINES.items()}
    return df, dict(hv_trace=log, hv_merged=hv1, hv_refined=float(hv(to_F(M))),
                    n_policies=len(df), baselines=base)


if __name__ == "__main__":
    out = dict(settings=dict(pop=POP, gen=GEN, seeds=list(SEEDS), pop2=POP2, gen2=GEN2,
                             ref=REF.tolist()))
    rec, _ = load_record()
    df, out["tennant"] = optimise(rec, tag="tennant")
    df.to_csv(C.OUT / "pareto_policies.csv")
    recn, _ = load_record(natural_cap=True)
    dfn, out["natural_cap"] = optimise(recn, tag="natural-cap")
    dfn.to_csv(C.OUT / "pareto_policies_naturalcap.csv")
    json.dump(out, open(C.OUT / "optimisation_log.json", "w"), indent=1)
