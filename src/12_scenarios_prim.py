"""
Step 12 - What happens to the selected policy, and when does it fail?
(a) RRV under history and under the 12 CMIP6 change-factor futures (2026-2060),
    at today's draw-off (455 ML/day) and at the top of the observed range (537 ML/day).
(b) Scenario discovery with PRIM (Friedman & Fisher, 1999; Bryant & Lempert, 2010):
    the box of uncertain factors that best isolates the futures in which it fails.
Output: outputs/cmip6_rrv.csv, outputs/prim.json
"""
import importlib
import json
import numpy as np
import pandas as pd
import config as C
import fast

opt = importlib.import_module("08_optimise")
rb = importlib.import_module("10_robustness")
FACT = ["rain_mult", "onset_shift_d", "dT_C", "demand", "cap_loss", "runoff_coeff"]


def select(rob):
    """Recommended policy: largest satisficing share among policies that also meet all
    three criteria on the 1991-2025 record."""
    d = rob[(rob.set != "baseline") & rob.hist_ok]
    return d.satisficing.idxmax()


def prim(X, y, alpha=0.05, min_mass=0.10):
    """PRIM peeling: repeatedly remove the 5 % slice that raises failure density the most."""
    inb = np.ones(len(y), bool)
    lim = {c: [X[c].min(), X[c].max()] for c in X}
    traj = []
    while inb.mean() > min_mass:
        traj.append(dict(density=y[inb].mean(), coverage=y[inb].sum() / y.sum(), mass=inb.mean(),
                         lim={c: list(v) for c, v in lim.items()}))
        best = None
        for c in X:
            v = X[c].values
            for side, q in ((0, np.quantile(v[inb], alpha)), (1, np.quantile(v[inb], 1 - alpha))):
                keep = inb & ((v > q) if side == 0 else (v < q))
                if keep.sum() == 0 or keep.sum() == inb.sum():
                    continue
                d = y[keep].mean()
                if best is None or d > best[0]:
                    best = (d, c, side, q, keep)
        if best is None or best[0] <= y[inb].mean():
            break
        _, c, side, q, inb = best
        lim[c][side] = float(q)
    return traj


if __name__ == "__main__":
    rec, _ = opt.load_record()
    rob = pd.read_csv(C.OUT / "robustness.csv", index_col=0)
    rob.index = rob.index.astype(str)
    pol = pd.read_csv(C.OUT / "pareto_policies.csv", index_col=0)
    frob = C.OUT / "pareto_policies_robust.csv"
    if frob.exists():
        pol = pd.concat([pol, pd.read_csv(frob, index_col=0)])
    pol.index = pol.index.astype(str)
    p = select(rob)
    x = pol.loc[p, opt.LEVERS].values.astype(float)
    print("selected policy", p)

    # (a) CMIP6 change-factor runs with the calibrated GR4J set
    par = json.load(open(C.PROC / "gr4j_params.json"))
    beh0 = pd.DataFrame([dict(X1=par["X1_mm"], X2=par["X2_mm_per_day"], X3=par["X3_mm"], X4=par["X4_days"])])
    fac = pd.read_csv(C.PROC / "cmip6_factors.csv")
    rows = []
    pols = {"recommended": x, **opt.BASELINES}
    for dem in (455.0, 537.0):
        runs = [("history", "1991-2025", 1.0, 0, 0.0)] + \
               [(r.scenario, r.model, r.rain_ratio, round(r.onset_shift_d), r.dT_C) for r in fac.itertuples()]
        for sc, m, rm, sh, dT in runs:
            r = pd.Series(dict(rain_mult=rm, onset_shift_d=sh, dT_C=dT, demand=dem, cap_loss=0.0, gr4j=0))
            fut = rb.future(rec, beh0, r)
            for n, xx in pols.items():
                rows.append(dict(demand=dem, scenario=sc, model=m, policy=n,
                                 **dict(zip(opt.NAMES, fast.evaluate(xx, *fut.values())))))
    rr = pd.DataFrame(rows)
    rr.to_csv(C.OUT / "cmip6_rrv.csv", index=False)
    print(rr.groupby(["policy", "demand", "scenario"])[["reliability", "resilience", "event_deficit_ML",
          "worst7_deficit", "eflow_adequacy", "eflow_compliance", "spill_ML_yr"]].median().round(3).to_string())

    # (b) PRIM on the 1000 test futures
    sow = pd.read_csv(C.OUT / "sow.csv", index_col=0)
    beh = pd.read_csv(C.PROC / "gr4j_behavioural.csv")
    A = rb.A_CATCH
    # reference runoff coefficient of each GR4J set under the historical forcing
    rc = np.array([fast.gr4j_nb(rb.P, rb.E, *b).mean() / rb.P.mean()
                   for b in beh[["X1", "X2", "X3", "X4"]].values[np.unique(sow.gr4j)]])
    sow["runoff_coeff"] = pd.Series(rc, index=np.unique(sow.gr4j)).reindex(sow.gr4j).values
    k = list(rob.index).index(p)
    M = np.load(C.OUT / "sow_measures.npz")["M"][:, k]
    fail = ~rb.satisfies(M)
    traj = prim(sow[FACT], fail)
    base = fail.mean()
    # largest box with density >= 0.9
    dense = [t for t in traj if t["density"] >= 0.9]
    box = dense[0] if dense else max(traj, key=lambda t: t["density"])
    rho = {c: float(sow[c].corr(pd.Series(fail.astype(float), index=sow.index), method="spearman")) for c in FACT}
    single = {"reliability": float((M[:, 0] < rb.CRIT["reliability"]).mean()),
              "worst7": float((M[:, 1] > rb.CRIT["worst7_deficit"]).mean()),
              "adequacy": float((M[:, 2] < rb.CRIT["eflow_adequacy"]).mean())}
    full = {c: [float(sow[c].min()), float(sow[c].max())] for c in FACT}
    json.dump(dict(policy=p, fail_rate=float(base), fails_by_criterion=single, spearman=rho, box=box,
                   full_range=full, trajectory=traj), open(C.OUT / "prim.json", "w"), indent=1, default=float)
    sow.assign(fail=fail).to_csv(C.OUT / "sow_selected.csv")
    print("fail rate", base, single)
    print("spearman", {a: round(b, 2) for a, b in rho.items()})
    print("PRIM box: density %.2f coverage %.2f" % (box["density"], box["coverage"]))
    for c in FACT:
        if box["lim"][c] != full[c]:
            print("  ", c, np.round(box["lim"][c], 3), "full", np.round(full[c], 3))
