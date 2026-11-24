"""
Step 10 - Re-evaluate every Pareto policy in 1000 alternative futures (deep uncertainty).

(a) Behavioural GR4J sets: 600,000 LHS parameter sets, kept if KGE >= 0.65 and volume
    bias within +/-15 % on the 27 storage blocks of step 4 (model-parameter uncertainty).
(b) 1000 states of the world by Latin hypercube sampling of six uncertain factors:
      rainfall multiplier   0.80 to the CMIP6 maximum   (all 12 CMIP6 runs are wetter; the
                            lower bound is extended so that drying is also tested)
      monsoon shift (days)  CMIP6 minimum to +14        (extended towards late onset)
      warming (deg C)       0 to the CMIP6 maximum
      draw-off (ML/day)     372-537, the spread implied by the BMC storage record (step 3)
      capacity loss         0-10 % of useful storage (assumed: sedimentation)
      GR4J parameter set    uniform over the behavioural sets
    Each future perturbs the 1991-2025 daily record; PET is recomputed (Oudin) and
    GR4J is rerun. The Tennant target stays fixed in ML/day.
(c) Robustness = share of futures in which a policy satisfies all three criteria
    (satisficing), plus minimax regret as a second criterion.

Output: outputs/sow.csv, outputs/robustness.csv, outputs/sow_measures.npz,
        data/processed/gr4j_behavioural.csv
"""
import importlib
import json
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.stats import qmc
import config as C
import fast
from models import kge, GR4J_BOUNDS

opt = importlib.import_module("08_optimise")
N_SOW, N_GLUE, KGE_MIN, BIAS_MAX = 1000, 600_000, 0.65, 0.15
CRIT = dict(reliability=0.90, worst7_deficit=0.20, eflow_adequacy=0.50)

A_CATCH = json.load(open(C.PROC / "catchment_info.json"))["area_km2"]
fz = pd.read_csv(C.PROC / "forcing_daily.csv", parse_dates=["date"], index_col="date")
P, E, Tm = fz.P_mm.values, fz.PET_mm.values, fz.Tmean_C.values


@njit(parallel=True)
def glue_blocks(PAR, P, E, i0, i1):
    out = np.empty((len(PAR), len(i0)))
    for k in prange(len(PAR)):
        q = fast.gr4j_nb(P, E, PAR[k, 0], PAR[k, 1], PAR[k, 2], PAR[k, 3])
        for b in range(len(i0)):
            out[k, b] = q[i0[b]:i1[b]].mean()
    return out


def behavioural():
    blk = pd.read_csv(C.PROC / "inflow_blocks_fit.csv", parse_dates=["start", "end"])
    sub = fz.loc["2020":]                                   # 3.5-year warm-up is enough
    i0, i1 = sub.index.get_indexer(blk.start), sub.index.get_indexer(blk.end)
    lo, hi = np.array(GR4J_BOUNDS).T
    par = qmc.scale(qmc.LatinHypercube(4, seed=1).random(N_GLUE), lo, hi)
    sim = glue_blocks(par, sub.P_mm.values, sub.PET_mm.values, i0, i1)
    obs = blk.I_mm_d.values
    k = np.array([kge(s, obs)[0] for s in sim])
    nd = blk.ndays.values
    bias = (sim * nd).sum(1) / (obs * nd).sum() - 1
    ok = (k >= KGE_MIN) & (np.abs(bias) <= BIAS_MAX)
    df = pd.DataFrame(par[ok], columns=["X1", "X2", "X3", "X4"]).assign(KGE=k[ok])
    df.to_csv(C.PROC / "gr4j_behavioural.csv", index=False)
    print(f"GLUE: {ok.sum()} of {N_GLUE} sets with KGE >= {KGE_MIN}, |bias| <= {BIAS_MAX} (best {k.max():.3f})")
    print(df.describe().loc[["min", "50%", "max"]].round(2).to_string())
    return df


def sample_sow(beh, n=N_SOW, seed=7):
    f = pd.read_csv(C.PROC / "cmip6_factors.csv")
    lo = [0.80, f.onset_shift_d.min(), 0.0, 372, 0.0, 0]
    hi = [f.rain_ratio.max(), 14, f.dT_C.max(), 537, 0.10, len(beh)]
    u = qmc.scale(qmc.LatinHypercube(6, seed=seed).random(n), lo, hi)
    s = pd.DataFrame(u, columns=["rain_mult", "onset_shift_d", "dT_C", "demand", "cap_loss", "gr4j"])
    s["onset_shift_d"] = s.onset_shift_d.round().astype(int)
    s["gr4j"] = s.gr4j.astype(int).clip(0, len(beh) - 1)
    return s


def future(rec, beh, r):
    """Perturbed inputs for one state of the world (row r of the SOW table)."""
    p = np.roll(P, int(r.onset_shift_d)) * r.rain_mult
    e = E * np.maximum(Tm + r.dT_C + 5, 0) / np.maximum(Tm + 5, 1e-9)      # Oudin is linear in T+5
    b = beh.iloc[int(r.gr4j)]
    q = fast.gr4j_nb(p, e, b.X1, b.X2, b.X3, b.X4) * A_CATCH
    return dict(I=q, P=p, E=e, month=rec["month"], smax=rec["smax"] * (1 - r.cap_loss),
                demand=float(r.demand), T=rec["T"], a_full=rec["a_full"])


def satisfies(M):
    return ((M[..., 0] >= CRIT["reliability"]) & (M[..., 1] <= CRIT["worst7_deficit"]) &
            (M[..., 2] >= CRIT["eflow_adequacy"]))


if __name__ == "__main__":
    rec, _ = opt.load_record()
    beh = behavioural()
    sow = sample_sow(beh)
    pol = pd.read_csv(C.OUT / "pareto_policies.csv", index_col=0)
    frob = C.OUT / "pareto_policies_robust.csv"          # from step 11, if already run
    if frob.exists():
        pol = pd.concat([pol, pd.read_csv(frob, index_col=0)])
    X = np.vstack([pol[opt.LEVERS].values] + [x for x in opt.BASELINES.values()])
    M = np.empty((N_SOW, len(X), len(opt.NAMES)))
    inflow = np.empty(N_SOW)
    for j, r in enumerate(sow.itertuples()):
        fut = future(rec, beh, r)
        inflow[j] = fut["I"].mean() * 365.25 / 1000
        M[j] = fast.evaluate_pop(X, **fut)
    sow["inflow_MCM_yr"] = inflow
    sow.to_csv(C.OUT / "sow.csv", index_label="sow")
    np.savez_compressed(C.OUT / "sow_measures.npz", M=M[:, :, [0, 1, 2, 3, 7, 9]].astype("float32"))

    ok = satisfies(M)
    hist = fast.evaluate_pop(X, **rec)
    F = np.stack([1 - M[..., 0], M[..., 1], 1 - M[..., 2], M[..., 9]], -1)
    span = F.max((0, 1)) - F.min((0, 1))
    regret = ((F - F.min(axis=1, keepdims=True)) / span).max(-1)      # worst objective, per future
    rob = pd.DataFrame(hist[:, :3], columns=["hist_reliability", "hist_worst7", "hist_adequacy"],
                       index=list(pol.index) + list(opt.BASELINES))
    rob["set"] = ["baseline" if p in opt.BASELINES else "robust" if str(p).startswith("R")
                  else "historical" for p in rob.index]
    for n, i in [("resilience", 3), ("vulnerability_ML", 4), ("compliance", 7), ("spill_ML", 9), ("shortfall", 6)]:
        rob["hist_" + n] = hist[:, i]
    rob["hist_ok"] = satisfies(hist)
    rob["satisficing"] = ok.mean(0)
    rob["sat_reliability"] = (M[..., 0] >= CRIT["reliability"]).mean(0)
    rob["sat_worst7"] = (M[..., 1] <= CRIT["worst7_deficit"]).mean(0)
    rob["sat_adequacy"] = (M[..., 2] >= CRIT["eflow_adequacy"]).mean(0)
    rob["regret_max"] = regret.max(0)
    rob["regret_90"] = np.quantile(regret, 0.9, axis=0)
    for i, n in enumerate(["reliability", "worst7", "adequacy"]):
        rob[f"sow_median_{n}"] = np.median(M[..., i], axis=0)
        rob[f"sow_p10_{n}"] = np.quantile(M[..., i], 0.1, axis=0)
        rob[f"sow_p90_{n}"] = np.quantile(M[..., i], 0.9, axis=0)
    rob.index.name = "policy"
    rob.to_csv(C.OUT / "robustness.csv")
    show = ["hist_reliability", "hist_worst7", "hist_adequacy", "hist_ok", "satisficing",
            "sat_reliability", "sat_worst7", "sat_adequacy", "regret_90"]
    for g, d in rob.groupby("set"):
        print(g, len(d), "hist_ok:", int(d.hist_ok.sum()))
        print(d.sort_values("satisficing", ascending=False)[show].head(6).round(3).to_string())
