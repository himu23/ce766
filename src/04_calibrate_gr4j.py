"""
Step 4 - Calibrate and validate GR4J against the derived inflow blocks.

GR4J runs daily over the whole 1991-2025 forcing (so the warm-up is long);
for each storage block (>= 7 days between BMC reports) the simulated runoff is
summed over the same days and compared with the water-balance inflow.
Objective: KGE (Gupta et al., 2009) on block-mean flows, lake-full blocks excluded.

Output: data/processed/gr4j_params.json
        data/processed/inflow_simulated.csv  (GR4J inflow for the full 1991-2025 period)
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import config as C
from models import gr4j, kge, nse, GR4J_BOUNDS

info = json.load(open(C.PROC / "catchment_info.json"))
A_CATCH = info["area_km2"]
fz = pd.read_csv(C.PROC / "forcing_daily.csv", parse_dates=["date"], index_col="date")
blk = pd.read_csv(C.PROC / "inflow_derived.csv", parse_dates=["start", "end"])
blk = blk[~blk.exclude].reset_index(drop=True)

P, E = fz.P_mm.values, fz.PET_mm.values
# position of each block in the daily forcing; block covers days [start, end)
i0 = fz.index.get_indexer(blk.start)
i1 = fz.index.get_indexer(blk.end)


def block_means(q):
    cs = np.concatenate([[0.0], np.cumsum(q)])
    return (cs[i1] - cs[i0]) / (i1 - i0)


def in_period(period):
    return ((blk.start >= period[0]) & (blk.end <= pd.Timestamp(period[1]) + pd.Timedelta(days=1))).values


CAL, VAL = in_period(C.CAL_PERIOD), in_period(C.VAL_PERIOD)
obs = blk.I_mm_d.values


def objective(params):
    sim = block_means(gr4j(P, E, params))
    return 1 - kge(sim[CAL], obs[CAL])[0]


if __name__ == "__main__":
    print(f"Blocks: calibration {CAL.sum()}, validation {VAL.sum()}")
    res = differential_evolution(objective, GR4J_BOUNDS, seed=42, maxiter=60,
                                 popsize=12, tol=1e-4, polish=True, workers=1)
    best = res.x
    q_all = gr4j(P, E, best)
    sim = block_means(q_all)
    out = {"X1_mm": best[0], "X2_mm_per_day": best[1], "X3_mm": best[2], "X4_days": best[3]}
    for name, m, per in [("calibration", CAL, C.CAL_PERIOD), ("validation", VAL, C.VAL_PERIOD)]:
        k, r, a, b = kge(sim[m], obs[m])
        vol = blk.I_ML.values[m].sum()
        out[name] = dict(period=per, n_blocks=int(m.sum()), KGE=k, r=r, alpha=a, beta=b,
                         NSE=nse(sim[m], obs[m]),
                         volume_bias_pct=100 * ((sim[m] * blk.ndays.values[m]).sum() * A_CATCH / vol - 1))
    json.dump(out, open(C.PROC / "gr4j_params.json", "w"), indent=2, default=float)

    pd.DataFrame({"Q_mm": q_all, "I_ML": q_all * A_CATCH}, index=fz.index).to_csv(
        C.PROC / "inflow_simulated.csv")
    blk.assign(sim_mm_d=sim).to_csv(C.PROC / "inflow_blocks_fit.csv", index=False)

    print(json.dumps(out, indent=2, default=lambda x: round(float(x), 3)))
    yr = pd.Series(q_all, index=fz.index).resample("YE").sum()
    print(f"Mean annual simulated inflow: {yr.mean() * A_CATCH / 1000:.0f} MCM "
          f"(runoff coefficient {yr.mean() / fz.P_mm.resample('YE').sum().mean():.2f})")
