"""
Step 5 - Baseline (no-intervention) reservoir runs, 1991-2025.

(a) Current practice: full supply draw-off whenever water is available, no
    dedicated downstream release - the river only receives spill.
(b) SOP + e-flow: same, plus the Tennant environmental release whenever water
    is available (no hedging).
These are the reference points for the optimised hedging policies.

Output: outputs/baseline_timeseries.csv  (run a), outputs/baseline_eflow_timeseries.csv (run b)
        outputs/baseline_performance.json
"""
import json
import numpy as np
import pandas as pd
import config as C
from models import simulate_reservoir, performance

fz = pd.read_csv(C.PROC / "forcing_daily.csv", parse_dates=["date"], index_col="date")
q = pd.read_csv(C.PROC / "inflow_simulated.csv", parse_dates=["date"], index_col="date")
df = fz.join(q).loc[f"{C.HIST_START}-01-01":f"{C.HIST_END}-12-31"]

NO_HEDGE = dict(trigger=np.zeros(12), alpha=np.ones(12))
runs = {"current_practice": dict(NO_HEDGE, eflow_mult=0.0),
        "sop_with_eflow": dict(NO_HEDGE, eflow_mult=1.0)}

mons = df.P_mm[df.index.month.isin(C.MONSOON_MONTHS)]
mons = mons.groupby(mons.index.year).sum()
anom = 100 * (mons / mons.mean() - 1)

perf_all = {}
for name, pol in runs.items():
    sim = simulate_reservoir(df.I_ML.values, df.P_mm.values, df.PET_mm.values,
                             df.index, C, policy=pol, s0_frac=1.0)
    sim.to_csv(C.OUT / ("baseline_timeseries.csv" if name == "current_practice"
                        else "baseline_eflow_timeseries.csv"))
    perf = performance(sim)
    # failures grouped by drawdown year (Oct-Sep): the lake refilled by monsoon y must
    # last until the monsoon of y+1 arrives, so pre-onset failures belong to monsoon y
    fail = sim.supply_ML < sim.supply_req_ML - 1e-6
    wy = np.where(sim.index.month >= 10, sim.index.year, sim.index.year - 1)
    perf["failure_water_years"] = {f"{y}-{(y + 1) % 100:02d}": dict(
        fail_days=int(fail.values[wy == y].sum()),
        monsoon_anomaly_pct=round(float(anom.get(y, np.nan)), 1)) for y in sorted(set(wy[fail.values]))}
    perf["min_storage_frac"] = float(sim.storage_ML.min() / C.S_MAX_ML)
    perf["min_storage_date"] = str(sim.storage_ML.idxmin().date())
    perf["years_spilled"] = int((sim.spill_ML.resample("YE").sum() > 0).sum())
    perf["n_years"] = int(df.index.year.nunique())
    dry = ~sim.index.month.isin(C.MONSOON_MONTHS)
    downstream = sim.release_ML + sim.spill_ML
    perf["eflow_compliance_dry_season"] = float(
        np.mean(downstream[dry] >= sim.eflow_req_ML[dry] - 1e-6))
    perf["eflow_target_ML_d"] = dict(monsoon=float(sim.eflow_req_ML[~dry].iloc[0]),
                                     dry=float(sim.eflow_req_ML[dry].iloc[0]))
    perf_all[name] = perf

# Model-chain check: start from the first continuous BMC report and compare the
# simulated storage (GR4J inflow -> reservoir model) with the observed reports.
obs = pd.read_csv(C.RAW / "tansa_storage.csv", parse_dates=["date"], index_col="date").storage_ML
t0 = obs.loc["2023-06-01":].index[0]
win = df.loc[t0:]
chk = simulate_reservoir(win.I_ML.values, win.P_mm.values, win.PET_mm.values, win.index, C,
                         policy=runs["current_practice"], s0_frac=obs.loc[t0] / C.S_MAX_ML)
chk.to_csv(C.OUT / "storage_check_2023_2025.csv")
both = pd.concat([chk.storage_ML.shift(-1), obs], axis=1, join="inner").dropna()  # 6 AM report = end of previous day
both.columns = ["sim", "obs"]
perf_all["storage_check"] = dict(start=str(t0.date()), n_reports=len(both),
                                 r=float(both.sim.corr(both.obs)),
                                 rmse_pct_capacity=float(100 * np.sqrt(((both.sim - both.obs) ** 2).mean())
                                                         / C.S_MAX_ML),
                                 bias_pct_capacity=float(100 * (both.sim - both.obs).mean() / C.S_MAX_ML))

perf_all["inputs"] = dict(demand_MLD=C.D_TARGET_MLD, S_max_ML=C.S_MAX_ML,
                          mean_inflow_ML_d=float(df.I_ML.mean()),
                          mean_annual_inflow_MCM=float(df.I_ML.mean() * 365 / 1000),
                          demand_to_inflow=float(C.D_TARGET_MLD / df.I_ML.mean()))
json.dump(perf_all, open(C.OUT / "baseline_performance.json", "w"), indent=2, default=float)
print(json.dumps(perf_all, indent=2, default=lambda x: round(float(x), 3)))
