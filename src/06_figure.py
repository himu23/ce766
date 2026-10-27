"""
Step 6 - Figure 1 for the preliminary submission.
(a) derived (water-balance) vs GR4J inflow per storage block, calibration + validation
(b) simulated vs BMC-observed useful storage, 2023-2025 (checks the whole model chain)
(c) baseline storage 1991-2025 under current practice, with supply-failure days
Output: outputs/fig1_preliminary.png
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import config as C

par = json.load(open(C.PROC / "gr4j_params.json"))
blk = pd.read_csv(C.PROC / "inflow_blocks_fit.csv", parse_dates=["start", "end"])
obs_st = pd.read_csv(C.RAW / "tansa_storage.csv", parse_dates=["date"], index_col="date")
base = pd.read_csv(C.OUT / "baseline_timeseries.csv", parse_dates=[0], index_col=0)
base_ef = pd.read_csv(C.OUT / "baseline_eflow_timeseries.csv", parse_dates=[0], index_col=0)
chk = pd.read_csv(C.OUT / "storage_check_2023_2025.csv", parse_dates=[0], index_col=0)

fig = plt.figure(figsize=(10, 8.2), constrained_layout=True)
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.05])
ax_a = fig.add_subplot(gs[0, :])
ax_b = fig.add_subplot(gs[1, :])
ax_c = fig.add_subplot(gs[2, :])

# (a) block inflows drawn as steps over each block's days
for _, r in blk.iterrows():
    ax_a.plot([r.start, r.end], [r.I_mm_d, r.I_mm_d], color="0.35", lw=1.6)
    ax_a.plot([r.start, r.end], [r.sim_mm_d, r.sim_mm_d], color="#1f6fb4", lw=1.6)
ax_a.plot([], [], color="0.35", lw=1.6, label="Derived from BMC storage (water balance)")
ax_a.plot([], [], color="#1f6fb4", lw=1.6, label="GR4J simulated")
ax_a.axvline(pd.Timestamp(C.VAL_PERIOD[0]), color="k", ls="--", lw=0.8)
c, v = par["calibration"], par["validation"]
ax_a.text(0.01, 0.9, f"Calibration KGE = {c['KGE']:.2f} (n = {c['n_blocks']})",
          transform=ax_a.transAxes, fontsize=9)
ax_a.text(0.60, 0.9, f"Validation KGE = {v['KGE']:.2f} (n = {v['n_blocks']})",
          transform=ax_a.transAxes, fontsize=9)
ax_a.set_ylabel("Inflow (mm/day,\nblock mean)")
ax_a.legend(loc="center left", fontsize=8, frameon=False, bbox_to_anchor=(0.0, 0.62))
ax_a.set_title("(a) Tansa inflow per ≥7-day storage block (lake-full blocks excluded)",
               loc="left", fontsize=10)
ax_a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# (b) model chain check against observed storage
ax_b.plot(chk.index, 100 * chk.storage_ML / C.S_MAX_ML, color="#1f6fb4", lw=1,
          label="Simulated (GR4J inflow + reservoir model, current practice)")
o = obs_st.loc[chk.index.min():chk.index.max()]
ax_b.scatter(o.index, 100 * o.storage_ML / C.S_MAX_ML, s=5, color="0.2", zorder=3,
             label="BMC 6 AM report")
ax_b.set_ylim(0, 105)
ax_b.set_ylabel("Useful storage\n(% of full)")
ax_b.legend(loc="lower left", fontsize=8, frameon=False)
ax_b.set_title("(b) Simulated vs observed storage, initialised from the first BMC report",
               loc="left", fontsize=10)
ax_b.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

# (c) 35-year baselines: current practice vs SOP + Tennant e-flow release
fail = base_ef.supply_ML < base_ef.supply_req_ML - 1e-6
ax_c.fill_between(base_ef.index, 0, 105, where=fail, color="#d62728", alpha=0.3,
                  step="mid", lw=0, label="Supply failure day (SOP + e-flow)")
ax_c.plot(base.index, 100 * base.storage_ML / C.S_MAX_ML, color="#1f6fb4", lw=0.6,
          label="Current practice (no e-flow release)")
ax_c.plot(base_ef.index, 100 * base_ef.storage_ML / C.S_MAX_ML, color="#e08a00", lw=0.6,
          label="SOP + Tennant e-flow release")
ax_c.set_ylim(0, 105)
ax_c.set_ylabel("Useful storage\n(% of full)")
ax_c.set_title(f"(c) Baseline runs, {C.D_TARGET_MLD:.0f} ML/day draw-off, "
               f"{C.HIST_START}-{C.HIST_END}", loc="left", fontsize=10)
ax_c.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.9, ncol=3)
fig.savefig(C.OUT / "fig1_preliminary.png", dpi=200)
print("saved outputs/fig1_preliminary.png")
