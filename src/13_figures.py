"""
Step 13 - Figures for the final submission.
Output: outputs/fig1_tradeoffs.png, outputs/fig2_robustness.png
"""
import importlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import config as C

opt = importlib.import_module("08_optimise")
rb = importlib.import_module("10_robustness")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
SEQ = LinearSegmentedColormap.from_list("seq", ["#86b6ef", "#2a78d6", "#0d366b"])
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.grid": True,
                     "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "axes.axisbelow": True})
H = pd.read_csv(C.OUT / "pareto_policies.csv", index_col=0)
N = pd.read_csv(C.OUT / "pareto_policies_naturalcap.csv", index_col=0)
Rb = pd.read_csv(C.OUT / "pareto_policies_robust.csv", index_col=0)
rob = pd.read_csv(C.OUT / "robustness.csv", index_col=0); rob.index = rob.index.astype(str)
log = json.load(open(C.OUT / "optimisation_log.json"))
pr = json.load(open(C.OUT / "prim.json"))
sel = pr["policy"]
allp = pd.concat([H, Rb]); allp.index = allp.index.astype(str)
CR = rb.CRIT

# ---------------------------------------------------------------- Figure 1
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True, constrained_layout=True)
for ax, d, key, title in [(axs[0], H, "tennant", "(a) Fixed Tennant e-flow target"),
                          (axs[1], N, "natural_cap", "(b) Tennant target capped at natural inflow")]:
    d = d.sort_values("worst7_deficit", ascending=False)
    sc = ax.scatter(100 * d.eflow_adequacy, 100 * d.reliability, c=100 * d.worst7_deficit, cmap=SEQ,
                    vmin=0, vmax=100, s=16, lw=0)
    b = log[key]["baselines"]
    for n, mk, lab in [("current_practice", "s", "Current practice (no release)"),
                       ("sop_eflow", "^", "Full supply + full e-flow, no hedging")]:
        ax.scatter(100 * b[n]["eflow_adequacy"], 100 * b[n]["reliability"], marker=mk, s=110,
                   color=ORANGE, edgecolor="w", lw=1.2, zorder=5, label=lab, clip_on=False)
    if key == "tennant":
        s = rob.loc[sel]
        ax.scatter(100 * s.hist_adequacy, 100 * s.hist_reliability, marker="*", s=330, color=ORANGE,
                   edgecolor=INK, lw=1, zorder=6, label="Recommended policy")
        ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=10)
    ax.set_title(title, loc="left", fontsize=12)
    ax.set_xlabel("E-flow adequacy, worse-served season (% of target)")
    ax.set_xlim(0, 102); ax.set_ylim(40, 101.5)
axs[0].set_ylabel("Supply reliability, 1991-2025 (% of days)")
cb = fig.colorbar(sc, ax=axs, pad=0.01)
cb.set_label("Vulnerability: worst 7-day supply deficit (% of demand)"); cb.outline.set_visible(False)
fig.savefig(C.OUT / "fig1_tradeoffs.png", dpi=250)

# ---------------------------------------------------------------- Figure 2
fig, axs = plt.subplots(1, 3, figsize=(15.5, 5.4), constrained_layout=True)
ax = axs[0]
m = np.arange(1, 13)
x = allp.loc[sel]
k = 100 * x[[f"k{i:02d}" for i in m]].values.astype(float)
a = x[[f"a{i:02d}" for i in m]].values.astype(float)
ax.bar(m, k, color="#b7d3f6", width=0.82, label="Hedging trigger (storage, % of capacity)")
for i in range(12):
    ax.text(m[i], k[i] + 1.5, f"{100 * a[i]:.0f}", ha="center", fontsize=9.5, color=INK)
ax.set_xticks(m, list("JFMAMJJASOND"))
ax.set_ylim(0, 119); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="x", visible=False)
ax.set_xlabel("Month"); ax.set_ylabel("Storage trigger (% of capacity)")
ax.set_title("(a) Recommended policy", loc="left", fontsize=12)
ax.text(0.02, 0.985, f"Numbers: supply when below trigger (% of demand)\n"
        f"E-flow release: {100*x.b_dry:.0f}% (Oct-May), {100*x.b_mon:.0f}% (Jun-Sep) of Tennant;\n"
        f"×{x.g:.2f} when below trigger", transform=ax.transAxes, va="top", fontsize=9.5, color=INK)

ax = axs[1]
for g, c, lab in [("historical", BLUE, "Optimised on 1991-2025"), ("robust", ORANGE, "Optimised on 30 futures")]:
    d = rob[rob.set == g]
    ax.scatter(100 * d.hist_adequacy, 100 * d.satisficing, s=13, color=c, lw=0, alpha=0.75, label=lab)
for n, mk, lab in [("current_practice", "s", "Current practice"), ("sop_eflow", "^", "No hedging + full e-flow")]:
    ax.scatter(100 * rob.loc[n].hist_adequacy, 100 * rob.loc[n].satisficing, marker=mk, s=90, color=MUTED,
               edgecolor="w", zorder=5, label=lab, clip_on=False)
s = rob.loc[sel]
ax.scatter(100 * s.hist_adequacy, 100 * s.satisficing, marker="*", s=330, color="w", edgecolor=INK, lw=1.4,
           zorder=6, label="Recommended policy")
ax.set_xlabel("E-flow adequacy on 1991-2025 record (%)")
ax.set_ylabel("Futures meeting all 3 criteria (% of 1000)")
ax.set_xlim(0, 102); ax.set_ylim(-1, 68)
ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95)
ax.set_title("(b) Robustness of every policy", loc="left", fontsize=12)

ax = axs[2]
sw = pd.read_csv(C.OUT / "sow_selected.csv", index_col=0)
lim = pr["box"]["lim"]; full = pr["full_range"]
restr = [c for c in lim if not np.allclose(lim[c], full[c])]
rank = sorted(pr["spearman"], key=lambda c: -abs(pr["spearman"][c]))
xv, yv = rank[0], rank[1]
LAB = {"rain_mult": "Rainfall (× 1991-2025)", "demand": "Draw-off (ML/day)", "onset_shift_d": "Monsoon delay (days)",
       "dT_C": "Warming (°C)", "cap_loss": "Capacity loss (fraction)", "runoff_coeff": "Runoff coefficient"}
for f, c, mk, lab in [(False, BLUE, "o", "Meets all criteria"), (True, ORANGE, "x", "Fails at least one")]:
    d = sw[sw.fail == f]
    ax.scatter(d[xv], d[yv], s=20, color=c, marker=mk, lw=1 if f else 0, alpha=0.8, label=lab)
x0, x1 = lim[xv]; y0, y1 = lim[yv]
ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=2, ls="--",
             label=f"PRIM failure box (density {pr['box']['density']:.2f},\ncoverage {pr['box']['coverage']:.2f})"))
ax.set_xlabel(LAB[xv]); ax.set_ylabel(LAB[yv])
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9, frameon=False)
ax.set_title("(c) Where the recommended policy fails", loc="left", fontsize=12)
fig.savefig(C.OUT / "fig2_robustness.png", dpi=250)
print("selected", sel, "PRIM restricts", restr, "axes", xv, yv)
