"""
Step 3 - Back-calculate reservoir inflow from the water balance.

There is no gauge above Tansa dam, so "observed" inflow is reconstructed from
the BMC 6 AM useful-content reports. The record has gaps, so the balance is
closed over the interval between two reports (t0, t1) rather than day by day:

    I[t0,t1] = S(t1) - S(t0) + sum_t [ D + E(t)*A(S) - P(t)*A(S) ]

and intervals are merged into blocks of >= 7 days to damp level-reading noise
(1 cm of lake level ~ 190 ML ~ 1.4 mm over the catchment).

Inputs
  data/raw/tansa_storage.csv     : date, storage_ML (BMC 6 AM report, useful content)
  data/processed/forcing_daily.csv
Outputs
  data/processed/inflow_derived.csv : one row per block (start, end, ndays, I_ML, I_mm_d, exclude)
  data/processed/demand_estimate.json : dry-season draw-off implied by the storage record

Blocks touching a full lake (>= 99 % of capacity) are excluded because spill is
not reported, so inflow there is only a lower bound.
"""
import json
import numpy as np
import pandas as pd
import config as C

info = json.load(open(C.PROC / "catchment_info.json"))
A_CATCH = info["area_km2"]
MIN_BLOCK_DAYS = 7
FULL = 0.99 * C.S_MAX_ML

st = pd.read_csv(C.RAW / "tansa_storage.csv", parse_dates=["date"], index_col="date").sort_index()
fz = pd.read_csv(C.PROC / "forcing_daily.csv", parse_dates=["date"], index_col="date")
st = st[(st.index >= fz.index.min()) & (st.index <= fz.index.max())]

# Daily storage interpolated between reports - only used for the lake area A(S)
S_daily = st.storage_ML.reindex(pd.date_range(st.index.min(), st.index.max())).interpolate()
A_daily = C.A_LAKE_FULL_KM2 * (S_daily.clip(lower=0) / C.S_MAX_ML) ** C.AREA_EXPONENT  # km2
net_lake = ((C.PAN_COEFF * fz.PET_mm - fz.P_mm).reindex(A_daily.index) * A_daily)       # ML/day


def lake_term(t0, t1):
    """sum of (E - P)*A over the days between the 6 AM reports of t0 and t1."""
    return net_lake.loc[t0:t1 - pd.Timedelta(days=1)].sum()


# ---- 1. dry-season draw-off: Jan-May intervals, inflow assumed ~0 -------------
dry = st[st.index.month.isin([1, 2, 3, 4, 5])]
est = []
for (t0, s0), (t1, s1) in zip(dry.storage_ML.items(), list(dry.storage_ML.items())[1:]):
    n = (t1 - t0).days
    if 0 < n <= 200 and t1.month <= 5 and t0.year == t1.year:
        est.append(dict(start=str(t0.date()), end=str(t1.date()), days=n,
                        decline_ML_d=(s0 - s1) / n,
                        draw_off_ML_d=(s0 - s1 - lake_term(t0, t1)) / n))
json.dump(dict(intervals=est, note="draw-off = storage decline minus net lake evaporation, "
                                    "inflow assumed zero in Jan-May"),
          open(C.PROC / "demand_estimate.json", "w"), indent=2)

# ---- 2. inflow blocks ------------------------------------------------------
dates = list(st.index)
rows, i = [], 0
while i < len(dates) - 1:
    j = i + 1
    while j < len(dates) - 1 and (dates[j] - dates[i]).days < MIN_BLOCK_DAYS:
        j += 1
    t0, t1 = dates[i], dates[j]
    n = (t1 - t0).days
    S_seg = st.storage_ML.iloc[i:j + 1]
    I = S_seg.iloc[-1] - S_seg.iloc[0] + n * C.D_TARGET_MLD + lake_term(t0, t1)
    rows.append(dict(start=t0, end=t1, ndays=n, n_reports=j - i + 1,
                     S0_ML=S_seg.iloc[0], S1_ML=S_seg.iloc[-1], I_ML=I,
                     I_mm_d=I / A_CATCH / n,              # ML / km2 = mm
                     full_flag=bool((S_seg >= FULL).any()),
                     gap_flag=n > 45))
    i = j
out = pd.DataFrame(rows)
out["exclude"] = out.full_flag | out.gap_flag
out.to_csv(C.PROC / "inflow_derived.csv", index=False)

print(f"Storage reports used: {len(st)} ({st.index.min().date()} to {st.index.max().date()})")
for e in est:
    print(f"  dry-season {e['start']}..{e['end']}: decline {e['decline_ML_d']:.0f} ML/d, "
          f"implied draw-off {e['draw_off_ML_d']:.0f} ML/d")
print(f"Blocks: {len(out)}, excluded (lake full / long gap): {int(out.exclude.sum())}, "
      f"usable: {int((~out.exclude).sum())}")
print(f"Negative-inflow usable blocks: {int(((out.I_ML < 0) & ~out.exclude).sum())}")
ok = out[~out.exclude]
print(f"Usable blocks cover {ok.ndays.sum()} days; mean derived inflow "
      f"{ok.I_mm_d.mean():.1f} mm/d, max {ok.I_mm_d.max():.1f} mm/d")
