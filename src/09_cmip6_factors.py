"""
Step 9 - Change factors from NEX-GDDP-CMIP6 (2026-2060 against 1991-2014).

For each GCM x SSP: rainfall ratio (annual), monsoon-onset shift (days), warming (deg C).
Onset = first day after 1 May on which accumulated rainfall reaches 10 % of that
period's own mean annual total, so the timing signal is not mixed with the volume signal.
Output: data/processed/cmip6_factors.csv
"""
import numpy as np
import pandas as pd
import config as C

D = C.RAW / "cmip6"
MODELS = sorted({f.name.split("_")[0] for f in D.glob("*_historical_pr.csv")})


def read(m, s, v):
    d = pd.read_csv(D / f"{m}_{s}_{v}.csv", parse_dates=["date"], index_col="date").iloc[:, 0]
    return d


def onset(p, frac=0.10):
    ann = p.groupby(p.index.year).sum().mean()
    out = []
    for y, g in p.groupby(p.index.year):
        c = g[f"{y}-05-01":].cumsum()
        out.append((c >= frac * ann).idxmax().dayofyear - int(c.index[0].is_leap_year)
                   if (c >= frac * ann).any() else np.nan)
    return pd.Series(out)


rows = []
for m in MODELS:
    ph, th = read(m, "historical", "pr"), read(m, "historical", "tas")
    oh = onset(ph)
    for s in ("ssp245", "ssp585"):
        pf, tf = read(m, s, "pr"), read(m, s, "tas")
        of = onset(pf)
        rows.append(dict(model=m, scenario=s,
                         hist_rain_mm=ph.groupby(ph.index.year).sum().mean(),
                         rain_ratio=pf.groupby(pf.index.year).sum().mean() / ph.groupby(ph.index.year).sum().mean(),
                         onset_hist_doy=oh.mean(), onset_shift_d=of.mean() - oh.mean(),
                         onset_sd_hist=oh.std(), onset_sd_fut=of.std(),
                         dT_C=tf.mean() - th.mean()))
df = pd.DataFrame(rows)
df.to_csv(C.PROC / "cmip6_factors.csv", index=False)
print(df.round(2).to_string())
obs = pd.read_csv(C.PROC / "forcing_daily.csv", parse_dates=["date"], index_col="date").P_mm
oo = onset(obs)
print(f"IMD onset: mean doy {oo.mean():.1f}, sd {oo.std():.1f}, 1991-2014 rain {obs[:'2014'].groupby(obs[:'2014'].index.year).sum().mean():.0f}")
