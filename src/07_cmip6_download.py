"""
Step 7 - NEX-GDDP-CMIP6 daily point series at the Tansa catchment.

Data : NASA NEX-GDDP-CMIP6 v2 (Thrasher et al., 2022), bias-corrected and
       downscaled to 0.25 deg, read one grid cell (19.625 N, 73.375 E) at a time
       through the NCCS THREDDS subset service (no login, no global download).
Runs : 6 GCMs x {historical 1991-2014, ssp245 and ssp585 2026-2060} x {pr, tas}
Output: data/raw/cmip6/<model>_<scenario>_<var>.csv  (date, value; mm/day or deg C)

Re-running resumes from the per-year cache in data/raw/cmip6/cache/.
"""
import io
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import config as C

BASE = "https://ds.nccs.nasa.gov/thredds"
ROOT = "AMES/NEX/GDDP-CMIP6"
LAT, LON = 19.57, 73.32            # catchment centroid
MODELS = {"ACCESS-CM2": "r1i1p1f1", "EC-Earth3": "r1i1p1f1", "GFDL-ESM4": "r1i1p1f1",
          "MPI-ESM1-2-HR": "r1i1p1f1", "MRI-ESM2-0": "r1i1p1f1", "NorESM2-MM": "r1i1p1f1"}
PERIODS = {"historical": (1991, 2014), "ssp245": (2026, 2060), "ssp585": (2026, 2060)}
VARS = ("pr", "tas")
OUTD = C.RAW / "cmip6"
CACHE = OUTD / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def get(url, tries=6, timeout=180):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode()
        except Exception as e:
            err = e
            time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url}: {err}")


def year_files(model, member, scen, var):
    """Latest-version file name of each year in the THREDDS catalog."""
    html = get(f"{BASE}/catalog/{ROOT}/{model}/{scen}/{member}/{var}/catalog.html")
    best = {}
    for f in set(re.findall(rf"{var}_day_[\w\-.]+?\.nc", html)):
        m = re.search(r"_(\d{4})(_v[\d.]+)?\.nc$", f)
        if m and f > best.get(int(m[1]), ""):
            best[int(m[1])] = f
    return best


def fetch(job):
    model, member, scen, var, y, f = job
    out = CACHE / f"{model}_{scen}_{var}_{y}.csv"
    if out.exists():
        return
    q = urllib.parse.urlencode(dict(var=var, latitude=LAT, longitude=LON, accept="csv",
                                    time_start=f"{y}-01-01T00:00:00Z", time_end=f"{y}-12-31T23:59:59Z"))
    txt = get(f"{BASE}/ncss/grid/{ROOT}/{model}/{scen}/{member}/{var}/{f}?{q}")
    df = pd.read_csv(io.StringIO(txt))
    if not 360 <= len(df) <= 366:
        raise RuntimeError(f"{f}: {len(df)} rows")
    df.to_csv(out, index=False)


if __name__ == "__main__":
    jobs = []
    for model, member in MODELS.items():
        for scen, (y0, y1) in PERIODS.items():
            for var in VARS:
                files = year_files(model, member, scen, var)
                jobs += [(model, member, scen, var, y, files[y]) for y in range(y0, y1 + 1)]
    print(len(jobs), "point-years", flush=True)
    with ThreadPoolExecutor(8) as ex:
        for i, _ in enumerate(ex.map(fetch, jobs)):
            if i % 50 == 0:
                print(i, flush=True)
    for model in MODELS:
        for scen, (y0, y1) in PERIODS.items():
            for var in VARS:
                d = pd.concat([pd.read_csv(CACHE / f"{model}_{scen}_{var}_{y}.csv")
                               for y in range(y0, y1 + 1)])
                s = pd.Series(d.iloc[:, -1].values, index=pd.to_datetime(d["time"]).dt.date, name=var)
                s = s * 86400 if var == "pr" else s - 273.15
                s.index.name = "date"
                s.round(4).to_csv(OUTD / f"{model}_{scen}_{var}.csv")
    print("done")
