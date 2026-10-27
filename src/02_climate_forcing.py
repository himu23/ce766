"""
Step 2 - Catchment-averaged daily rainfall, temperature and PET (1991-2025).

Data : IMD gridded rainfall 0.25 deg (Pai et al., 2014) and IMD gridded
       max/min temperature 1 deg, downloaded with `imdlib`.
Output: data/processed/forcing_daily.csv  (date, P_mm, Tmean_C, PET_mm)

pip install imdlib xarray geopandas shapely
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import imdlib as imd
import xarray as xr
import config as C

catch = gpd.read_file(C.PROC / "catchment.geojson").to_crs("EPSG:4326")
poly = catch.geometry.iloc[0]


def download(var):
    """Read (downloading if missing) one year at a time and keep only the study box,
    so the full India grid (~50 MB/yr for rain) is never held in memory at once."""
    d = C.RAW / "imd"
    d.mkdir(exist_ok=True)
    pad = 1.0
    parts = []
    for y in range(C.HIST_START, C.HIST_END + 1):
        have = any((d / var / f"{y}.{e}").exists() for e in ("grd", "GRD"))
        fn = imd.open_data if have else imd.get_data
        ds = fn(var, y, y, fn_format="yearwise", file_dir=str(d)).get_xarray()
        parts.append(ds[var].sel(lat=slice(C.BBOX["lat_min"] - pad, C.BBOX["lat_max"] + pad),
                                 lon=slice(C.BBOX["lon_min"] - pad, C.BBOX["lon_max"] + pad)).load())
    da = xr.concat(parts, dim="time")
    return da.where(da > -900).where(da < 99)  if var != "rain" else da.where(da > -900)


def area_weights(da, res):
    """Fraction of each grid cell lying inside the catchment polygon."""
    lats = da["lat"].values
    lons = da["lon"].values
    w = np.zeros((len(lats), len(lons)))
    for i, la in enumerate(lats):
        for j, lo in enumerate(lons):
            cell = box(lo - res / 2, la - res / 2, lo + res / 2, la + res / 2)
            w[i, j] = cell.intersection(poly).area / cell.area
    return w


def catchment_mean(da, res):
    sub = da.sel(lat=slice(C.BBOX["lat_min"] - res, C.BBOX["lat_max"] + res),
                 lon=slice(C.BBOX["lon_min"] - res, C.BBOX["lon_max"] + res))
    w = area_weights(sub, res)
    if w.sum() == 0:   # catchment smaller than one cell -> nearest cell
        return sub.sel(lat=C.DAM_LAT, lon=C.DAM_LON, method="nearest").to_series(), 1
    vals = sub.values                                  # time x lat x lon
    wt = np.where(np.isnan(vals), 0, w[None])
    series = np.nansum(np.nan_to_num(vals) * wt, axis=(1, 2)) / wt.sum(axis=(1, 2))
    return pd.Series(series, index=pd.to_datetime(sub["time"].values)), int((w > 0).sum())


def extraterrestrial_radiation(doy, lat_deg):
    """FAO-56 eq. 21, MJ m-2 day-1."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(delta))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def oudin_pet(tmean, dates, lat):
    """Oudin et al. (2005) temperature-based PET, mm/day - designed for GR4J."""
    re = extraterrestrial_radiation(dates.dayofyear.values, lat)
    pet = re / 2.45 * (tmean + 5) / 100
    return np.where(tmean + 5 > 0, pet, 0.0)


if __name__ == "__main__":
    P, n_rain = catchment_mean(download("rain"), 0.25)
    tmax, n_t = catchment_mean(download("tmax"), 1.0)
    tmin, _ = catchment_mean(download("tmin"), 1.0)

    df = pd.DataFrame({"P_mm": P, "Tmax_C": tmax, "Tmin_C": tmin}).dropna()
    df["Tmean_C"] = (df.Tmax_C + df.Tmin_C) / 2
    df["PET_mm"] = oudin_pet(df.Tmean_C.values, df.index, C.DAM_LAT)
    df.index.name = "date"
    df.to_csv(C.PROC / "forcing_daily.csv")

    ann = df.P_mm.resample("YE").sum()
    mons = df.P_mm[df.index.month.isin(C.MONSOON_MONTHS)].sum() / df.P_mm.sum()
    print(f"Rainfall grid cells used: {n_rain}, temperature cells: {n_t}")
    print(f"Mean annual rainfall: {ann.mean():.0f} mm (CV {ann.std()/ann.mean():.2f})")
    print(f"Share of rainfall in Jun-Sep: {100*mons:.1f}%")
    print(f"Mean annual PET: {df.PET_mm.resample('YE').sum().mean():.0f} mm")
    print(f"Driest year: {ann.idxmin().year} ({ann.min():.0f} mm), "
          f"wettest: {ann.idxmax().year} ({ann.max():.0f} mm)")

    # Independent check: IMD catchment rainfall vs the BMC rain gauge at Tansa dam
    # (BMC reports cumulative rainfall since 1 June in its 6 AM bulletin)
    st = pd.read_csv(C.RAW / "tansa_storage.csv", parse_dates=["date"], index_col="date")
    if "bmc_cum_rain_mm" in st:
        for y in sorted(set(st.index.year)):
            g = st.bmc_cum_rain_mm[(st.index.year == y) & (st.index.month <= 10)].dropna()
            if len(g) and g.index[-1] - pd.Timestamp(f"{y}-06-01") > pd.Timedelta(days=60) \
                    and g.index[-1] <= df.index.max():
                imd_sum = df.P_mm.loc[f"{y}-06-01":g.index[-1] - pd.Timedelta(days=1)].sum()
                print(f"  {y} Jun 1-{g.index[-1]:%b %d}: BMC gauge {g.iloc[-1]:.0f} mm, "
                      f"IMD catchment {imd_sum:.0f} mm ({100*(imd_sum/g.iloc[-1]-1):+.0f}%)")
