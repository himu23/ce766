"""
Step 1 - Delineate the Tansa catchment upstream of the dam.

Input : data/raw/dem.tif  Copernicus GLO-30 DEM (30 m) tile N19 E073, free from
        https://copernicus-dem-30m.s3.amazonaws.com/ (no login needed).
        SRTM 30 m from USGS EarthExplorer works the same way.
Output: data/processed/catchment.geojson + catchment_info.json

pip install pyflwdir rasterio geopandas shapely
"""
import json
import numpy as np
import rasterio
import rasterio.features
import pyflwdir
import geopandas as gpd
from shapely.geometry import shape
import config as C

with rasterio.open(C.RAW / "dem.tif") as src:
    dem = src.read(1).astype("float32")
    transform, crs, nodata = src.transform, src.crs, src.nodata
dem = np.where(dem == nodata, -9999, dem) if nodata is not None else dem

# D8 flow directions after depression filling (Wang & Liu 2006, inside pyflwdir)
flw = pyflwdir.from_dem(data=dem, nodata=-9999, transform=transform, latlon=True)
upa = flw.upstream_area(unit="km2")

# The DEM flattens the lake surface, so D8 routing across it splits arbitrarily
# and no single "dam cell" collects all the flow. Instead the catchment is every
# cell that drains into ANY lake cell (lake outline from OpenStreetMap), plus the lake.
lake_gdf = gpd.read_file(C.RAW / "tansa_lake_osm.geojson").to_crs(crs)
lake = rasterio.features.rasterize(lake_gdf.geometry, out_shape=dem.shape,
                                   transform=transform) == 1
lake_idx = np.flatnonzero(lake)
basin = ((flw.basins(idxs=lake_idx) > 0) | lake).astype(np.uint8)
# report the lake cell with the largest upstream area as the nominal outlet
row, col = np.unravel_index(lake_idx[np.argmax(upa.ravel()[lake_idx])], dem.shape)
lon, lat = rasterio.transform.xy(transform, row, col)
feats = [shape(g) for g, v in rasterio.features.shapes(basin, mask=basin == 1,
                                                       transform=transform)]
gdf = gpd.GeoDataFrame(geometry=feats, crs=crs).dissolve()
area_km2 = gdf.to_crs("EPSG:32643").area.iloc[0] / 1e6       # UTM 43N
gdf["area_km2"] = area_km2
gdf.to_file(C.PROC / "catchment.geojson", driver="GeoJSON")

zs = dem[basin == 1]
info = {"outlet_lat": lat, "outlet_lon": lon, "area_km2": area_km2,
        "elev_min_m": float(zs.min()), "elev_mean_m": float(zs.mean()),
        "elev_max_m": float(zs.max()),
        "bounds": [float(b) for b in gdf.total_bounds]}
json.dump(info, open(C.PROC / "catchment_info.json", "w"), indent=2)
print(f"Snapped outlet: {lat:.4f} N, {lon:.4f} E (upstream area {upa[row, col]:.0f} km2)")
print(f"Catchment area: {area_km2:.1f} km2, elevation {zs.min():.0f}-{zs.max():.0f} m "
      f"(mean {zs.mean():.0f})")
print("Bounds (lon/lat):", info["bounds"])
