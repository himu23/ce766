"""
Project configuration for the Tansa reservoir study (CE 766 term project).

Every value marked VERIFY is an assumption that must be checked against an
official source (BMC lake-level bulletins, India-WRIS reservoir page, MCGM
water supply reports) before the results are reported.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
for p in (RAW, PROC, OUT):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- study area
DAM_LAT, DAM_LON = 19.5744, 73.2494          # lake outlet cell from step 1 (DEM + OSM lake outline)
# Bounding box used to download / clip gridded data (covers the catchment)
BBOX = dict(lat_min=19.50, lat_max=19.63, lon_min=73.23, lon_max=73.40)   # catchment bounds (step 1)

HIST_START, HIST_END = 1991, 2025

# ---------------------------------------------------------------- reservoir
S_MAX_ML = 145_080        # useful (live) storage, ML -- BMC 6 AM lake report (141,180 ML = 97.31%, 25-09-2026)
S_MIN_ML = 0.0            # dead storage excluded: BMC reports *useful* content
A_LAKE_FULL_KM2 = 19.1    # water-spread area at FSL (128.63 m THD), km^2 -- India-WRIS/Wikipedia;
                          # OSM outline is 13.6 km^2 (lower stage), used only for delineation
AREA_EXPONENT = 2 / 3     # A(S) = A_full * (S/Smax)^b, a standard approximation
                          # replace with the real elevation-area-capacity table if available
PAN_COEFF = 1.0           # lake evaporation = PAN_COEFF * PET

# ---------------------------------------------------------------- demand
D_TARGET_MLD = 455.0      # Tansa share of Mumbai supply, ML/day  -- VERIFY (MCGM)

# ---------------------------------------------------------------- env. flow
# Tennant (1976) method adapted to the monsoon calendar (no official e-flow exists)
EFLOW_FRAC_MONSOON = 0.30   # Jun-Sep, fraction of mean annual flow
EFLOW_FRAC_DRY = 0.10       # Oct-May
MONSOON_MONTHS = (6, 7, 8, 9)

# ---------------------------------------------------------------- calibration
CAL_PERIOD = ("2023-07-01", "2024-12-31")   # BMC storage reports available 2023-07 onward
VAL_PERIOD = ("2025-01-01", "2025-12-31")   # IMD gridded data ends 2025
WARMUP_DAYS = 365
