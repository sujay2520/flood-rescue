import os
import sys
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pystac_client import Client
import planetary_computer as pc

sys.path.insert(0, ".")
from src.flood import create_baseline_mask, get_flood_summary

dest_dir = "data/western_dadu_check"
os.makedirs(dest_dir, exist_ok=True)
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

# Western Dadu / Johi / Lake Manchar corridor
bbox = [67.65, 26.65, 67.85, 26.85]

post_coll = catalog.search(
    collections=["sentinel-1-rtc"],
    bbox=bbox,
    ids=["S1A_IW_GRDH_1SDV_20220908T133610_20220908T133635_044914_055D60_rtc"]
).item_collection()

pre_coll = catalog.search(
    collections=["sentinel-1-rtc"],
    bbox=bbox,
    ids=["S1A_IW_GRDH_1SDV_20220722T133607_20220722T133632_044214_0546F0_rtc"]
).item_collection()

if not post_coll or not pre_coll:
    print("Could not find exact items.")
    sys.exit(1)

post_item = post_coll[0]
pre_item = pre_coll[0]

def stream_clip(item, out_p):
    with rasterio.open(item.assets["vv"].href) as src:
        rb = transform_bounds("EPSG:4326", src.crs, *bbox)
        win = from_bounds(*rb, transform=src.transform)
        arr = src.read(1, window=win)
        meta = src.meta.copy()
        meta.update({
            "height": arr.shape[0],
            "width": arr.shape[1],
            "transform": src.window_transform(win),
            "driver": "GTiff"
        })
        with rasterio.open(out_p, "w", **meta) as dst:
            dst.write(arr, 1)
        return arr, src.crs, src.window_transform(win)

post_p = os.path.join(dest_dir, "post_sar.tif")
pre_p = os.path.join(dest_dir, "pre_sar.tif")

if not os.path.exists(post_p):
    print("Streaming post-flood SAR (Sept 8, 2022)...")
    post_arr, crs, transform = stream_clip(post_item, post_p)
else:
    with rasterio.open(post_p) as s:
        post_arr, crs, transform = s.read(1), s.crs, s.transform

if not os.path.exists(pre_p):
    print("Streaming pre-flood SAR (July 22, 2022)...")
    pre_arr, _, _ = stream_clip(pre_item, pre_p)
else:
    with rasterio.open(pre_p) as s:
        pre_arr = s.read(1)

print(f"Western Dadu raster shape: {post_arr.shape}")
mask = create_baseline_mask(pre_arr, post_arr, is_db=False, water_db=-17.0, drop_db=2.5, min_pixels=40)
summary = get_flood_summary(mask, transform=transform, crs=crs)

area = summary["flood_area_km2"]
pct = summary["flood_percentage"]
pixels = summary["flood_pixels"]

print("-" * 60)
print(f"WESTERN DADU / JOHI CORRIDOR FLOOD RESULTS:")
print(f"  Flooded Area: {area:.2f} km²")
print(f"  Flooded Percentage of AOI: {pct:.2f}%")
print(f"  Flooded Pixels: {pixels:,}")
print("-" * 60)
