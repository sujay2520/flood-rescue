"""
scripts/download_planetary_computer.py - Stream and clip Sentinel-1 SAR imagery.

Connects to Microsoft Planetary Computer STAC API to search, sign, stream, and clip
pre- and post-flood Sentinel-1 GRD imagery directly to a target bounding box (AOI),
avoiding downloading entire ~1GB satellite granules.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


def search_sentinel1_scenes(
    bbox: List[float],
    date_range: str,
    orbit_direction: Optional[str] = None,
    polarization: str = "vv",
) -> List[dict]:
    """
    Search Planetary Computer STAC API for Sentinel-1 GRD scenes intersecting bbox.

    Args:
        bbox: [west, south, east, north] in EPSG:4326
        date_range: ISO range, e.g. "2022-08-01/2022-08-15"
        orbit_direction: Optional "ascending" or "descending"
        polarization: Target band, e.g. "vv" or "vh"

    Returns:
        List of matching signed STAC items.
    """
    try:
        from pystac_client import Client
        import planetary_computer as pc
    except ImportError:
        raise ImportError(
            "pystac-client and planetary-computer packages are required. "
            "Install with: uv pip install pystac-client planetary-computer"
        )

    print(f"[*] Querying Planetary Computer STAC for bbox={bbox}, dates={date_range}...")
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )

    query = {
        "sar:instrument_mode": {"eq": "IW"},
    }
    if orbit_direction:
        query["sat:orbit_state"] = {"eq": orbit_direction}

    search = catalog.search(
        collections=["sentinel-1-grd"],
        bbox=bbox,
        datetime=date_range,
        query=query,
    )

    items = list(search.items())
    print(f"[+] Found {len(items)} matching Sentinel-1 scenes.")
    return items


def clip_and_save_sar(
    stac_item,
    bbox: List[float],
    output_path: str,
    band: str = "vv",
) -> str:
    """
    Stream and clip a signed Sentinel-1 asset from Planetary Computer.

    Args:
        stac_item: Signed STAC item from search_sentinel1_scenes
        bbox: [west, south, east, north] in EPSG:4326
        output_path: Target path to write clipped GeoTIFF
        band: "vv" or "vh"

    Returns:
        Path to saved GeoTIFF.
    """
    if band not in stac_item.assets:
        available = list(stac_item.assets.keys())
        raise KeyError(f"Band '{band}' not found in asset. Available: {available}")

    asset_url = stac_item.assets[band].href
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[*] Streaming asset from: {stac_item.id} (date: {stac_item.datetime})...")
    with rasterio.open(asset_url) as src:
        # Transform bbox from EPSG:4326 to raster CRS if necessary
        west, south, east, north = bbox
        if src.crs and src.crs.to_string() != "EPSG:4326":
            bounds = transform_bounds("EPSG:4326", src.crs, west, south, east, north)
        else:
            bounds = (west, south, east, north)

        window = from_bounds(*bounds, transform=src.transform)
        # Ensure window indices are valid integers
        window = window.round_shape()

        data = src.read(1, window=window)
        win_transform = src.window_transform(window)

        meta = src.meta.copy()
        meta.update({
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": win_transform,
            "count": 1,
            "dtype": data.dtype,
            "compress": "lzw",
        })

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(data, 1)

    print(f"[SUCCESS] Clipped SAR image saved to: {output_path} ({data.shape[1]}x{data.shape[0]} px)")
    return output_path


def fetch_flood_pair(
    bbox: List[float],
    pre_dates: str,
    post_dates: str,
    output_dir: str = "data/downloaded",
    band: str = "vv",
) -> Tuple[str, str]:
    """
    Automated fetcher for pre- and post-flood Sentinel-1 pairs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Search Pre-flood
    pre_items = search_sentinel1_scenes(bbox, pre_dates, polarization=band)
    if not pre_items:
        raise ValueError(f"No pre-flood scenes found for date range: {pre_dates}")
    pre_item = pre_items[-1] # closest to flood event

    # 2. Search Post-flood
    post_items = search_sentinel1_scenes(bbox, post_dates, polarization=band)
    if not post_items:
        raise ValueError(f"No post-flood scenes found for date range: {post_dates}")
    post_item = post_items[0] # first scene after flood peak

    pre_path = os.path.join(output_dir, f"pre_{pre_item.id[:20]}_{band}.tif")
    post_path = os.path.join(output_dir, f"post_{post_item.id[:20]}_{band}.tif")

    clip_and_save_sar(pre_item, bbox, pre_path, band=band)
    clip_and_save_sar(post_item, bbox, post_path, band=band)

    return pre_path, post_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download & clip Sentinel-1 SAR imagery from Planetary Computer.")
    parser.add_argument("--bbox", nargs=4, type=float, default=[67.70, 26.65, 67.90, 26.85],
                        help="Bounding box: min_lon min_lat max_lon max_lat (default: Dadu, Sindh, Pakistan)")
    parser.add_argument("--pre_dates", type=str, default="2022-07-01/2022-07-20", help="Pre-flood ISO date range")
    parser.add_argument("--post_dates", type=str, default="2022-08-25/2022-09-10", help="Post-flood ISO date range")
    parser.add_argument("--band", type=str, default="vv", choices=["vv", "vh"], help="SAR Polarization band")
    parser.add_argument("--output_dir", type=str, default="data/downloaded", help="Destination folder")

    args = parser.parse_args()
    print("=" * 60)
    print("Planetary Computer Sentinel-1 SAR Clip Downloader")
    print(f"AOI BBox: {args.bbox}")
    print(f"Pre-flood: {args.pre_dates} | Post-flood: {args.post_dates}")
    print("=" * 60)

    try:
        pre_f, post_f = fetch_flood_pair(
            bbox=args.bbox,
            pre_dates=args.pre_dates,
            post_dates=args.post_dates,
            output_dir=args.output_dir,
            band=args.band,
        )
        print(f"\n[DONE] Pre-flood: {pre_f}\n[DONE] Post-flood: {post_f}")
    except Exception as e:
        print(f"[!] Error or rate-limit during Planetary Computer streaming: {e}")
        print("[*] Note: You can use the offline synthetic demo generator in src/ingest.py for 100% offline demonstration.")
