"""
src/dem_terrain.py

Production-grade Copernicus DEM GLO-30 processing and SAR terrain geometry masking
for Track B: Mapping Flood Damage from Space (August 2026 Trishuli/Bhote Koshi River Disaster, Nepal).

Handles:
- Copernicus DEM GLO-30 streaming from Planetary Computer (`cop-dem-glo-30`)
- Offline synthetic Himalayan DEM fallback (1500m - 5000m V-shaped valley)
- Horn 3x3 slope and aspect
- Local Incidence Angle (LIA) calculation
- Radar Shadow Masking (geometric + Line-of-Sight ray-tracing)
- Radar Layover Masking (foreshortening / slope towards radar > look angle)
- Valid SAR Mask & Slope Plausibility Mask (slope <= 15 deg)
- Outputs 8 GeoTIFF products matching S1 SAR raster grid

Author: Track B Engineering Team
License: MIT
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from scipy.ndimage import convolve
import planetary_computer as pc
import pystac_client

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dem_terrain")

# Constants
COP_DEM_COLLECTION = "cop-dem-glo-30"
TARGET_CRS = "EPSG:32645"  # UTM Zone 45N for Nepal
DEFAULT_RESOLUTION = 30.0  # meters
NOMINAL_INCIDENCE_ANGLE = 39.0  # degrees
SLOPE_THRESHOLD_DEG = 15.0  # degrees for water pooling / SAR validity

# Sentinel-1 look geometries
ORBIT_GEOMETRY = {
    "ascending": {"look_azimuth": 78.0, "nominal_incidence": 39.0},
    "descending": {"look_azimuth": 258.0, "nominal_incidence": 39.0},
}

# Horn 3x3 kernels for slope/aspect
HORN_DX_KERNEL = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32) / 8.0

HORN_DY_KERNEL = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1],
], dtype=np.float32) / 8.0


# =============================================================================
# Utility Functions
# =============================================================================

def save_geotiff(
    data: np.ndarray,
    transform: rasterio.Affine,
    crs: Union[str, CRS],
    out_path: Union[str, Path],
    dtype: str = "float32",
    nodata: Optional[float] = None,
) -> Path:
    """Save 2D numpy array as GeoTIFF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 3 if "float" in dtype else 1,
        "blockxsize": 256,
        "blockysize": 256,
        "tiled": True,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)

    return out_path


# =============================================================================
# DEM Fetching & Synthetic Generation
# =============================================================================

def fetch_or_generate_dem(
    bbox: List[float],
    target_transform: rasterio.Affine,
    target_shape: Tuple[int, int],
    target_crs: Union[str, CRS] = TARGET_CRS,
    force_offline: bool = False,
) -> np.ndarray:
    """
    Fetch Copernicus DEM GLO-30 from Planetary Computer or generate realistic
    Himalayan valley DEM matching the target grid.
    """
    height, width = target_shape
    if force_offline:
        logger.info("Offline mode requested: generating synthetic Himalayan DEM")
        return generate_synthetic_dem(target_transform, target_shape)

    try:
        logger.info(f"Searching Planetary Computer for Copernicus DEM GLO-30 in {bbox}...")
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )
        search = catalog.search(
            collections=[COP_DEM_COLLECTION],
            bbox=bbox,
            limit=10,
        )
        items = list(search.items())
        if not items:
            logger.warning("No Copernicus DEM items found; falling back to synthetic DEM")
            return generate_synthetic_dem(target_transform, target_shape)

        # Reproject first matching item directly into target grid
        out_dem = np.full(target_shape, fill_value=np.nan, dtype=np.float32)
        for item in items:
            asset_href = pc.sign(item.assets["data"].href)
            with rasterio.open(asset_href) as src:
                tile_dem = np.empty(target_shape, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, 1),
                    destination=tile_dem,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                    dst_nodata=np.nan,
                )
                valid = np.isfinite(tile_dem)
                out_dem[valid] = tile_dem[valid]

        if np.all(np.isnan(out_dem)):
            logger.warning("Reprojected DEM is empty; falling back to synthetic DEM")
            return generate_synthetic_dem(target_transform, target_shape)

        # Fill any remaining NaNs
        nan_mask = np.isnan(out_dem)
        if np.any(nan_mask):
            median_val = np.nanmedian(out_dem)
            out_dem[nan_mask] = median_val

        logger.info(f"Successfully loaded Copernicus DEM (min: {np.min(out_dem):.1f}m, max: {np.max(out_dem):.1f}m)")
        return out_dem

    except Exception as e:
        logger.warning(f"Error fetching Copernicus DEM from STAC: {e}. Generating synthetic Himalayan DEM.")
        return generate_synthetic_dem(target_transform, target_shape)


def generate_synthetic_dem(
    target_transform: rasterio.Affine,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Generate a realistic Himalayan steep gorge DEM (Trishuli/Bhote Koshi style)."""
    height, width = target_shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = target_transform * (cols, rows)

    # Base elevation rises from south to north (1500m to 4500m)
    y_norm = (ys - np.min(ys)) / (np.max(ys) - np.min(ys) + 1e-5)
    x_norm = (xs - np.min(xs)) / (np.max(xs) - np.min(xs) + 1e-5)

    base = 1600.0 + 2400.0 * y_norm

    # Deep river valley cutting through center
    valley_center_x = 0.5 + 0.1 * np.sin(y_norm * np.pi * 3)
    dist_to_river = np.abs(x_norm - valley_center_x)

    # V-shaped gorge profile: steep canyon walls (35 to 55 degrees)
    gorge = 1200.0 * np.minimum(dist_to_river * 4.0, 1.0) ** 0.8

    # Mountain ridges and tributaries
    ridges = 400.0 * np.sin(x_norm * 12.0) * np.cos(y_norm * 8.0)

    elevation = base + gorge + ridges
    elevation = np.maximum(elevation, 1200.0).astype(np.float32)
    return elevation


# =============================================================================
# Terrain Derivatives (Horn 3x3)
# =============================================================================

def compute_slope_aspect(
    elevation: np.ndarray,
    transform: rasterio.Affine,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope (degrees) and aspect (degrees azimuth, 0=N, 90=E) using Horn 3x3.
    """
    px = abs(transform.a)
    py = abs(transform.e)

    dzdx = convolve(elevation, HORN_DX_KERNEL, mode="reflect") / px
    dzdy = convolve(elevation, HORN_DY_KERNEL, mode="reflect") / py

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)

    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_deg = np.degrees(aspect_rad) % 360.0

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


# =============================================================================
# Radar Physics: LIA, Shadow & Layover Masking
# =============================================================================

def compute_local_incidence_angle(
    slope: np.ndarray,
    aspect: np.ndarray,
    look_azimuth: float,
    nominal_incidence: float,
) -> np.ndarray:
    """
    Compute Local Incidence Angle (LIA) in degrees.
    LIA = arccos(cos(theta) * cos(slope) + sin(theta) * sin(slope) * cos(aspect - look_azimuth))
    """
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    look_rad = np.radians(look_azimuth)
    theta_rad = np.radians(nominal_incidence)

    cos_lia = (
        np.cos(theta_rad) * np.cos(slope_rad) +
        np.sin(theta_rad) * np.sin(slope_rad) * np.cos(aspect_rad - look_rad)
    )
    cos_lia = np.clip(cos_lia, -1.0, 1.0)
    lia_deg = np.degrees(np.arccos(cos_lia))
    return lia_deg.astype(np.float32)


def compute_radar_shadow_mask(
    elevation: np.ndarray,
    transform: rasterio.Affine,
    look_azimuth: float,
    nominal_incidence: float,
    max_search_distance: float = 4000.0,
) -> np.ndarray:
    """
    Compute radar shadow mask using geometric LIA >= 90° and ray-tracing line-of-sight check.
    True = shadowed (radar beam cannot illuminate surface).
    """
    height, width = elevation.shape
    slope, aspect = compute_slope_aspect(elevation, transform)
    lia = compute_local_incidence_angle(slope, aspect, look_azimuth, nominal_incidence)

    # Geometric shadow
    shadow = (lia >= 90.0)

    # Line of sight ray tracing along range direction
    px = abs(transform.a)
    py = abs(transform.e)
    pixel_size = max(px, py)

    look_az_rad = np.radians(look_azimuth)
    # Unit vector pointing towards satellite (opposite to look azimuth)
    sat_dx = -np.sin(look_az_rad)
    sat_dy = -np.cos(look_az_rad)

    max_steps = min(int(max_search_distance / pixel_size), 50)
    look_angle_rad = np.radians(90.0 - nominal_incidence)

    rows, cols = np.mgrid[0:height, 0:width]
    elev_flat = elevation.ravel()
    shadow_flat = shadow.ravel()

    # Step along satellite line of sight
    for step in range(1, max_steps + 1):
        step_r = (rows + int(round(step * sat_dy))).clip(0, height - 1)
        step_c = (cols + int(round(step * sat_dx))).clip(0, width - 1)

        intervening_elev = elevation[step_r, step_c]
        dist = step * pixel_size
        elev_angle = np.arctan2(intervening_elev - elevation, dist)

        los_blocked = elev_angle > look_angle_rad
        shadow |= los_blocked

    return shadow


def compute_radar_layover_mask(
    slope: np.ndarray,
    aspect: np.ndarray,
    look_azimuth: float,
    nominal_incidence: float,
) -> np.ndarray:
    """
    Compute radar layover mask.
    Layover occurs when terrain slope facing the radar exceeds the look angle:
    slope * cos(aspect - look_azimuth) > (90° - nominal_incidence)
    """
    slope_rad = np.radians(slope)
    aspect_rad = np.radians(aspect)
    look_az_rad = np.radians(look_azimuth)

    slope_towards_radar = slope_rad * np.cos(aspect_rad - look_az_rad)
    look_angle_rad = np.radians(90.0 - nominal_incidence)

    layover = slope_towards_radar > look_angle_rad
    return layover


def compute_valid_sar_mask(
    shadow: np.ndarray,
    layover: np.ndarray,
    slope: np.ndarray,
    slope_threshold: float = SLOPE_THRESHOLD_DEG,
) -> np.ndarray:
    """Valid SAR mask: True where valid for SAR analysis (not shadow, not layover, slope <= threshold)."""
    invalid = shadow | layover | (slope > slope_threshold)
    return (~invalid)


def compute_slope_plausibility_mask(
    slope: np.ndarray,
    slope_threshold: float = SLOPE_THRESHOLD_DEG,
) -> np.ndarray:
    """Slope plausibility mask: True where slope <= 15° (water cannot pond on steep mountain cliffs)."""
    return (slope <= slope_threshold)


# =============================================================================
# High-Level Pipeline Function
# =============================================================================

def prepare_dem_and_masks(
    bbox: List[float],
    reference_raster_path: Optional[Union[str, Path]] = None,
    orbit_state: str = "ascending",
    out_dir: Optional[Union[str, Path]] = None,
    force_offline: bool = False,
    slope_threshold: float = SLOPE_THRESHOLD_DEG,
) -> Dict[str, Path]:
    """
    High-level coordinator to generate DEM and all 8 terrain/SAR masks.

    Returns Dict with keys:
    - 'dem': Elevation (m)
    - 'slope': Slope (deg)
    - 'aspect': Aspect (deg)
    - 'lia': Local Incidence Angle (deg)
    - 'shadow': Radar shadow mask (uint8 0/1)
    - 'layover': Radar layover mask (uint8 0/1)
    - 'valid_sar': Valid SAR mask (uint8 0/1)
    - 'slope_plausible': Slope plausibility mask (uint8 0/1)
    """
    out_dir = Path(out_dir or "./outputs/dem_masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    if reference_raster_path and Path(reference_raster_path).exists():
        logger.info(f"Aligning DEM grid to reference raster: {reference_raster_path}")
        with rasterio.open(reference_raster_path) as ref:
            target_transform = ref.transform
            target_shape = (ref.height, ref.width)
            target_crs = ref.crs
    else:
        # Construct target transform from bbox (WGS84 -> UTM 45N)
        minx, miny, maxx, maxy = bbox
        bounds_utm = transform_bounds("EPSG:4326", TARGET_CRS, minx, miny, maxx, maxy, densify_pts=21)
        res = DEFAULT_RESOLUTION
        width = int(round((bounds_utm[2] - bounds_utm[0]) / res))
        height = int(round((bounds_utm[3] - bounds_utm[1]) / res))
        target_shape = (height, width)
        target_transform = from_bounds(bounds_utm[0], bounds_utm[1], bounds_utm[2], bounds_utm[3], width, height)
        target_crs = CRS.from_string(TARGET_CRS)

    # 1. Fetch or generate DEM
    elevation = fetch_or_generate_dem(
        bbox=bbox,
        target_transform=target_transform,
        target_shape=target_shape,
        target_crs=target_crs,
        force_offline=force_offline,
    )
    dem_path = save_geotiff(elevation, target_transform, target_crs, out_dir / "dem.tif", dtype="float32", nodata=-9999.0)

    # 2. Slope and Aspect
    slope, aspect = compute_slope_aspect(elevation, target_transform)
    slope_path = save_geotiff(slope, target_transform, target_crs, out_dir / "slope.tif", dtype="float32", nodata=-9999.0)
    aspect_path = save_geotiff(aspect, target_transform, target_crs, out_dir / "aspect.tif", dtype="float32", nodata=-9999.0)

    # 3. Radar look geometry
    geom = ORBIT_GEOMETRY.get(orbit_state.lower(), ORBIT_GEOMETRY["ascending"])
    look_azimuth = geom["look_azimuth"]
    nominal_incidence = geom["nominal_incidence"]

    # 4. Local Incidence Angle
    lia = compute_local_incidence_angle(slope, aspect, look_azimuth, nominal_incidence)
    lia_path = save_geotiff(lia, target_transform, target_crs, out_dir / "lia.tif", dtype="float32", nodata=-9999.0)

    # 5. Radar Shadow Mask
    shadow = compute_radar_shadow_mask(elevation, target_transform, look_azimuth, nominal_incidence)
    shadow_path = save_geotiff(shadow.astype(np.uint8), target_transform, target_crs, out_dir / "shadow.tif", dtype="uint8", nodata=255)

    # 6. Radar Layover Mask
    layover = compute_radar_layover_mask(slope, aspect, look_azimuth, nominal_incidence)
    layover_path = save_geotiff(layover.astype(np.uint8), target_transform, target_crs, out_dir / "layover.tif", dtype="uint8", nodata=255)

    # 7. Valid SAR Mask
    valid_sar = compute_valid_sar_mask(shadow, layover, slope, slope_threshold)
    valid_sar_path = save_geotiff(valid_sar.astype(np.uint8), target_transform, target_crs, out_dir / "valid_sar.tif", dtype="uint8", nodata=255)

    # 8. Slope Plausibility Mask (<= 15 deg)
    slope_plausible = compute_slope_plausibility_mask(slope, slope_threshold)
    slope_plausible_path = save_geotiff(slope_plausible.astype(np.uint8), target_transform, target_crs, out_dir / "slope_plausible.tif", dtype="uint8", nodata=255)

    outputs = {
        "dem": dem_path,
        "slope": slope_path,
        "aspect": aspect_path,
        "lia": lia_path,
        "shadow": shadow_path,
        "layover": layover_path,
        "valid_sar": valid_sar_path,
        "slope_plausible": slope_plausible_path,
    }
    logger.info(f"DEM & terrain masks ready in {out_dir}")
    return outputs


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="DEM & SAR Geometry Masking Pipeline")
    parser.add_argument("--bbox", nargs=4, type=float, default=[85.15, 27.85, 85.45, 28.25], help="minx miny maxx maxy in EPSG:4326")
    parser.add_argument("--orbit-state", type=str, default="ascending", choices=["ascending", "descending"])
    parser.add_argument("--out-dir", type=str, default="./outputs/dem_masks")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("DEM & SAR Geometry Pipeline - Track B: Trishuli Flood")
    print("=" * 70)
    res = prepare_dem_and_masks(
        bbox=args.bbox,
        orbit_state=args.orbit_state,
        out_dir=args.out_dir,
        force_offline=args.offline,
    )
    for k, v in res.items():
        print(f"  {k:15s}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
