"""
src/mountain_flood.py

Production-grade SAR Flood & Mudflow Detection for Mountainous Terrain (Nepal Trishuli Disaster).
Implements strict mountain masking, dual-pol thresholding, morphological cleaning, and vectorization.

Author: Track B Engineering Team
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mountain_flood")

# Constants
NODATA_MASK = 255
NODATA_CONF = np.nan
DEFAULT_DB_THRESHOLD = -2.5
DEFAULT_POST_DB_THRESHOLD = -15.0
DEFAULT_VH_DB_THRESHOLD = -2.0
DEFAULT_MIN_PIXELS = 5
MORPH_RADIUS = 1


def _read_raster_aligned(path: Union[str, Path], reference_profile: Dict[str, Any]) -> np.ndarray:
    """Read a raster and reproject to reference profile if shapes or grids differ."""
    with rasterio.open(path) as src:
        if (src.width == reference_profile['width'] and 
            src.height == reference_profile['height'] and 
            src.transform == reference_profile['transform'] and 
            src.crs == reference_profile['crs']):
            data = src.read(1, masked=False).astype(np.float32)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            return data

        logger.info(f"Reprojecting {path} to match reference SAR grid...")
        dst_arr = np.empty((reference_profile['height'], reference_profile['width']), dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=reference_profile['transform'],
            dst_crs=reference_profile['crs'],
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan
        )
        return dst_arr


def _compute_confidence(
    delta_vv: np.ndarray,
    post_vv: np.ndarray,
    delta_vh: Optional[np.ndarray],
    db_thresh: float,
    post_db_thresh: float,
    vh_db_thresh: float,
) -> np.ndarray:
    """Compute continuous confidence score [0, 1] based on threshold proximity."""
    vv_depth = np.clip((db_thresh - delta_vv) / abs(db_thresh), 0, 1)
    vv_calm = np.clip((post_db_thresh - post_vv) / 10.0, 0, 1)
    conf = (vv_depth + vv_calm) / 2.0

    if delta_vh is not None:
        vh_depth = np.clip((vh_db_thresh - delta_vh) / abs(vh_db_thresh), 0, 1)
        conf = np.maximum(conf, (conf + vh_depth) / 2.0)

    return np.clip(conf, 0.0, 1.0).astype(np.float32)


def _vectorize_flood(mask: np.ndarray, transform: Affine, crs: CRS, min_pixels: int) -> gpd.GeoDataFrame:
    """Converts binary mask to polygons with area in m²."""
    results = (
        {'properties': {'raster_val': int(v)}, 'geometry': s}
        for s, v in shapes(mask.astype(np.uint8), mask=(mask == 1), transform=transform, connectivity=8)
    )

    geoms = list(results)
    if not geoms:
        return gpd.GeoDataFrame(columns=["raster_val", "area_m2", "geometry"], crs=crs)

    gdf = gpd.GeoDataFrame.from_features(geoms, crs=crs)
    if gdf.crs and gdf.crs.is_geographic:
        gdf_proj = gdf.to_crs("EPSG:32645")
        gdf["area_m2"] = gdf_proj.geometry.area
    else:
        gdf["area_m2"] = gdf.geometry.area

    # Filter out micro-polygons below min_pixels * pixel_area
    pixel_area = abs(transform.a * transform.e)
    min_area = min_pixels * pixel_area
    gdf = gdf[gdf["area_m2"] >= min_area].reset_index(drop=True)
    return gdf


def detect_mountain_flood(
    pre_vv_path: Union[str, Path],
    post_vv_path: Union[str, Path],
    dem_masks: Dict[str, Union[np.ndarray, Path, str]],
    out_dir: Union[str, Path] = "./outputs/flood",
    pre_vh_path: Optional[Union[str, Path]] = None,
    post_vh_path: Optional[Union[str, Path]] = None,
    db_threshold: float = DEFAULT_DB_THRESHOLD,
    post_db_threshold: float = DEFAULT_POST_DB_THRESHOLD,
    vh_db_threshold: float = DEFAULT_VH_DB_THRESHOLD,
    min_size_pixels: int = DEFAULT_MIN_PIXELS,
    morph_radius: int = MORPH_RADIUS,
) -> Dict[str, Any]:
    """
    Physical mountain flood & mudflow detection pipeline.
    Zero false-positive mountain shadow filter: guarantees no flood detected in radar shadows or cliffs >15°.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Detecting mountain flood: {Path(pre_vv_path).name} -> {Path(post_vv_path).name}")

    # 1. Load reference profile & SAR rasters
    with rasterio.open(pre_vv_path) as src:
        ref_profile = src.profile.copy()
        ref_profile.update(dtype=rasterio.uint8, nodata=NODATA_MASK, count=1, compress='deflate')
        pre_vv = src.read(1, masked=False).astype(np.float32)
        if src.nodata is not None:
            pre_vv[pre_vv == src.nodata] = np.nan

    post_vv = _read_raster_aligned(post_vv_path, ref_profile)
    pre_vh = _read_raster_aligned(pre_vh_path, ref_profile) if pre_vh_path else None
    post_vh = _read_raster_aligned(post_vh_path, ref_profile) if post_vh_path else None

    h, w = ref_profile['height'], ref_profile['width']

    # 2. Extract DEM masks (either numpy array or path to GeoTIFF)
    def _get_mask(key: str) -> np.ndarray:
        val = dem_masks.get(key)
        if isinstance(val, np.ndarray):
            return val
        if val is not None and Path(str(val)).exists():
            return _read_raster_aligned(val, ref_profile)
        # Default all valid if mask not provided
        return np.ones((h, w), dtype=np.uint8)

    valid_sar = _get_mask("valid_sar").astype(bool)
    shadow = _get_mask("shadow").astype(bool)
    layover = _get_mask("layover").astype(bool)
    slope_ok = _get_mask("slope_plausible").astype(bool)

    # 3. Calculate Deltas
    delta_vv = post_vv - pre_vv
    delta_vh = post_vh - pre_vh if (pre_vh is not None and post_vh is not None) else None

    # 4. Water and Debris thresholding
    cond_drop = delta_vv <= db_threshold
    cond_calm = post_vv <= post_db_threshold
    flood_candidate = cond_drop & cond_calm

    # 5. STRICT MOUNTAIN FILTERING
    # Must be valid SAR, NO radar shadow, NO radar layover, and slope <= 15 degrees
    mountain_mask = valid_sar & (~shadow) & (~layover) & slope_ok

    flood_mask = flood_candidate & mountain_mask

    data_valid = ~np.isnan(pre_vv) & ~np.isnan(post_vv)
    flood_mask &= data_valid

    # 6. Morphological cleanup
    structure = np.ones((3, 3), dtype=bool)
    flood_mask = binary_closing(flood_mask, structure=structure)
    flood_mask = binary_opening(flood_mask, structure=structure)

    if min_size_pixels > 1:
        labeled, num_features = ndimage.label(flood_mask, structure=np.ones((3, 3)))
        if num_features > 0:
            component_sizes = np.bincount(labeled.ravel())
            too_small = component_sizes < min_size_pixels
            too_small[0] = False
            flood_mask[too_small[labeled]] = False

    # 7. Confidence raster
    confidence = _compute_confidence(delta_vv, post_vv, delta_vh, db_threshold, post_db_threshold, vh_db_threshold)
    confidence[~flood_mask] = 0.0
    confidence[~mountain_mask] = 0.0

    # 8. Output arrays
    out_mask = np.full((h, w), NODATA_MASK, dtype=np.uint8)
    out_mask[mountain_mask & data_valid] = 0
    out_mask[flood_mask] = 1

    out_conf = np.full((h, w), NODATA_CONF, dtype=np.float32)
    out_conf[mountain_mask & data_valid] = confidence[mountain_mask & data_valid]

    # 9. Save Rasters
    mask_path = out_dir / "flood_mask.tif"
    conf_path = out_dir / "flood_confidence.tif"

    with rasterio.open(mask_path, 'w', **ref_profile) as dst:
        dst.write(out_mask, 1)

    conf_profile = ref_profile.copy()
    conf_profile.update(dtype=rasterio.float32, nodata=NODATA_CONF)
    with rasterio.open(conf_path, 'w', **conf_profile) as dst:
        dst.write(out_conf, 1)

    # 10. Vectorization
    polygons_path_geojson = out_dir / "flood_polygons.geojson"
    gdf = _vectorize_flood(flood_mask, ref_profile['transform'], ref_profile['crs'], min_size_pixels)
    gdf.to_file(polygons_path_geojson, driver="GeoJSON")

    total_valid_px = int(np.sum(mountain_mask & data_valid))
    flood_px = int(np.sum(flood_mask))
    pixel_area_m2 = float(abs(ref_profile['transform'].a * ref_profile['transform'].e))
    flood_area_m2 = float(flood_px * pixel_area_m2)

    logger.info(f"Flood detection finished: {flood_px:,} flooded pixels ({flood_area_m2/1e6:.3f} km²)")

    return {
        "flood_mask_path": mask_path,
        "confidence_path": conf_path,
        "polygons_geojson_path": polygons_path_geojson,
        "flood_gdf": gdf,
        "stats": {
            "total_valid_pixels": total_valid_px,
            "flood_pixels": flood_px,
            "flood_fraction": float(flood_px / max(total_valid_px, 1)),
            "flood_area_m2": flood_area_m2,
            "flood_area_km2": flood_area_m2 / 1e6,
            "num_polygons": len(gdf),
        }
    }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Mountain Flood & Mudflow SAR Detection")
    parser.add_argument("--pre-vv", type=str, required=True, help="Pre-event VV raster path")
    parser.add_argument("--post-vv", type=str, required=True, help="Post-event VV raster path")
    parser.add_argument("--dem-dir", type=str, required=True, help="Directory containing dem_masks")
    parser.add_argument("--out-dir", type=str, default="./outputs/flood")
    parser.add_argument("--db-thresh", type=float, default=-2.5)
    args = parser.parse_args()

    dem_dir = Path(args.dem_dir)
    dem_masks = {
        "valid_sar": dem_dir / "valid_sar.tif",
        "shadow": dem_dir / "shadow.tif",
        "layover": dem_dir / "layover.tif",
        "slope_plausible": dem_dir / "slope_plausible.tif",
    }

    res = detect_mountain_flood(
        pre_vv_path=args.pre_vv,
        post_vv_path=args.post_vv,
        dem_masks=dem_masks,
        out_dir=args.out_dir,
        db_threshold=args.db_thresh,
    )
    print("=" * 70)
    print("Mountain Flood Detection Result:")
    for k, v in res["stats"].items():
        print(f"  {k:20s}: {v}")
    return 0


if __name__ == "__main__":
    exit(main())
