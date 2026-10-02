"""
flood.py - SAR flood detection, U-Net inference, geospatial polygonization,
and area analysis utilities for the Flood Rescue pipeline.

This module provides production-grade functions for:
  1. SAR change detection baseline flood masking with strict parameter validation.
  2. Sliding-window U-Net deep learning inference with smooth Hann / Gaussian
     window blending to eliminate tile boundary artifacts.
  3. Exporting flood masks and rasters to georeferenced GeoTIFFs (LZW compression).
  4. Vectorizing flood masks into GeoJSON-compatible Shapely MultiPolygons or
     GeoPandas GeoDataFrames with geodesic area filtering.
  5. Calculating flood surface area (km²) and AOI flood percentage with CRS-aware
     metric projections.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
from pyproj import CRS, Geod
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, shape
import shapely

try:
    from src.core_logic import baseline_flood_mask, to_db
except ImportError:
    from .core_logic import baseline_flood_mask, to_db

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# 1. Baseline Flood Mask with Strict Validation
# ----------------------------------------------------------------------------

def create_baseline_mask(
    pre: Any,
    post: Any,
    is_db: bool = False,
    water_db: float = -18.0,
    drop_db: float = 3.0,
    speckle_size: int = 3,
    min_pixels: int = 50,
) -> np.ndarray:
    """
    Creates a new-flood boolean mask from a pre/post Sentinel-1 SAR image pair
    with parameter validation.

    A pixel is identified as newly flooded if:
      - It is dark in the post-flood image (open water backscatter < water_db).
      - It decreased in backscatter by at least drop_db compared to pre-flood.
      - Permanent water (rivers/lakes) and shadow are excluded automatically
        because they are dark in both images (change ~ 0).

    Args:
        pre: Pre-flood SAR raster array (2D or single-band 3D).
        post: Post-flood SAR raster array (2D or single-band 3D).
        is_db: True if inputs are already in decibel (dB) scale.
               False if linear amplitude or power (will be converted via to_db).
        water_db: Backscatter threshold in dB for open water (typically -18.0 dB).
        drop_db: Minimum required backscatter drop in dB (post - pre < -drop_db).
        speckle_size: Median filter window size (odd integer >= 1, e.g. 3).
        min_pixels: Minimum connected component size in pixels to retain.

    Returns:
        2D boolean numpy array where True represents detected flood pixels.

    Raises:
        TypeError: If parameter types are invalid.
        ValueError: If array dimensions, shapes, or parameter values are out of bounds.
    """
    if pre is None or post is None:
        raise ValueError("Inputs 'pre' and 'post' cannot be None.")

    # Convert to numpy arrays
    pre_arr = np.asarray(pre)
    post_arr = np.asarray(post)

    # Handle singleton channel dimensions (e.g. (1, H, W) or (H, W, 1))
    if pre_arr.ndim == 3 and (pre_arr.shape[0] == 1 or pre_arr.shape[-1] == 1):
        pre_arr = np.squeeze(pre_arr)
    if post_arr.ndim == 3 and (post_arr.shape[0] == 1 or post_arr.shape[-1] == 1):
        post_arr = np.squeeze(post_arr)

    if pre_arr.ndim != 2 or post_arr.ndim != 2:
        raise ValueError(
            f"'pre' and 'post' must be 2D arrays, got pre.shape={pre_arr.shape} "
            f"and post.shape={post_arr.shape}."
        )

    if pre_arr.shape != post_arr.shape:
        raise ValueError(
            f"Shape mismatch: 'pre' shape {pre_arr.shape} != 'post' shape {post_arr.shape}."
        )

    h, w = pre_arr.shape
    if h == 0 or w == 0:
        raise ValueError("Input raster arrays cannot be empty.")

    # Validate is_db
    if not isinstance(is_db, (bool, np.bool_)):
        raise TypeError(f"is_db must be a boolean, got {type(is_db).__name__}.")

    # Validate water_db
    if not isinstance(water_db, (int, float, np.number)) or not np.isfinite(water_db):
        raise ValueError(f"water_db must be a finite numeric value, got {water_db}.")
    if water_db > 0.0:
        logger.warning(
            "water_db is positive (%.2f dB). Open water SAR backscatter is typically "
            "substantially negative (e.g. -18 dB to -22 dB).",
            water_db,
        )

    # Validate drop_db
    if not isinstance(drop_db, (int, float, np.number)) or not np.isfinite(drop_db):
        raise ValueError(f"drop_db must be a finite numeric value, got {drop_db}.")
    if drop_db <= 0.0:
        raise ValueError(f"drop_db must be positive (> 0.0), got {drop_db}.")

    # Validate speckle_size
    if not isinstance(speckle_size, (int, np.integer)):
        raise TypeError(f"speckle_size must be an integer, got {type(speckle_size).__name__}.")
    if speckle_size < 1 or speckle_size % 2 == 0:
        raise ValueError(
            f"speckle_size must be an odd positive integer (e.g. 1, 3, 5), got {speckle_size}."
        )

    # Validate min_pixels
    if not isinstance(min_pixels, (int, np.integer)):
        raise TypeError(f"min_pixels must be an integer, got {type(min_pixels).__name__}.")
    if min_pixels < 0:
        raise ValueError(f"min_pixels must be non-negative, got {min_pixels}.")

    # Check if user mistakenly passed is_db=True on linear data (where values are non-negative)
    if is_db:
        valid_post = post_arr[np.isfinite(post_arr)]
        if len(valid_post) > 0 and np.nanmin(valid_post) >= 0.0 and np.nanmedian(valid_post) > 0.0 and np.nanmax(valid_post) < 100.0:
            logger.warning(
                "is_db=True was specified, but input array values are non-negative and appear to be linear backscatter (median=%.4f). Converting to dB.",
                float(np.nanmedian(valid_post)),
            )
            is_db = False

    # Call core baseline logic
    mask = baseline_flood_mask(
        pre=pre_arr,
        post=post_arr,
        is_db=bool(is_db),
        water_db=float(water_db),
        drop_db=float(drop_db),
        speckle_size=int(speckle_size),
        min_pixels=int(min_pixels),
    )

    return np.asarray(mask, dtype=bool)


# ----------------------------------------------------------------------------
# 2. GeoTIFF Export with LZW Compression
# ----------------------------------------------------------------------------

def save_geotiff(
    mask_or_array: np.ndarray,
    transform: Union[Affine, Tuple[float, ...], List[float]],
    crs: Any,
    output_path: Union[str, Path],
    dtype: str = "uint8",
    nodata: Optional[Union[int, float]] = None,
    compress: str = "lzw",
) -> Path:
    """
    Saves a 2D or 3D raster array / flood mask as a georeferenced GeoTIFF.

    Sets CRS, affine transform, nodata values, and applies LZW compression
    with cloud-optimized tiling.

    Args:
        mask_or_array: 2D (H, W) or 3D (C, H, W) numpy array.
                       If boolean, True maps to 1, False to 0.
        transform: Affine transform object or 6-element affine sequence.
        crs: Coordinate reference system (e.g. 'EPSG:4326', pyproj.CRS, or rasterio CRS).
        output_path: Destination file path for the GeoTIFF.
        dtype: Raster output data type ('uint8', 'float32', 'int16', etc.).
        nodata: Nodata pixel value. Defaults to 255 for uint8 and np.nan for float32.
        compress: Compression algorithm ('lzw', 'deflate', etc.). Defaults to 'lzw'.

    Returns:
        Path object pointing to the written GeoTIFF.
    """
    arr = np.asarray(mask_or_array)

    # Squeeze singleton 3D arrays to 2D
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    if arr.ndim not in (2, 3):
        raise ValueError(f"mask_or_array must be 2D (H, W) or 3D (C, H, W), got shape {arr.shape}.")

    # Format boolean masks to specified dtype
    if arr.dtype == bool:
        arr = arr.astype(dtype)

    # Standardize transform
    if not isinstance(transform, Affine):
        if len(transform) < 6:
            raise ValueError(f"transform must have at least 6 coefficients, got {len(transform)}.")
        transform = Affine(*transform[:6])

    # Standardize CRS
    if isinstance(crs, RioCRS):
        rio_crs = crs
    elif isinstance(crs, CRS):
        rio_crs = RioCRS.from_user_input(crs.to_string())
    elif isinstance(crs, str):
        rio_crs = RioCRS.from_string(crs)
    else:
        rio_crs = RioCRS.from_user_input(crs)

    # Set default nodata value if not provided
    if nodata is None:
        if dtype in ("uint8", "int8"):
            nodata = 255
        elif "float" in dtype:
            nodata = np.nan
        else:
            nodata = 0

    count = 1 if arr.ndim == 2 else arr.shape[0]
    height = arr.shape[-2]
    width = arr.shape[-1]

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": dtype,
        "crs": rio_crs,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
        "tiled": True,
    }

    with rasterio.open(out_file, "w", **profile) as dst:
        if arr.ndim == 2:
            dst.write(arr.astype(dtype), 1)
        else:
            dst.write(arr.astype(dtype))

    logger.info("Successfully wrote GeoTIFF to %s (size: %dx%d)", out_file, width, height)
    return out_file


# ----------------------------------------------------------------------------
# 3. Vectorization: Mask to Polygons / GeoDataFrame
# ----------------------------------------------------------------------------

def mask_to_polygons(
    mask: np.ndarray,
    transform: Union[Affine, Tuple[float, ...], List[float]],
    crs: Any = "EPSG:4326",
    min_area_m2: float = 500.0,
    as_multipolygon: bool = False,
) -> Union[gpd.GeoDataFrame, MultiPolygon]:
    """
    Converts a binary flood mask into GeoJSON-compatible vector polygons
    (Shapely MultiPolygon or GeoPandas GeoDataFrame), filtering out noise specks
    smaller than min_area_m2.

    Area filtering is CRS-aware: if CRS is geographic (degrees / EPSG:4326),
    geodesic surface area on the WGS84 ellipsoid is used. If CRS is projected,
    metric Cartesian area is used.

    Args:
        mask: 2D boolean or integer numpy array (non-zero indicates flood).
        transform: Affine transform object or 6-element affine sequence.
        crs: Coordinate reference system of the raster (default 'EPSG:4326').
        min_area_m2: Minimum polygon area in square meters. Polygons smaller
                     than this threshold are filtered out as speckle noise.
        as_multipolygon: If True, returns a single unified Shapely MultiPolygon.
                         If False, returns a GeoPandas GeoDataFrame.

    Returns:
        GeoPandas GeoDataFrame with columns ['flood_id', 'area_m2', 'geometry'],
        or a Shapely MultiPolygon if as_multipolygon is True.
    """
    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr.squeeze(0)
    if mask_arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask_arr.shape}.")

    if min_area_m2 < 0:
        raise ValueError(f"min_area_m2 must be non-negative, got {min_area_m2}.")

    # Standardize transform
    if not isinstance(transform, Affine):
        transform = Affine(*transform[:6])

    # Standardize CRS
    pyproj_crs = CRS.from_user_input(crs)
    is_geo = pyproj_crs.is_geographic
    geod = Geod(ellps="WGS84") if is_geo else None

    # Binary uint8 mask for rasterio.features.shapes
    bin_mask = (mask_arr > 0).astype(np.uint8)

    empty_gdf = gpd.GeoDataFrame(
        columns=["flood_id", "area_m2", "geometry"],
        geometry="geometry",
        crs=crs,
    )

    if np.count_nonzero(bin_mask) == 0:
        return MultiPolygon([]) if as_multipolygon else empty_gdf

    # Extract polygon geometries for flooded pixels (value == 1)
    shapes_iter = shapes(bin_mask, mask=(bin_mask == 1), transform=transform)

    kept_polys: List[Polygon] = []
    kept_areas: List[float] = []

    for geom_dict, _ in shapes_iter:
        poly_geom = shape(geom_dict)
        if poly_geom.is_empty:
            continue

        # Fix self-intersections or topological invalidities
        if not poly_geom.is_valid:
            poly_geom = shapely.make_valid(poly_geom)

        # Decompose MultiPolygons into individual component polygons
        parts = poly_geom.geoms if isinstance(poly_geom, MultiPolygon) else [poly_geom]

        for p in parts:
            if p.is_empty or not isinstance(p, Polygon):
                continue

            # Compute ground area in square meters
            if geod is not None:
                area_m2, _ = geod.geometry_area_perimeter(p)
                area_m2 = abs(area_m2)
            else:
                area_m2 = abs(p.area)

            if area_m2 >= min_area_m2:
                kept_polys.append(p)
                kept_areas.append(area_m2)

    if as_multipolygon:
        if not kept_polys:
            return MultiPolygon([])
        return MultiPolygon(kept_polys)

    if not kept_polys:
        return empty_gdf

    gdf = gpd.GeoDataFrame(
        {
            "flood_id": list(range(1, len(kept_polys) + 1)),
            "area_m2": [round(a, 2) for a in kept_areas],
            "geometry": kept_polys,
        },
        crs=crs,
    )
    return gdf


# ----------------------------------------------------------------------------
# 4. U-Net Sliding-Window Inference with Smooth Window Blending
# ----------------------------------------------------------------------------

def _build_window(size: int, window_type: str = "hann") -> np.ndarray:
    """
    Constructs a 2D weighting window (Hann or Gaussian) for smooth tile blending.
    Window values taper towards the edges so overlapping tile seams blend invisibly.
    """
    if window_type.lower() == "hann":
        w1d = np.hanning(size).astype(np.float32)
        w2d = np.outer(w1d, w1d)
    elif window_type.lower() == "gaussian":
        coords = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        xx, yy = np.meshgrid(coords, coords)
        sigma = 0.5
        w2d = np.exp(-(xx**2 + yy**2) / (2.0 * (sigma**2))).astype(np.float32)
    else:
        raise ValueError(f"Unsupported window_type '{window_type}'. Choose 'hann' or 'gaussian'.")

    # Clip lower bound to prevent numerical underflow / divide-by-zero
    return np.maximum(w2d, 1e-4)


def _get_tile_coordinates(length: int, tile_size: int, stride: int) -> List[int]:
    """
    Generates 1D starting indices for sliding-window tiles, guaranteeing
    that the final tile touches the boundary edge.
    """
    if length <= tile_size:
        return [0]
    coords = list(range(0, length - tile_size + 1, stride))
    last_coord = length - tile_size
    if coords[-1] != last_coord:
        coords.append(last_coord)
    return coords


def _init_resnet18_unet(
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> Tuple[Any, Any]:
    """
    Initializes a ResNet18 U-Net in eval mode and loads weights if available.
    """
    try:
        import torch
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise ImportError(
            "PyTorch and segmentation_models_pytorch are required for unet_inference. "
            "Please ensure torch and segmentation-models-pytorch are installed."
        ) from exc

    torch_device = torch.device(device if (torch.cuda.is_available() or device == "cpu") else "cpu")

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1,
    )

    if model_path is not None:
        p = Path(model_path)
        if p.is_file():
            logger.info("Loading U-Net weights from %s", p)
            state = torch.load(p, map_location=torch_device)
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]
            cleaned_state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(cleaned_state, strict=False)
        else:
            logger.warning(
                "model_path '%s' was provided but does not exist. "
                "Initializing standard ResNet18 U-Net in eval mode.",
                p,
            )
    else:
        logger.info("No model_path specified; initializing standard ResNet18 U-Net in eval mode.")

    model.to(torch_device)
    model.eval()
    return model, torch_device


def unet_inference(
    post_sar: np.ndarray,
    model_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
    tile_size: int = 256,
    stride: int = 128,
    threshold: float = 0.5,
    window_type: str = "hann",
    batch_size: int = 8,
    is_db: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs sliding-window / tiled U-Net inference on post-flood SAR imagery
    using smooth 2D window blending (Hann or Gaussian) to eliminate tile boundary seams.

    If model_path is None or the weights file does not exist, a ResNet18 U-Net
    is initialized in eval mode as a baseline.

    Args:
        post_sar: Post-flood SAR raster array (2D HxW or 1xHxW).
        model_path: Optional path to trained PyTorch model weights (.pt / .pth).
        device: Execution device ('cpu' or 'cuda').
        tile_size: Tiling spatial dimension (must be divisible by 32, default 256).
        stride: Step size between sliding tiles (default 128 for 50% overlap).
        threshold: Classification probability threshold for binary mask (default 0.5).
        window_type: Blending window filter ('hann' or 'gaussian').
        batch_size: Number of tiles to infer per forward pass (default 8).
        is_db: If True, input is treated as decibels (dB). If False, converted via to_db.
               If None, auto-detected from raster statistics.

    Returns:
        Tuple of:
          - prob_map: 2D float32 numpy array with predicted flood probabilities [0.0, 1.0].
          - binary_mask: 2D boolean numpy array where True indicates predicted flood.
    """
    import torch

    sar_arr = np.asarray(post_sar, dtype=np.float32)
    if sar_arr.ndim == 3 and sar_arr.shape[0] == 1:
        sar_arr = sar_arr.squeeze(0)
    if sar_arr.ndim != 2:
        raise ValueError(f"post_sar must be a 2D array, got shape {sar_arr.shape}.")

    h_orig, w_orig = sar_arr.shape
    if h_orig == 0 or w_orig == 0:
        raise ValueError("post_sar array cannot be empty.")

    if tile_size <= 0 or tile_size % 32 != 0:
        raise ValueError(f"tile_size must be a positive integer divisible by 32, got {tile_size}.")
    if stride <= 0 or stride > tile_size:
        raise ValueError(f"stride must be between 1 and tile_size ({tile_size}), got {stride}.")

    valid_mask = np.isfinite(sar_arr)

    # Auto-detect whether SAR is in dB scale
    if is_db is None:
        finite_vals = sar_arr[valid_mask]
        is_db = bool(len(finite_vals) > 0 and np.nanmedian(finite_vals) < 0.0)

    if not is_db:
        sar_db = to_db(sar_arr)
    else:
        sar_db = np.copy(sar_arr)

    # Fill NaNs/Infs with -30.0 dB floor (dark water)
    sar_db = np.where(valid_mask, sar_db, -30.0)

    # Standardize SAR dB: map [-30.0 dB, 0.0 dB] to [0.0, 1.0]
    sar_norm = np.clip((sar_db - (-30.0)) / 30.0, 0.0, 1.0).astype(np.float32)

    # Pad if dimensions are smaller than tile_size
    pad_h = max(0, tile_size - h_orig)
    pad_w = max(0, tile_size - w_orig)
    if pad_h > 0 or pad_w > 0:
        padded_sar = np.pad(sar_norm, ((0, pad_h), (0, pad_w)), mode="reflect")
    else:
        padded_sar = sar_norm

    h_pad, w_pad = padded_sar.shape

    # Initialize model
    model, torch_device = _init_resnet18_unet(model_path=model_path, device=device)

    # Prepare window and grid
    window = _build_window(tile_size, window_type=window_type)
    ys = _get_tile_coordinates(h_pad, tile_size, stride)
    xs = _get_tile_coordinates(w_pad, tile_size, stride)

    prob_accum = np.zeros((h_pad, w_pad), dtype=np.float32)
    weight_accum = np.zeros((h_pad, w_pad), dtype=np.float32)

    # Collect all tile coordinates
    tile_coords = [(y, x) for y in ys for x in xs]

    # Process tiles in batches
    for i in range(0, len(tile_coords), batch_size):
        batch_slice = tile_coords[i : i + batch_size]
        patches = [
            padded_sar[y : y + tile_size, x : x + tile_size]
            for (y, x) in batch_slice
        ]

        batch_np = np.stack(patches, axis=0)[:, np.newaxis, :, :]  # (B, 1, H, W)
        batch_tensor = torch.from_numpy(batch_np).to(torch_device)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        for (y, x), prob in zip(batch_slice, probs):
            prob_accum[y : y + tile_size, x : x + tile_size] += prob * window
            weight_accum[y : y + tile_size, x : x + tile_size] += window

    # Normalize by accumulated window weights and crop back to original shape
    blended_prob = (prob_accum / np.maximum(weight_accum, 1e-8))[:h_orig, :w_orig]

    # Zero out invalid/nodata pixels
    blended_prob[~valid_mask] = 0.0

    binary_mask = (blended_prob >= threshold) & valid_mask

    return blended_prob, binary_mask


# ----------------------------------------------------------------------------
# 5. Flood Surface Area and Percentage Calculation Helpers
# ----------------------------------------------------------------------------

def calculate_pixel_area_m2(
    transform: Union[Affine, Tuple[float, ...], List[float]],
    crs: Any = "EPSG:4326",
    center_lat: Optional[float] = None,
    height_pixels: Optional[int] = None,
) -> float:
    """
    Calculates the real ground surface area of a single raster pixel in square meters.

    If CRS is geographic (degrees), computes metric ground area at the specified
    or raster-derived center latitude using the WGS84 ellipsoidal model.
    If CRS is projected (e.g. UTM), computes metric Cartesian area.

    Args:
        transform: Affine transform object or 6-element tuple.
        crs: Coordinate reference system (default 'EPSG:4326').
        center_lat: Optional latitude in degrees for geographic CRS distortion.
        height_pixels: Total raster height in pixels (used to find center lat if None).

    Returns:
        Pixel area in square meters (float).
    """
    if not isinstance(transform, Affine):
        transform = Affine(*transform[:6])

    pyproj_crs = CRS.from_user_input(crs)

    if pyproj_crs.is_geographic:
        if center_lat is None:
            if height_pixels is not None:
                center_lat = transform.f + (height_pixels / 2.0) * transform.e
            else:
                center_lat = transform.f

        lat_rad = math.radians(center_lat)
        # WGS84 meters per degree formula
        m_per_deg_lat = 111132.92 - 559.82 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
        m_per_deg_lon = 111412.84 * math.cos(lat_rad) - 93.5 * math.cos(3 * lat_rad)

        pixel_w_m = abs(transform.a) * m_per_deg_lon
        pixel_h_m = abs(transform.e) * m_per_deg_lat
        return float(pixel_w_m * pixel_h_m)

    # Projected CRS: units are typically meters
    return float(abs(transform.a * transform.e))


def calculate_flood_area_km2(
    mask: np.ndarray,
    transform: Optional[Union[Affine, Tuple[float, ...], List[float]]] = None,
    crs: Any = "EPSG:4326",
    pixel_size_m: Optional[float] = None,
) -> float:
    """
    Calculates total flood surface area in square kilometers (km²).

    Args:
        mask: 2D boolean or integer numpy array (non-zero indicates flood).
        transform: Affine transform for pixel dimensions (optional if pixel_size_m is set).
        crs: Coordinate reference system (default 'EPSG:4326').
        pixel_size_m: Ground pixel dimension in meters (e.g. 10.0 for 10m Sentinel-1).
                      If provided, overrides transform calculation.

    Returns:
        Flood surface area in square kilometers (km²), rounded to 4 decimals.
    """
    mask_arr = np.asarray(mask)
    flood_pixels = int(np.count_nonzero(mask_arr > 0))

    if flood_pixels == 0:
        return 0.0

    if pixel_size_m is not None:
        pixel_area_m2 = float(pixel_size_m * pixel_size_m)
    elif transform is not None:
        pixel_area_m2 = calculate_pixel_area_m2(
            transform=transform,
            crs=crs,
            height_pixels=mask_arr.shape[-2],
        )
    else:
        # Default Sentinel-1 10m x 10m ground resolution
        pixel_area_m2 = 100.0

    total_m2 = flood_pixels * pixel_area_m2
    return round(float(total_m2 / 1_000_000.0), 4)


def calculate_flood_percentage(
    mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    nodata_val: Optional[Union[int, float]] = 255,
) -> float:
    """
    Calculates the percentage of the Area of Interest (AOI) covered by floodwater.

    Args:
        mask: 2D boolean or integer numpy array (non-zero indicates flood).
        valid_mask: Optional boolean array specifying valid AOI pixels.
                    If None, finite pixels not equal to nodata_val are counted.
        nodata_val: Pixel value indicating nodata/off-scene (default 255).

    Returns:
        Flood coverage percentage (0.0 to 100.0), rounded to 2 decimals.
    """
    mask_arr = np.asarray(mask)

    if valid_mask is None:
        if nodata_val is not None and not issubclass(mask_arr.dtype.type, np.bool_):
            valid_mask = np.isfinite(mask_arr) & (mask_arr != nodata_val)
        else:
            valid_mask = np.isfinite(mask_arr)

    total_valid = int(np.count_nonzero(valid_mask))
    if total_valid == 0:
        return 0.0

    flood_count = int(np.count_nonzero((mask_arr > 0) & valid_mask))
    pct = (flood_count / total_valid) * 100.0
    return round(float(pct), 2)


def get_flood_summary(
    mask: np.ndarray,
    transform: Optional[Union[Affine, Tuple[float, ...], List[float]]] = None,
    crs: Any = "EPSG:4326",
    valid_mask: Optional[np.ndarray] = None,
    nodata_val: Optional[Union[int, float]] = 255,
    pixel_size_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Generates a statistical summary of the flood extent over the AOI.

    Returns:
        Dictionary containing:
          - flood_pixels: Number of flooded pixels.
          - total_valid_pixels: Total non-nodata pixels in the AOI.
          - total_pixels: Total raster pixel count.
          - flood_area_km2: Flooded area in km².
          - flooded_area_km2: Alias for flood_area_km2.
          - aoi_area_km2: Total valid AOI area in km².
          - total_area_km2: Alias for aoi_area_km2.
          - flood_percentage: Percentage of AOI flooded.
          - flood_pct: Alias for flood_percentage.
          - pixel_area_m2: Ground area of one pixel in m².
    """
    mask_arr = np.asarray(mask)
    h, w = mask_arr.shape[-2], mask_arr.shape[-1]
    total_pixels = h * w

    if valid_mask is None:
        if nodata_val is not None and not issubclass(mask_arr.dtype.type, np.bool_):
            valid_mask = np.isfinite(mask_arr) & (mask_arr != nodata_val)
        else:
            valid_mask = np.isfinite(mask_arr)

    total_valid = int(np.count_nonzero(valid_mask))
    flood_pixels = int(np.count_nonzero((mask_arr > 0) & valid_mask))

    if pixel_size_m is not None:
        pixel_area_m2 = float(pixel_size_m * pixel_size_m)
    elif transform is not None:
        pixel_area_m2 = calculate_pixel_area_m2(transform, crs=crs, height_pixels=h)
    else:
        pixel_area_m2 = 100.0

    flood_area_km2 = round((flood_pixels * pixel_area_m2) / 1_000_000.0, 4)
    aoi_area_km2 = round((total_valid * pixel_area_m2) / 1_000_000.0, 4)
    pct = round((flood_pixels / total_valid * 100.0) if total_valid > 0 else 0.0, 2)

    return {
        "flood_pixels": flood_pixels,
        "total_valid_pixels": total_valid,
        "total_pixels": total_pixels,
        "flood_area_km2": flood_area_km2,
        "flooded_area_km2": flood_area_km2,
        "aoi_area_km2": aoi_area_km2,
        "total_area_km2": aoi_area_km2,
        "flood_percentage": pct,
        "flood_pct": pct,
        "pixel_area_m2": round(pixel_area_m2, 2),
    }


def calculate_flood_stats(
    mask: np.ndarray,
    transform: Optional[Union[Affine, Tuple[float, ...], List[float]]] = None,
    crs: Any = "EPSG:4326",
    valid_mask: Optional[np.ndarray] = None,
    nodata_val: Optional[Union[int, float]] = 255,
    pixel_size_m: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculates flood metrics and statistical summary across the AOI.
    Alias for get_flood_summary.
    """
    return get_flood_summary(
        mask=mask,
        transform=transform,
        crs=crs,
        valid_mask=valid_mask,
        nodata_val=nodata_val,
        pixel_size_m=pixel_size_m,
    )
