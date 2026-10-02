"""
tests/test_flood.py - Comprehensive Unit Tests for src/flood.py.

Covers:
  1. create_baseline_mask with detailed parameter validation & edge cases.
  2. save_geotiff with proper CRS, transform, nodata, and LZW compression.
  3. mask_to_polygons with geodesic area filtering and MultiPolygon / GeoDataFrame export.
  4. unet_inference with sliding-window Hann/Gaussian blending and small raster padding.
  5. calculate_pixel_area_m2, calculate_flood_area_km2, calculate_flood_percentage,
     and get_flood_summary / calculate_flood_stats.
"""

import os
import shutil
import tempfile
import unittest

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import MultiPolygon, Polygon

from src.flood import (
    calculate_flood_area_km2,
    calculate_flood_percentage,
    calculate_flood_stats,
    calculate_pixel_area_m2,
    create_baseline_mask,
    get_flood_summary,
    mask_to_polygons,
    save_geotiff,
    unet_inference,
)


class TestFloodModule(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------------
    # 1. Baseline Flood Mask & Parameter Validation
    # ------------------------------------------------------------------------

    def test_create_baseline_mask_detection(self):
        pre = np.full((60, 60), -12.0, dtype=np.float32)
        post = np.full((60, 60), -12.0, dtype=np.float32)
        # Create flooded zone (low backscatter < -18 dB and drop > 3 dB)
        post[20:45, 20:45] = -23.0

        mask = create_baseline_mask(
            pre, post, is_db=True, water_db=-18.0, drop_db=3.0, speckle_size=3, min_pixels=20
        )
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, (60, 60))
        self.assertTrue(mask[30, 30])
        self.assertFalse(mask[5, 5])

    def test_create_baseline_mask_permanent_water_exclusion(self):
        pre = np.full((40, 40), -12.0, dtype=np.float32)
        post = np.full((40, 40), -12.0, dtype=np.float32)
        # Permanent river is dark in both pre and post
        pre[:, 15:20] = -22.0
        post[:, 15:20] = -22.0

        mask = create_baseline_mask(pre, post, is_db=True)
        self.assertEqual(np.count_nonzero(mask), 0)

    def test_create_baseline_mask_linear_scale_conversion(self):
        # Linear scale (pre ~ 0.063, post ~ 0.005)
        pre_lin = np.full((50, 50), 0.063, dtype=np.float32)
        post_lin = np.full((50, 50), 0.063, dtype=np.float32)
        post_lin[15:35, 15:35] = 0.005

        mask = create_baseline_mask(pre_lin, post_lin, is_db=False, min_pixels=10)
        self.assertTrue(mask[25, 25])
        self.assertFalse(mask[5, 5])

    def test_create_baseline_mask_validations(self):
        pre = np.full((30, 30), -12.0, dtype=np.float32)
        post = np.full((30, 30), -22.0, dtype=np.float32)

        # None inputs
        with self.assertRaises(ValueError):
            create_baseline_mask(None, post)

        # Shape mismatch
        with self.assertRaises(ValueError):
            create_baseline_mask(pre[:20, :], post)

        # Non-2D
        with self.assertRaises(ValueError):
            create_baseline_mask(pre.flatten(), post.flatten())

        # Empty array
        with self.assertRaises(ValueError):
            create_baseline_mask(np.empty((0, 0)), np.empty((0, 0)))

        # Invalid is_db
        with self.assertRaises(TypeError):
            create_baseline_mask(pre, post, is_db="yes")

        # Invalid drop_db (non-positive)
        with self.assertRaises(ValueError):
            create_baseline_mask(pre, post, drop_db=0.0)
        with self.assertRaises(ValueError):
            create_baseline_mask(pre, post, drop_db=-2.0)

        # Invalid speckle_size (even or < 1)
        with self.assertRaises(ValueError):
            create_baseline_mask(pre, post, speckle_size=2)
        with self.assertRaises(ValueError):
            create_baseline_mask(pre, post, speckle_size=0)
        with self.assertRaises(TypeError):
            create_baseline_mask(pre, post, speckle_size=3.5)

        # Invalid min_pixels (< 0)
        with self.assertRaises(ValueError):
            create_baseline_mask(pre, post, min_pixels=-1)

    # ------------------------------------------------------------------------
    # 2. GeoTIFF Export
    # ------------------------------------------------------------------------

    def test_save_geotiff_uint8_mask(self):
        transform = from_bounds(85.0, 25.0, 85.5, 25.5, 50, 50)
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 10:30] = True

        out_path = os.path.join(self.tmp_dir, "test_mask.tif")
        res_path = save_geotiff(mask, transform, "EPSG:4326", out_path, dtype="uint8")
        self.assertTrue(res_path.exists())

        with rasterio.open(res_path) as src:
            self.assertEqual(src.shape, (50, 50))
            self.assertEqual(src.count, 1)
            self.assertEqual(src.crs.to_string(), "EPSG:4326")
            self.assertEqual(src.nodata, 255.0)
            self.assertEqual(src.profile["compress"], "lzw")
            data = src.read(1)
            self.assertEqual(data[20, 20], 1)
            self.assertEqual(data[0, 0], 0)

    def test_save_geotiff_float32(self):
        transform = from_bounds(85.0, 25.0, 85.5, 25.5, 30, 30)
        probs = np.full((30, 30), 0.85, dtype=np.float32)
        probs[0, 0] = np.nan

        out_path = os.path.join(self.tmp_dir, "test_probs.tif")
        res_path = save_geotiff(probs, transform, "EPSG:4326", out_path, dtype="float32")

        with rasterio.open(res_path) as src:
            self.assertEqual(src.dtypes[0], "float32")
            self.assertTrue(np.isnan(src.nodata))
            data = src.read(1)
            self.assertAlmostEqual(data[15, 15], 0.85, places=4)

    # ------------------------------------------------------------------------
    # 3. Polygonization & Vector GeoDataFrame
    # ------------------------------------------------------------------------

    def test_mask_to_polygons_gdf(self):
        # 0.01 degrees across 100 pixels = ~10m per pixel (111 m2 per pixel)
        transform = from_bounds(85.0, 25.0, 85.01, 25.01, 100, 100)
        mask = np.zeros((100, 100), dtype=bool)
        # Large flooded block (30x30 pixels = ~100,000 m2)
        mask[20:50, 20:50] = True
        # Tiny 1-pixel noise speck (~111 m2 < 500 m2)
        mask[80, 80] = True

        gdf = mask_to_polygons(mask, transform, crs="EPSG:4326", min_area_m2=500.0)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertIn("geometry", gdf.columns)
        self.assertIn("area_m2", gdf.columns)
        self.assertIn("flood_id", gdf.columns)
        self.assertEqual(gdf.crs.to_string(), "EPSG:4326")

        # The tiny noise speck should have been filtered out
        self.assertEqual(len(gdf), 1)
        self.assertGreater(gdf["area_m2"].iloc[0], 500.0)

    def test_mask_to_polygons_multipolygon(self):
        transform = from_bounds(85.0, 25.0, 85.5, 25.5, 50, 50)
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 10:30] = True

        mp = mask_to_polygons(mask, transform, crs="EPSG:4326", as_multipolygon=True)
        self.assertIsInstance(mp, MultiPolygon)
        self.assertFalse(mp.is_empty)

    def test_mask_to_polygons_empty(self):
        transform = from_bounds(85.0, 25.0, 85.5, 25.5, 50, 50)
        empty_mask = np.zeros((50, 50), dtype=bool)

        gdf = mask_to_polygons(empty_mask, transform, crs="EPSG:4326")
        self.assertEqual(len(gdf), 0)

        mp = mask_to_polygons(empty_mask, transform, crs="EPSG:4326", as_multipolygon=True)
        self.assertTrue(mp.is_empty)

    # ------------------------------------------------------------------------
    # 4. U-Net Sliding-Window Inference with Smooth Window Blending
    # ------------------------------------------------------------------------

    def test_unet_inference_shapes_and_blending(self):
        sar = np.random.uniform(-25.0, -5.0, size=(280, 300)).astype(np.float32)
        sar[50:120, 50:120] = -24.0

        prob_map, bin_mask = unet_inference(
            post_sar=sar,
            model_path=None,
            device="cpu",
            tile_size=256,
            stride=128,
            threshold=0.5,
            window_type="hann",
        )

        self.assertEqual(prob_map.shape, (280, 300))
        self.assertEqual(bin_mask.shape, (280, 300))
        self.assertEqual(prob_map.dtype, np.float32)
        self.assertEqual(bin_mask.dtype, bool)
        self.assertGreaterEqual(prob_map.min(), 0.0)
        self.assertLessEqual(prob_map.max(), 1.0)

    def test_unet_inference_small_raster_padding(self):
        # Raster dimensions smaller than tile_size (256)
        small_sar = np.random.uniform(-25.0, -5.0, size=(100, 120)).astype(np.float32)

        prob_map, bin_mask = unet_inference(
            post_sar=small_sar,
            tile_size=256,
            stride=128,
            window_type="gaussian",
        )

        self.assertEqual(prob_map.shape, (100, 120))
        self.assertEqual(bin_mask.shape, (100, 120))

    # ------------------------------------------------------------------------
    # 5. Flood Surface Area and Percentage Helpers
    # ------------------------------------------------------------------------

    def test_calculate_pixel_area_m2(self):
        # EPSG:4326 (geographic degrees)
        transform_geo = from_bounds(85.0, 25.0, 85.5, 25.5, 100, 100)
        px_geo = calculate_pixel_area_m2(transform_geo, crs="EPSG:4326", height_pixels=100)
        self.assertGreater(px_geo, 0.0)

        # UTM (projected meters)
        transform_utm = from_bounds(500000, 2900000, 510000, 2910000, 1000, 1000)
        px_utm = calculate_pixel_area_m2(transform_utm, crs="EPSG:32645")
        self.assertAlmostEqual(px_utm, 100.0, places=1)

    def test_calculate_flood_area_and_percentage(self):
        mask = np.zeros((100, 100), dtype=bool)
        # 1000 flood pixels
        mask[0:10, :] = True

        pct = calculate_flood_percentage(mask)
        self.assertEqual(pct, 10.0)

        # Area with explicit pixel_size_m = 10m (100 m² per pixel)
        # 1000 pixels * 100 m² = 100,000 m² = 0.1 km²
        area_km2 = calculate_flood_area_km2(mask, pixel_size_m=10.0)
        self.assertAlmostEqual(area_km2, 0.1, places=4)

        summary = get_flood_summary(mask, pixel_size_m=10.0)
        self.assertEqual(summary["flood_pixels"], 1000)
        self.assertEqual(summary["total_valid_pixels"], 10000)
        self.assertEqual(summary["flood_percentage"], 10.0)
        self.assertAlmostEqual(summary["flood_area_km2"], 0.1, places=4)

        # Test calculate_flood_stats alias
        stats = calculate_flood_stats(mask, pixel_size_m=10.0)
        self.assertEqual(stats["flooded_area_km2"], 0.1)


if __name__ == "__main__":
    unittest.main()
