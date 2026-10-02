"""
tests/test_pipeline.py - Comprehensive Unit and Integration Test Suite
for the Flood Rescue AI Pipeline.

Test Coverage:
1. Baseline SAR Change Detection (baseline_flood_mask):
   - Permanent water exclusion (dark pre & post -> not flood)
   - Newly submerged land detection (bright pre, dark post with >3dB drop)
   - Nodata / NaN / Inf robustness
   - Linear scale backscatter conversion (to_db)
   - Morphological opening and minimum flood cluster size filtering
2. Road Network Tagging (tag_edges & assess_bridges):
   - Dense spatial sampling along curved road linestrings
   - WGS84 (EPSG:4326) to UTM Projected CRS transformation
   - Off-raster road handling (covered=False, flood_frac=NaN, avoiding false negative dry)
   - Bridge vulnerability assessment and critical severance tagging
3. Topological Isolation Analysis (find_isolated & cluster_isolated):
   - Pre-existing unconnected fragments not falsely reported as cut-offs
   - Super-node (__HUB__) reachability when all hubs are cut off
   - Multi-hub failover (partial hub severance where alternate hub remains reachable)
   - Trapped road network clustering into distinct connected communities
4. Population Summation and Priority Ranking Calculation:
   - Zonal population summation with buffer masking and NaN safety
   - Multi-factor priority score calculation (high population + distance -> Rank 1)
   - Operational rescue manifest generation and priority tier classification
5. GeoTIFF and GeoJSON File Exports:
   - Georeferenced GeoTIFF round-trip verification with rasterio
   - Roads status GeoJSON export validation
   - Isolated settlements GeoJSON export validation
   - Rescue dispatch manifest CSV and GeoJSON export validation
"""

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.transform import Affine
from shapely.geometry import LineString, Point, shape

# Import core algorithms
from src.core_logic import (
    baseline_flood_mask,
    cluster_isolated,
    find_isolated,
    snap_to_nodes,
    tag_edges,
    to_db,
)

# Import pipeline modular functions
from src.flood import create_baseline_mask, save_geotiff
from src.roads import assess_bridges, compute_road_stats, roads_to_geojson, tag_road_network
from src.isolation import identify_cut_off_settlements, isolation_to_geojson
from src.priority import calculate_priority, export_rescue_plan, generate_rescue_manifest


class TestBaselineSARChangeDetection(unittest.TestCase):
    """
    Test suite for SAR change detection algorithms.
    Ensures physically grounded radar backscatter drop detection while
    filtering permanent water bodies and handling nodata/NaN values safely.
    """

    def setUp(self):
        # Create standard 60x60 synthetic radar grids (dB scale)
        # Background: dry land at -10 dB (bright backscatter)
        self.pre_db = np.full((60, 60), -10.0, dtype=np.float32)
        self.post_db = np.full((60, 60), -10.0, dtype=np.float32)

        # Region A: Pre-existing permanent water body (e.g. lake/river)
        # Dark in both pre and post (-22 dB)
        self.pre_db[5:20, 5:20] = -22.0
        self.post_db[5:20, 5:20] = -22.0

        # Region B: Newly submerged land (catastrophic flood)
        # Bright in pre (-9 dB), very dark in post (-23 dB -> drop = 14 dB > 3 dB)
        self.pre_db[25:50, 25:50] = -9.0
        self.post_db[25:50, 25:50] = -23.0

        # Region C: Minor soil moisture variation (not flood)
        # -10 dB pre, -12 dB post (drop = 2 dB < 3 dB, and -12 dB > -18 dB water threshold)
        self.pre_db[5:20, 35:50] = -10.0
        self.post_db[5:20, 35:50] = -12.0

    def test_permanent_water_not_marked_as_flood(self):
        """
        Verifies that pre-existing permanent water bodies (dark in both pre and post)
        are NOT marked as new floods.
        """
        mask = baseline_flood_mask(
            pre=self.pre_db,
            post=self.post_db,
            is_db=True,
            water_db=-18.0,
            drop_db=3.0,
            min_pixels=50,
        )

        # Permanent water region [5:20, 5:20] must have ZERO flooded pixels
        permanent_water_flooded = mask[5:20, 5:20].sum()
        self.assertEqual(
            permanent_water_flooded,
            0,
            f"Permanent water was erroneously flagged as flood ({permanent_water_flooded} pixels).",
        )

    def test_newly_submerged_land_accurately_detected(self):
        """
        Verifies that newly submerged land (bright pre, dark post with >3dB drop)
        is accurately marked as flood.
        """
        mask = baseline_flood_mask(
            pre=self.pre_db,
            post=self.post_db,
            is_db=True,
            water_db=-18.0,
            drop_db=3.0,
            min_pixels=50,
        )

        # Region B is 25x25 = 625 pixels.
        # After 3x3 median filter and 3x3 binary opening, core interior must be True.
        flooded_patch = mask[27:48, 27:48]
        self.assertTrue(
            flooded_patch.all(),
            "Core pixels of newly submerged land were not detected as flood.",
        )
        self.assertGreater(
            mask.sum(),
            500,
            "Detected flood area is smaller than expected for 625-pixel inundation.",
        )

    def test_nodata_and_nan_handling(self):
        """
        Verifies nodata / NaN / Inf handling.
        Nodata pixels must never be marked as flood or cause runtime exceptions.
        """
        pre_nan = self.pre_db.copy()
        post_nan = self.post_db.copy()

        # Inject NaN and Inf into various locations
        pre_nan[0:5, :] = np.nan
        post_nan[0:5, :] = np.nan
        pre_nan[55:60, 55:60] = -np.inf
        post_nan[55:60, 55:60] = np.inf

        # Inject NaN inside what would otherwise be a flooded region
        post_nan[30:35, 30:35] = np.nan

        mask = baseline_flood_mask(
            pre=pre_nan,
            post=post_nan,
            is_db=True,
            water_db=-18.0,
            drop_db=3.0,
            min_pixels=50,
        )

        # Verify no NaN or Inf pixels are marked as flood
        self.assertEqual(mask[0:5, :].sum(), 0, "NaN rows were falsely flagged as flood.")
        self.assertEqual(mask[55:60, 55:60].sum(), 0, "Inf cells were falsely flagged as flood.")
        self.assertEqual(mask[30:35, 30:35].sum(), 0, "NaN cells in flood zone must be False.")
        self.assertEqual(mask.dtype, bool, "Mask must be boolean.")

    def test_linear_backscatter_to_db_conversion(self):
        """
        Verifies that linear radar amplitude values (power) are properly converted to dB.
        Linear power: P_linear = 10^(dB / 10).
        """
        # Linear values corresponding to -10 dB (0.1) and -24 dB (~0.00398)
        pre_linear = np.full((60, 60), 0.1, dtype=np.float32)
        post_linear = np.full((60, 60), 0.1, dtype=np.float32)

        # Flood patch in linear scale
        pre_linear[25:50, 25:50] = 0.12589    # ~ -9 dB
        post_linear[25:50, 25:50] = 0.00398   # ~ -24 dB

        # Run with is_db=False
        mask = baseline_flood_mask(
            pre=pre_linear,
            post=post_linear,
            is_db=False,
            water_db=-18.0,
            drop_db=3.0,
            min_pixels=50,
        )
        self.assertGreater(mask[27:48, 27:48].sum(), 400)

        # Verify to_db helper handles zero / tiny values safely without crashing on log(0)
        zeros = np.zeros((5, 5), dtype=np.float32)
        db_vals = to_db(zeros)
        self.assertTrue(np.all(np.isfinite(db_vals)))
        self.assertAlmostEqual(float(db_vals[0, 0]), -60.0, places=1)

    def test_speckle_noise_and_min_pixels_filtering(self):
        """
        Verifies that radar speckle noise (single pixel false positives) is removed
        by binary opening and connected component thresholding.
        """
        pre_noisy = np.full((60, 60), -10.0, dtype=np.float32)
        post_noisy = np.full((60, 60), -10.0, dtype=np.float32)

        # Inject tiny isolated false alarms (single pixels)
        post_noisy[10, 10] = -25.0
        post_noisy[12, 12] = -25.0
        post_noisy[40, 10] = -25.0

        # Inject small 3x3 patch (9 pixels < min_pixels=50)
        pre_noisy[40:43, 40:43] = -9.0
        post_noisy[40:43, 40:43] = -24.0

        mask = baseline_flood_mask(
            pre=pre_noisy,
            post=post_noisy,
            is_db=True,
            water_db=-18.0,
            drop_db=3.0,
            speckle_size=3,
            min_pixels=50,
        )
        self.assertEqual(mask.sum(), 0, "Noise specks (<50 pixels) should have been filtered out.")

    def test_create_baseline_mask_wrapper(self):
        """
        Verifies that src.flood.create_baseline_mask wraps baseline_flood_mask accurately.
        """
        mask1 = baseline_flood_mask(self.pre_db, self.post_db, is_db=True, min_pixels=50)
        mask2 = create_baseline_mask(self.pre_db, self.post_db, is_db=True, min_pixels=50)
        np.testing.assert_array_equal(mask1, mask2)


class TestRoadNetworkTagging(unittest.TestCase):
    """
    Test suite for road network intersection and bridge risk tagging.
    Tests spatial interpolation along linestrings, CRS reprojection between
    WGS84 lon/lat and Projected UTM, and handling of off-raster edges.
    """

    def setUp(self):
        # We set up a 100x100 raster in UTM Zone 42N (EPSG:32642 - e.g. Sindh, Pakistan)
        # Resolution: 10m x 10m pixels
        # Extent: X from 380,000 to 381,000; Y from 2,959,000 to 2,960,000
        self.raster_crs = "EPSG:32642"
        self.graph_crs = "EPSG:4326"
        self.h, self.w = 100, 100
        self.pixel_size = 10.0
        self.x_min = 380000.0
        self.y_max = 2960000.0
        self.transform = Affine(self.pixel_size, 0.0, self.x_min, 0.0, -self.pixel_size, self.y_max)

        # Flood mask: Top half (rows 0-49) is DRY, Bottom half (rows 50-99) is FLOODED
        self.mask = np.zeros((self.h, self.w), dtype=bool)
        self.mask[50:100, :] = True

        # Helper transformer from UTM -> WGS84 lon/lat for constructing realistic graph nodes
        self.utm_to_wgs = Transformer.from_crs(self.raster_crs, self.graph_crs, always_xy=True).transform
        self.wgs_to_utm = Transformer.from_crs(self.graph_crs, self.raster_crs, always_xy=True).transform

    def _utm_to_lonlat(self, x_utm, y_utm):
        return self.utm_to_wgs(x_utm, y_utm)

    def test_sampling_along_linestring_geometry(self):
        """
        Verifies sampling along curved road linestrings.
        If endpoints are on dry land, but the middle of the linestring dips into
        a flooded zone, multi-point interpolation must detect the inundation.
        """
        G = nx.MultiDiGraph()

        # Both endpoints are in dry territory (row 20 -> y = 2959800)
        lon_u, lat_u = self._utm_to_lonlat(380200.0, 2959800.0)
        lon_v, lat_v = self._utm_to_lonlat(380800.0, 2959800.0)
        G.add_node("U", x=lon_u, y=lat_u)
        G.add_node("V", x=lon_v, y=lat_v)

        # Construct a curved road linestring that dips into the flooded half (row 75 -> y = 2959250)
        # in WGS84 degrees
        pt_mid1 = self._utm_to_lonlat(380350.0, 2959250.0)
        pt_mid2 = self._utm_to_lonlat(380650.0, 2959250.0)
        curved_geom = LineString([(lon_u, lat_u), pt_mid1, pt_mid2, (lon_v, lat_v)])

        G.add_edge("U", "V", key=0, geometry=curved_geom, length=1200.0)

        # Tag edges
        tagged_G = tag_edges(
            G=G,
            mask=self.mask,
            transform=self.transform,
            raster_crs=self.raster_crs,
            graph_crs=self.graph_crs,
            flooded_frac=0.3,
        )

        edge_data = tagged_G.edges["U", "V", 0]
        self.assertTrue(edge_data["covered"])
        self.assertGreaterEqual(
            edge_data["flood_frac"],
            0.3,
            "Curved linestring through floodwaters should exceed 30% flood fraction.",
        )
        self.assertTrue(
            edge_data["flooded"],
            "Edge dipping into flooded zone should be flagged as flooded=True.",
        )

    def test_crs_transformation_graph_wgs84_to_raster_utm(self):
        """
        Verifies coordinate transformation between graph CRS (EPSG:4326) and raster UTM.
        Straight edge placed strictly in dry upper half vs flooded lower half.
        """
        G = nx.MultiDiGraph()

        # Dry road in upper half (row 15 to 25 in UTM)
        u1_lon, u1_lat = self._utm_to_lonlat(380100.0, 2959850.0)
        v1_lon, v1_lat = self._utm_to_lonlat(380900.0, 2959750.0)
        G.add_node("dry_1", x=u1_lon, y=u1_lat)
        G.add_node("dry_2", x=v1_lon, y=v1_lat)
        G.add_edge("dry_1", "dry_2", key=0)

        # Flooded road in lower half (row 70 to 80 in UTM)
        u2_lon, u2_lat = self._utm_to_lonlat(380100.0, 2959300.0)
        v2_lon, v2_lat = self._utm_to_lonlat(380900.0, 2959200.0)
        G.add_node("flood_1", x=u2_lon, y=u2_lat)
        G.add_node("flood_2", x=v2_lon, y=v2_lat)
        G.add_edge("flood_1", "flood_2", key=0)

        tagged_G = tag_edges(
            G=G,
            mask=self.mask,
            transform=self.transform,
            raster_crs=self.raster_crs,
            graph_crs=self.graph_crs,
            flooded_frac=0.3,
        )

        dry_edge = tagged_G.edges["dry_1", "dry_2", 0]
        self.assertTrue(dry_edge["covered"])
        self.assertAlmostEqual(dry_edge["flood_frac"], 0.0, places=2)
        self.assertFalse(dry_edge["flooded"])

        flood_edge = tagged_G.edges["flood_1", "flood_2", 0]
        self.assertTrue(flood_edge["covered"])
        self.assertAlmostEqual(flood_edge["flood_frac"], 1.0, places=2)
        self.assertTrue(flood_edge["flooded"])

    def test_off_raster_segments_marked_uncovered(self):
        """
        Verifies off-raster road segments are marked covered=False instead of
        false negative dry.
        """
        G = nx.MultiDiGraph()

        # Place nodes far away from the raster footprint (e.g. 50 km east in UTM)
        off_u_lon, off_u_lat = self._utm_to_lonlat(430000.0, 2960000.0)
        off_v_lon, off_v_lat = self._utm_to_lonlat(435000.0, 2960000.0)
        G.add_node("off_u", x=off_u_lon, y=off_u_lat)
        G.add_node("off_v", x=off_v_lon, y=off_v_lat)
        G.add_edge("off_u", "off_v", key=0)

        tagged_G = tag_edges(
            G=G,
            mask=self.mask,
            transform=self.transform,
            raster_crs=self.raster_crs,
            graph_crs=self.graph_crs,
            flooded_frac=0.3,
        )

        edge = tagged_G.edges["off_u", "off_v", 0]
        self.assertFalse(
            edge["covered"],
            "Off-raster edge must have covered=False.",
        )
        self.assertFalse(
            edge["flooded"],
            "Off-raster edge must have flooded=False (unknown status, not confirmed flooded).",
        )
        self.assertTrue(
            math.isnan(edge["flood_frac"]),
            f"Off-raster edge must have flood_frac=NaN, got {edge['flood_frac']}.",
        )

    def test_assess_bridges_and_road_stats(self):
        """
        Verifies bridge tagging logic and aggregate road network damage statistics.
        """
        G = nx.MultiDiGraph()
        u_lon, u_lat = self._utm_to_lonlat(380100.0, 2959200.0)
        v_lon, v_lat = self._utm_to_lonlat(380300.0, 2959200.0)
        G.add_node(1, x=u_lon, y=u_lat)
        G.add_node(2, x=v_lon, y=v_lat)
        G.add_edge(1, 2, key=0, bridge="yes", length=200.0)

        # Tag network and assess bridges
        G = tag_road_network(G, self.mask, self.transform, self.raster_crs)
        G = assess_bridges(G, self.mask, self.transform, self.raster_crs)

        edge_data = G.edges[1, 2, 0]
        self.assertTrue(edge_data["is_bridge"])
        self.assertTrue(edge_data.get("likely_damaged", False) or edge_data.get("bridge_damaged", False))
        self.assertIn(edge_data["status"], ["damaged_bridge", "likely_damaged"])

        stats = compute_road_stats(G)
        self.assertEqual(stats["damaged_bridges_count"], 1)
        self.assertGreater(stats["flooded_road_km"], 0.0)
        self.assertGreater(stats["total_road_km"], 0.0)


class TestCutOffIsolationAnalysis(unittest.TestCase):
    """
    Test suite for topological cut-off isolation analysis.
    Verifies that settlements with no prior hub connection are ignored,
    super-node __HUB__ reachability functions under partial and total severance,
    and disconnected communities are properly clustered.
    """

    def setUp(self):
        # We construct a network of nodes:
        # Hubs: H1 (node 100), H2 (node 200)
        # Settlement Alpha: nodes 1, 2, 3
        # Settlement Beta:  nodes 4, 5, 6
        # Pre-existing Disconnected Island: nodes 98, 99 (never connected to H1 or H2)
        self.G = nx.MultiDiGraph()

        coords = {
            100: (67.80, 26.70),  # H1
            200: (67.90, 26.70),  # H2
            1: (67.81, 26.71),
            2: (67.82, 26.71),
            3: (67.83, 26.71),
            4: (67.85, 26.75),
            5: (67.86, 26.75),
            6: (67.87, 26.75),
            98: (67.60, 26.50),  # Pre-existing dead-end
            99: (67.61, 26.50),  # Pre-existing dead-end
        }
        for n, (x, y) in coords.items():
            self.G.add_node(n, x=x, y=y)

        # Edges before flood:
        # H1 connects to Alpha: 100 <-> 1
        # Alpha internal: 1 <-> 2 <-> 3
        # Alpha connects to Beta: 3 <-> 4
        # Beta internal: 4 <-> 5 <-> 6
        # Beta connects to H2: 6 <-> 200
        # Disconnected Island: 98 <-> 99 (isolated even before flood)
        edges = [
            (100, 1), (1, 2), (2, 3),
            (3, 4),
            (4, 5), (5, 6),
            (6, 200),
            (98, 99),
        ]
        for u, v in edges:
            self.G.add_edge(u, v, key=0, flooded=False)
            self.G.add_edge(v, u, key=0, flooded=False)

        self.hubs = [100, 200]

    def test_unconnected_nodes_not_falsely_reported_as_isolated(self):
        """
        Verifies that nodes with no prior connection to a hub are NOT falsely
        reported as flood cut-offs.
        """
        # No flood on any edge
        isolated, after_graph = find_isolated(self.G, self.hubs)

        # Nodes 98 and 99 had NO path to H1 or H2 in the before graph
        self.assertNotIn(98, isolated)
        self.assertNotIn(99, isolated)
        self.assertEqual(len(isolated), 0, "Zero nodes should be cut off when no roads are flooded.")

    def test_super_node_hub_behavior_when_all_hubs_cut_off(self):
        """
        Verifies super-node __HUB__ behavior when all access roads to all hubs are flooded.
        Both Alpha and Beta lose connection to both H1 and H2.
        """
        # Sever access to H1 (edge 100 <-> 1) and H2 (edge 6 <-> 200)
        for u, v in [(100, 1), (1, 100), (6, 200), (200, 6)]:
            self.G.edges[u, v, 0]["flooded"] = True

        isolated, after_graph = find_isolated(self.G, self.hubs)

        # Settlements 1, 2, 3, 4, 5, 6 and severed hub nodes 100, 200 had access before
        # and lost it -> MUST be isolated
        expected_settlements = {1, 2, 3, 4, 5, 6}
        self.assertTrue(expected_settlements.issubset(isolated))

        # Unconnected nodes 98 and 99 must STILL not be included
        self.assertNotIn(98, isolated)
        self.assertNotIn(99, isolated)

        # Verify super-node __HUB__ was properly cleaned up and removed from after_graph
        self.assertFalse(
            after_graph.has_node("__HUB__"),
            "Super-node __HUB__ must be removed from after_graph before returning.",
        )

    def test_super_node_hub_behavior_when_one_hub_remains_accessible(self):
        """
        Verifies super-node behavior when one hub is cut off but another hub
        remains reachable via an alternate dry route.
        """
        # Sever access only to H1 (edge 100 <-> 1)
        # H2 is still connected to Beta, and Alpha can reach Beta via edge (3, 4)
        for u, v in [(100, 1), (1, 100)]:
            self.G.edges[u, v, 0]["flooded"] = True

        isolated, after_graph = find_isolated(self.G, self.hubs)

        # Settlements {1, 2, 3, 4, 5, 6} can all route to H2, so NONE of them are cut off!
        settlements_isolated = isolated - set(self.hubs)
        self.assertEqual(
            len(settlements_isolated),
            0,
            f"Alternate route to H2 exists, but settlements were marked isolated: {settlements_isolated}",
        )
        # Hub 100 itself is severed (unreachable hub)
        self.assertIn(100, isolated)

        # Now also sever road between Alpha and Beta (edge 3 <-> 4)
        for u, v in [(3, 4), (4, 3)]:
            self.G.edges[u, v, 0]["flooded"] = True

        isolated_now, _ = find_isolated(self.G, self.hubs)

        # Now Alpha {1, 2, 3} has lost both H1 and H2 -> MUST be isolated
        # Beta {4, 5, 6} can still reach H2 -> MUST NOT be isolated
        self.assertTrue({1, 2, 3}.issubset(isolated_now))
        self.assertNotIn(4, isolated_now)
        self.assertNotIn(5, isolated_now)
        self.assertNotIn(6, isolated_now)

    def test_clustering_into_connected_communities(self):
        """
        Verifies clustering of cut-off nodes into locally connected communities
        via cluster_isolated.
        """
        # Sever access to both hubs AND sever the road between Alpha and Beta
        for u, v in [(100, 1), (1, 100), (6, 200), (200, 6), (3, 4), (4, 3)]:
            self.G.edges[u, v, 0]["flooded"] = True

        isolated, after_graph = find_isolated(self.G, self.hubs)
        self.assertTrue({1, 2, 3, 4, 5, 6}.issubset(isolated))

        # Cluster with min_nodes=2 to group multi-node settlements (Alpha and Beta)
        clusters = cluster_isolated(self.G, isolated, after_graph, min_nodes=2)

        # Should produce exactly 2 distinct multi-node clusters: Alpha {1, 2, 3} and Beta {4, 5, 6}
        self.assertEqual(len(clusters), 2)
        clustered_node_sets = [set(c["nodes"]) for c in clusters]
        self.assertIn({1, 2, 3}, clustered_node_sets)
        self.assertIn({4, 5, 6}, clustered_node_sets)


class TestPopulationSummationAndPriorityRanking(unittest.TestCase):
    """
    Test suite for zonal population summation and multi-factor rescue priority ranking.
    Verifies that the highest population + deepest medical distance cluster receives Rank 1.
    """

    def setUp(self):
        # Build 2 settlement clusters for priority ranking
        # Cluster Alpha: Trapped near Dadu (Sindh), high population, far from hub, bridge severed
        self.cluster_critical = {
            "cluster_id": "ISOL-CRITICAL",
            "population": 4500.0,
            "dist_to_hub_km": 16.5,
            "trapped_nodes": 12,
            "has_bridge_cut": True,
            "has_highway_cut": True,
            "lat": 26.75,
            "lon": 67.85,
            "nearest_hub": "Dadu Civil Hospital",
        }

        # Cluster Beta: Small hamlet, low population, close to staging hub, no bridge cut
        self.cluster_low = {
            "cluster_id": "ISOL-LOW",
            "population": 45.0,
            "dist_to_hub_km": 1.2,
            "trapped_nodes": 2,
            "has_bridge_cut": False,
            "has_highway_cut": False,
            "lat": 26.71,
            "lon": 67.81,
            "nearest_hub": "Dadu Civil Hospital",
        }

    def test_population_summation_zonal_integration(self):
        """
        Verifies population summation from raster cells inside buffered cluster footprints,
        including NaN safety.
        """
        G = nx.Graph()
        G.add_node(1, x=67.800, y=26.700)
        G.add_node(2, x=67.802, y=26.702)
        G.add_edge(1, 2)

        # Synthetic 50x50 population raster around (67.80, 26.70)
        # Resolution: ~0.001 deg (~110m)
        pop_arr = np.full((50, 50), 25.0, dtype=np.float32)
        # Inject NaN values to test robustness
        pop_arr[10:15, 10:15] = np.nan

        pop_transform = Affine(0.001, 0.0, 67.78, 0.0, -0.001, 26.72)

        clusters = cluster_isolated(
            G=G,
            isolated={1, 2},
            after_graph=G,
            pop_array=pop_arr,
            pop_transform=pop_transform,
            pop_crs="EPSG:4326",
            graph_crs="EPSG:4326",
            buffer_m=350,
        )

        self.assertEqual(len(clusters), 1)
        cluster = clusters[0]
        self.assertIsNotNone(cluster["population"])
        self.assertTrue(np.isfinite(cluster["population"]))
        self.assertGreater(
            cluster["population"],
            0.0,
            "Population under buffered footprint must be greater than zero.",
        )

    def test_highest_population_and_distance_gets_rank_1(self):
        """
        Verifies that the settlement with highest population + distance gets Rank 1.
        """
        raw_clusters = [self.cluster_low, self.cluster_critical]

        # Calculate multi-factor priority
        ranked_clusters = calculate_priority(raw_clusters)

        # First item in ranked_clusters must be the critical cluster
        top_cluster = ranked_clusters[0]
        self.assertEqual(
            top_cluster["cluster_id"],
            "ISOL-CRITICAL",
            f"Expected ISOL-CRITICAL to be top priority, got {top_cluster['cluster_id']}",
        )
        self.assertGreater(
            top_cluster["priority_score"],
            ranked_clusters[1]["priority_score"],
            "Critical cluster priority score must exceed low cluster score.",
        )

        # Generate dispatch manifest DataFrame
        manifest_df = generate_rescue_manifest(ranked_clusters)

        # Verify Rank 1 row
        rank1_row = manifest_df[manifest_df["Rank"] == 1].iloc[0]
        self.assertEqual(rank1_row["Cluster_ID"], "ISOL-CRITICAL")
        self.assertIn(rank1_row["Priority_Tier"], ["CRITICAL", "HIGH"])
        self.assertEqual(int(rank1_row["Estimated_Population"]), 4500)

        # Verify Rank 2 row
        rank2_row = manifest_df[manifest_df["Rank"] == 2].iloc[0]
        self.assertEqual(rank2_row["Cluster_ID"], "ISOL-LOW")
        self.assertIn(rank2_row["Priority_Tier"], ["LOW", "MEDIUM"])

    def test_empty_cluster_inputs(self):
        """
        Verifies that empty cluster lists are handled cleanly without exceptions.
        """
        empty_res = calculate_priority([])
        self.assertEqual(empty_res, [])

        empty_df = generate_rescue_manifest([])
        self.assertTrue(empty_df.empty)
        self.assertIn("Rank", empty_df.columns)
        self.assertIn("Priority_Score", empty_df.columns)


class TestGeoTIFFAndGeoJSONExports(unittest.TestCase):
    """
    Test suite for GeoTIFF raster and GeoJSON vector file exports.
    Verifies that spatial metadata (CRS, transform, coordinates, attributes)
    survives export round-trips.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_geotiff_export_and_read_roundtrip(self):
        """
        Verifies saving raster masks via save_geotiff and reading back with rasterio.
        """
        mask_data = np.zeros((30, 40), dtype=np.uint8)
        mask_data[10:25, 10:25] = 1  # Inundated patch

        transform = Affine(10.0, 0.0, 380000.0, 0.0, -10.0, 2960000.0)
        crs = "EPSG:32642"
        out_tif = os.path.join(self.output_dir, "test_flood_mask.tif")

        # Save GeoTIFF
        saved_path = save_geotiff(mask_data, transform, crs, out_tif)
        self.assertTrue(os.path.exists(saved_path))

        # Re-read with rasterio to verify integrity
        with rasterio.open(saved_path) as src:
            read_arr = src.read(1)
            self.assertEqual(src.shape, (30, 40))
            self.assertEqual(src.crs.to_string(), crs)
            self.assertEqual(src.transform.a, 10.0)
            self.assertEqual(src.transform.e, -10.0)
            np.testing.assert_array_equal(read_arr, mask_data)

    def test_roads_geojson_export(self):
        """
        Verifies exporting road network status to valid GeoJSON FeatureCollection.
        """
        G = nx.MultiDiGraph()
        G.add_node(1, x=67.80, y=26.70)
        G.add_node(2, x=67.85, y=26.75)
        G.add_edge(
            1, 2, key=0,
            status="flooded",
            flooded=True,
            flood_frac=0.85,
            length=1500.0,
            is_bridge=True,
            bridge_risk="CRITICAL / SEVERED",
        )

        out_geojson = os.path.join(self.output_dir, "roads_status.geojson")
        roads_to_geojson(G, output_path=out_geojson)

        self.assertTrue(os.path.exists(out_geojson))
        with open(out_geojson, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("type"), "FeatureCollection")
        self.assertGreater(len(data.get("features", [])), 0)

        feat = data["features"][0]
        self.assertEqual(feat["geometry"]["type"], "LineString")
        props = feat["properties"]
        self.assertEqual(props["status"], "flooded")
        self.assertAlmostEqual(props["flood_frac"], 0.85, places=2)

    def test_isolation_and_rescue_manifest_exports(self):
        """
        Verifies exporting isolated settlement clusters and ranked rescue manifest
        to GeoJSON and CSV formats.
        """
        cluster = {
            "cluster_id": "ISOL-01",
            "rank": 1,
            "population": 1200,
            "n_nodes": 8,
            "lon": 67.82,
            "lat": 26.73,
            "nearest_hub": "Dadu Hospital",
            "dist_hub_km": 8.5,
            "dist_to_hub_km": 8.5,
            "bridge_severed": True,
            "has_bridge_cut": True,
            "priority_score": 82.4,
            "priority_tier": "CRITICAL",
            "recommended_action": "Heavy Helicopter Airdrop",
        }

        # 1. Test isolation_to_geojson
        isol_json_path = os.path.join(self.output_dir, "isolated.geojson")
        isolation_to_geojson([cluster], output_path=isol_json_path)
        self.assertTrue(os.path.exists(isol_json_path))

        with open(isol_json_path, "r", encoding="utf-8") as f:
            isol_data = json.load(f)
        self.assertEqual(isol_data["type"], "FeatureCollection")
        self.assertEqual(isol_data["features"][0]["geometry"]["coordinates"], [67.82, 26.73])

        # 2. Test export_rescue_plan (CSV and GeoJSON)
        ranked_clusters = calculate_priority([cluster])
        manifest_df = generate_rescue_manifest(ranked_clusters)

        csv_path = os.path.join(self.output_dir, "manifest.csv")
        json_path = os.path.join(self.output_dir, "manifest.geojson")
        export_rescue_plan(manifest_df, csv_path, json_path)

        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(json_path))

        # Validate CSV contents
        loaded_df = pd.read_csv(csv_path)
        self.assertEqual(len(loaded_df), 1)
        self.assertEqual(loaded_df["Rank"].iloc[0], 1)
        self.assertEqual(loaded_df["Cluster_ID"].iloc[0], "ISOL-01")

        # Validate GeoJSON contents
        with open(json_path, "r", encoding="utf-8") as f:
            manifest_geojson = json.load(f)
        self.assertEqual(manifest_geojson["type"], "FeatureCollection")
        self.assertEqual(len(manifest_geojson["features"]), 1)
        feat_props = manifest_geojson["features"][0]["properties"]
        self.assertEqual(feat_props["Rank"], 1)


if __name__ == "__main__":
    unittest.main()
