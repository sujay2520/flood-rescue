"""
Unit and integration tests for src/isolation.py and src/priority.py
"""

import os
import shutil
import tempfile
import unittest
import networkx as nx
import numpy as np
import pandas as pd
from affine import Affine

from src.core_logic import snap_to_nodes
from src.isolation import identify_cut_off_settlements, isolation_to_geojson
from src.priority import (
    calculate_priority,
    generate_rescue_manifest,
    export_rescue_plan,
    DEFAULT_WEIGHTS,
)


class TestIsolationAndPriority(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

        # Build a synthetic road network
        # 1 - 2 - 3 - 4 (Hub)
        # |   |   |
        # 5 - 6 - 7
        # Hub is at node 4 (Hospital)
        # Nodes 1, 2, 5, 6 will be cut off by flooding edges (2, 3) and (6, 7)
        self.G = nx.Graph()
        nodes_data = {
            1: {"x": 67.80, "y": 26.70},
            2: {"x": 67.81, "y": 26.70},
            3: {"x": 67.82, "y": 26.70},
            4: {"x": 67.83, "y": 26.70, "name": "Civil Hospital Dadu", "amenity": "hospital"},
            5: {"x": 67.80, "y": 26.69},
            6: {"x": 67.81, "y": 26.69},
            7: {"x": 67.82, "y": 26.69},
        }
        for n, d in nodes_data.items():
            self.G.add_node(n, **d)

        # Edges
        # Bridge on (2, 3)
        self.G.add_edge(1, 2, flooded=False, flood_frac=0.0, highway="residential", name="Street 1")
        self.G.add_edge(1, 5, flooded=False, flood_frac=0.0, highway="residential", name="Street 2")
        self.G.add_edge(5, 6, flooded=False, flood_frac=0.0, highway="residential", name="Street 3")
        self.G.add_edge(2, 6, flooded=False, flood_frac=0.0, highway="residential", name="Street 4")
        self.G.add_edge(2, 3, flooded=True, flood_frac=0.9, bridge="yes", highway="primary", name="Indus River Bridge")
        self.G.add_edge(6, 7, flooded=True, flood_frac=0.8, highway="secondary", name="Canal Link Road")
        self.G.add_edge(3, 4, flooded=False, flood_frac=0.0, highway="primary", name="Hospital Way")
        self.G.add_edge(3, 7, flooded=False, flood_frac=0.0, highway="secondary", name="East Bypass")
        self.G.add_edge(7, 4, flooded=False, flood_frac=0.0, highway="secondary", name="South Bypass")

        # Population raster (small 10x10)
        self.pop_arr = np.full((10, 10), 100.0, dtype=np.float32)
        # Affine transform covering roughly 67.79 to 67.85, 26.68 to 26.72
        self.pop_transform = Affine.translation(67.79, 26.72) @ Affine.scale(0.006, -0.004)

        self.hub_nodes = [4]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_identify_cut_off_settlements(self):
        clusters, isolated = identify_cut_off_settlements(
            self.G,
            self.hub_nodes,
            pop_array=self.pop_arr,
            pop_transform=self.pop_transform,
            pop_crs="EPSG:4326",
            graph_crs="EPSG:4326",
            buffer_m=300.0,
        )

        self.assertEqual(isolated, {1, 2, 5, 6})
        self.assertEqual(len(clusters), 1)

        c = clusters[0]
        self.assertTrue(c["cluster_id"].startswith("Village_Cluster"))
        self.assertEqual(c["n_trapped_nodes"], 4)
        self.assertGreater(c["estimated_population"], 0.0)
        self.assertGreater(c["radius_m"], 0.0)
        self.assertGreater(c["dist_to_hub_km"], 0.0)
        self.assertEqual(c["nearest_hub"], "Civil Hospital Dadu")
        self.assertTrue(c["has_bridge_cut"])
        self.assertIn("bridge", c["critical_access_obstacles"].lower())

    def test_isolation_to_geojson(self):
        clusters, _ = identify_cut_off_settlements(
            self.G,
            self.hub_nodes,
            pop_array=self.pop_arr,
            pop_transform=self.pop_transform,
        )

        geojson_path = os.path.join(self.tmp_dir, "test_clusters.geojson")
        geojson_data = isolation_to_geojson(clusters, output_path=geojson_path)

        self.assertEqual(geojson_data["type"], "FeatureCollection")
        self.assertEqual(len(geojson_data["features"]), 1)
        feat = geojson_data["features"][0]
        self.assertIn("cluster_id", feat["properties"])
        self.assertIn("estimated_population", feat["properties"])
        self.assertIn("critical_access_obstacles", feat["properties"])
        self.assertTrue(os.path.exists(geojson_path))

    def test_calculate_priority_and_manifest(self):
        clusters, _ = identify_cut_off_settlements(
            self.G,
            self.hub_nodes,
            pop_array=self.pop_arr,
            pop_transform=self.pop_transform,
        )

        # Add a second dummy cluster to test ranking and comparison
        c2 = dict(clusters[0])
        c2["cluster_id"] = "Village_Cluster_02"
        c2["estimated_population"] = 50.0
        c2["trapped_nodes"] = 2
        c2["dist_to_hub_km"] = 1.0
        c2["has_bridge_cut"] = False
        c2["critical_access_obstacles"] = "Flooded residential street"
        multi_clusters = [clusters[0], c2]

        ranked = calculate_priority(multi_clusters)
        self.assertEqual(len(ranked), 2)
        # c1 has high pop + bridge cut + higher distance -> higher priority score
        self.assertGreaterEqual(ranked[0]["priority_score"], ranked[1]["priority_score"])
        self.assertIn(ranked[0]["priority_tier"], ["CRITICAL", "HIGH", "MEDIUM", "LOW"])

        # Generate manifest
        manifest_df = generate_rescue_manifest(ranked)
        expected_cols = [
            "Rank",
            "Cluster_ID",
            "Priority_Tier",
            "Priority_Score",
            "Estimated_Population",
            "Trapped_Nodes",
            "Dist_to_Hub_km",
            "Nearest_Hub",
            "Centroid_Lat",
            "Centroid_Lon",
            "Recommended_Action",
        ]
        self.assertEqual(list(manifest_df.columns), expected_cols)
        self.assertEqual(manifest_df.iloc[0]["Rank"], 1)

        # Test export
        csv_p = os.path.join(self.tmp_dir, "manifest.csv")
        json_p = os.path.join(self.tmp_dir, "manifest.geojson")
        res = export_rescue_plan(manifest_df, csv_p, json_p)
        self.assertTrue(os.path.exists(csv_p))
        self.assertTrue(os.path.exists(json_p))

    def test_multidigraph_compatibility(self):
        # OSMnx graphs are MultiDiGraphs with parallel edges and directional tags
        MG = nx.MultiDiGraph()
        MG.add_node(10, x=67.80, y=26.70)
        MG.add_node(20, x=67.81, y=26.70)
        MG.add_node(30, x=67.82, y=26.70, name="Base Hospital", amenity="hospital")
        MG.add_node(40, x=67.83, y=26.70)

        # Two parallel edges between 10 and 20 (both dry)
        MG.add_edge(10, 20, key=0, flooded=False, flood_frac=0.0, highway="residential", name="Local 1")
        MG.add_edge(10, 20, key=1, flooded=False, flood_frac=0.0, highway="residential", name="Local 2")
        MG.add_edge(20, 10, key=0, flooded=False, flood_frac=0.0, highway="residential", name="Local 1 Rev")

        # Flooded bridge connecting 20 and 30
        MG.add_edge(20, 30, key=0, flooded=True, flood_frac=0.95, bridge="yes", highway="primary", name="Delta Bridge")
        MG.add_edge(30, 20, key=0, flooded=True, flood_frac=0.95, bridge="yes", highway="primary", name="Delta Bridge Rev")

        # Dry road keeping Hub 30 accessible to node 40
        MG.add_edge(30, 40, key=0, flooded=False, flood_frac=0.0, highway="primary", name="Hospital Main")
        MG.add_edge(40, 30, key=0, flooded=False, flood_frac=0.0, highway="primary", name="Hospital Main Rev")

        clusters, isolated = identify_cut_off_settlements(MG, [30], min_nodes=1)
        self.assertEqual(isolated, {10, 20})
        self.assertEqual(len(clusters), 1)
        c = clusters[0]
        self.assertEqual(c["n_trapped_nodes"], 2)
        self.assertTrue(c["has_bridge_cut"])
        self.assertEqual(c["nearest_hub"], "Base Hospital")

    def test_priority_tiers_and_custom_weights(self):
        # Test clusters that exercise all priority tiers
        dummy_clusters = [
            # High pop, far away, bridge cut -> CRITICAL
            {
                "cluster_id": "Village_Cluster_01",
                "center_lon": 67.80,
                "center_lat": 26.70,
                "estimated_population": 3000,
                "trapped_nodes": 60,
                "dist_to_hub_km": 18.0,
                "nearest_hub": "Regional Hospital",
                "has_bridge_cut": True,
                "has_highway_cut": True,
                "num_cut_edges": 4,
            },
            # Moderate pop, moderate dist, highway cut -> HIGH
            {
                "cluster_id": "Village_Cluster_02",
                "center_lon": 67.82,
                "center_lat": 26.71,
                "estimated_population": 800,
                "trapped_nodes": 20,
                "dist_to_hub_km": 8.0,
                "nearest_hub": "Regional Hospital",
                "has_bridge_cut": False,
                "has_highway_cut": True,
                "num_cut_edges": 2,
            },
            # Small pop, close dist, minor cut -> MEDIUM
            {
                "cluster_id": "Village_Cluster_03",
                "center_lon": 67.83,
                "center_lat": 26.71,
                "estimated_population": 250,
                "trapped_nodes": 8,
                "dist_to_hub_km": 3.0,
                "nearest_hub": "Regional Hospital",
                "has_bridge_cut": False,
                "has_highway_cut": False,
                "num_cut_edges": 1,
            },
            # Very small, very close -> LOW
            {
                "cluster_id": "Village_Cluster_04",
                "center_lon": 67.84,
                "center_lat": 26.72,
                "estimated_population": 20,
                "trapped_nodes": 2,
                "dist_to_hub_km": 0.5,
                "nearest_hub": "Regional Hospital",
                "has_bridge_cut": False,
                "has_highway_cut": False,
                "num_cut_edges": 1,
            },
        ]

        # Test default weights
        ranked = calculate_priority(dummy_clusters)
        self.assertEqual(len(ranked), 4)
        tiers = [r["priority_tier"] for r in ranked]
        self.assertIn("CRITICAL", tiers)
        self.assertIn("LOW", tiers)

        # Test custom weights
        custom_weights = {
            "population": 0.70,
            "distance": 0.10,
            "trapped_nodes": 0.10,
            "vulnerability": 0.10,
        }
        ranked_custom = calculate_priority(dummy_clusters, weights=custom_weights)
        self.assertEqual(ranked_custom[0]["cluster_id"], "Village_Cluster_01")

        # Test manifest recommendations
        manifest = generate_rescue_manifest(ranked)
        actions = manifest["Recommended_Action"].tolist()
        self.assertTrue(any("Airdrop" in a or "Boat" in a for a in actions))
        self.assertTrue(any("detour" in a.lower() for a in actions))

    def test_empty_network_edge_cases(self):
        # Empty graph or no isolated nodes
        empty_G = nx.Graph()
        clusters, isolated = identify_cut_off_settlements(empty_G, [])
        self.assertEqual(clusters, [])
        self.assertEqual(isolated, set())

        empty_manifest = generate_rescue_manifest([])
        self.assertTrue(empty_manifest.empty)
        csv_p = os.path.join(self.tmp_dir, "empty_manifest.csv")
        json_p = os.path.join(self.tmp_dir, "empty_manifest.geojson")
        export_rescue_plan(empty_manifest, csv_p, json_p)
        self.assertTrue(os.path.exists(csv_p))
        self.assertTrue(os.path.exists(json_p))


if __name__ == "__main__":
    unittest.main()

