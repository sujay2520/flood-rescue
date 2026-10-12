"""
src/cutoff_analysis.py

Production-grade module for Track B: Mapping Flood Damage from Space (Nepal Trishuli Disaster).
Performs dual-run Dijkstra reachability analysis, cut-off classification, and isolated settlement clustering.

CRITICAL TRACK B RULE: No WorldPop or synthetic population! Reports building counts per isolated cluster.

Author: Track B Engineering Team
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cutoff_analysis")

DELAY_RATIO_THRESHOLD = 2.0
DELAY_MIN_THRESHOLD = 30.0  # minutes


@dataclass
class IsolatedCluster:
    """Represents an isolated settlement cluster."""
    cluster_id: int
    name: str
    node_ids: List[int]
    n_buildings: int
    bottlenecks: List[Dict[str, Any]]
    suggested_response: str
    geometry: Union[Point, Polygon]


def validate_graph(G: nx.Graph) -> None:
    """Ensure graph has required attributes."""
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes")

    for node, data in G.nodes(data=True):
        if 'x' not in data or 'y' not in data:
            raise ValueError(f"Node {node} missing required coordinates")
        data.setdefault('is_facility', False)
        data.setdefault('n_buildings', 0)
        data.setdefault('name', f"node_{node}")

    for u, v, data in G.edges(data=True):
        data.setdefault('status', 'not_affected')
        data.setdefault('length_m', 100.0)
        data.setdefault('is_bridge', False)


def get_facility_nodes(G: nx.Graph) -> List[int]:
    """Find all medical facility / hospital destination nodes."""
    return [n for n, d in G.nodes(data=True) if d.get('is_facility', False)]


def run_dual_dijkstra(G: nx.Graph) -> Tuple[Dict[int, float], Dict[int, float], List[Tuple[int, int]]]:
    """
    Dual-Run Dijkstra:
    Run 1 (Pre-event): Shortest distance in intact network to nearest facility.
    Run 2 (Post-event): Shortest distance in damaged network (damaged edges removed).
    """
    validate_graph(G)
    facilities = get_facility_nodes(G)

    # If no explicit facility nodes, use lowest-elevation southern exit node as regional hub
    if not facilities:
        lowest_node = min(G.nodes(), key=lambda n: G.nodes[n]['y'])
        facilities = [lowest_node]
        G.nodes[lowest_node]['is_facility'] = True
        G.nodes[lowest_node]['facility_name'] = "Southern Valley Exit / Bidur Hub"
        logger.warning(f"No hospital nodes marked; defaulted to southern hub node {lowest_node}")

    # Run 1: Pre-event intact
    pre_distances = {}
    for f in facilities:
        dists = nx.single_source_dijkstra_path_length(G, f, weight='length_m')
        for node, d in dists.items():
            if node not in pre_distances or d < pre_distances[node]:
                pre_distances[node] = d

    # Run 2: Post-event damaged graph
    G_damaged = G.copy()
    damaged_edges = []
    for u, v, data in G.edges(data=True):
        if data.get('status') == 'affected':
            G_damaged.remove_edge(u, v)
            damaged_edges.append((u, v))

    post_distances = {}
    for f in facilities:
        try:
            dists = nx.single_source_dijkstra_path_length(G_damaged, f, weight='length_m')
            for node, d in dists.items():
                if node not in post_distances or d < post_distances[node]:
                    post_distances[node] = d
        except Exception:
            pass

    logger.info(f"Dual Dijkstra complete: {len(damaged_edges)} damaged edges removed")
    return pre_distances, post_distances, damaged_edges


def classify_all_nodes(
    G: nx.Graph,
    pre_distances: Dict[int, float],
    post_distances: Dict[int, float],
) -> Dict[int, Dict[str, Any]]:
    """Classify each node into 'cut_off', 'severely_delayed', or 'accessible'."""
    classifications = {}

    for node in G.nodes():
        pre_d = pre_distances.get(node)
        post_d = post_distances.get(node)

        if pre_d is None:
            status = "accessible"
            delay_ratio = 1.0
        elif post_d is None:
            status = "cut_off"
            delay_ratio = float('inf')
        else:
            delay_ratio = post_d / max(pre_d, 1.0)
            delay_km = (post_d - pre_d) / 1000.0
            if delay_ratio >= DELAY_RATIO_THRESHOLD or delay_km >= 15.0:
                status = "severely_delayed"
            else:
                status = "accessible"

        classifications[node] = {
            "status": status,
            "pre_dist_km": (pre_d / 1000.0) if pre_d is not None else None,
            "post_dist_km": (post_d / 1000.0) if post_d is not None else None,
            "delay_ratio": delay_ratio if delay_ratio != float('inf') else 999.0,
        }

    return classifications


def find_severed_bottlenecks(
    G: nx.Graph,
    cluster_nodes: List[int],
    damaged_edges: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    """Find severed bridges or roads that isolate this settlement cluster."""
    cluster_set = set(cluster_nodes)
    bottlenecks = []

    for u, v in damaged_edges:
        u_in = u in cluster_set
        v_in = v in cluster_set
        if u_in != v_in:
            edge_data = G[u][v]
            bottlenecks.append({
                "edge": (u, v),
                "highway": edge_data.get("highway", "unclassified"),
                "is_bridge": edge_data.get("is_bridge", False),
                "length_m": edge_data.get("length_m", 0),
            })
    return bottlenecks


def generate_suggested_response(bottlenecks: List[Dict[str, Any]], n_buildings: int) -> str:
    """Rule-based suggested humanitarian response."""
    if not bottlenecks:
        return "Local road blockages - ground reconnaissance recommended"

    has_bridge = any(b.get("is_bridge") for b in bottlenecks)
    if has_bridge:
        return "Airdrop / Foot trail access only (Bridge severed by GLOF torrent)"
    return "Heavy equipment road clearance required (Highway corridor blocked)"


def cluster_isolated_settlements(
    G: nx.Graph,
    classifications: Dict[int, Dict[str, Any]],
    damaged_edges: List[Tuple[int, int]],
) -> List[IsolatedCluster]:
    """Group cut_off nodes into isolated settlement clusters."""
    cutoff_nodes = [n for n, c in classifications.items() if c["status"] == "cut_off"]
    if not cutoff_nodes:
        return []

    cutoff_subgraph = G.subgraph(cutoff_nodes).copy()
    components = list(nx.connected_components(cutoff_subgraph))

    clusters = []
    for idx, comp in enumerate(components, 1):
        comp_nodes = list(comp)
        n_buildings = sum(G.nodes[n].get("n_buildings", 0) for n in comp_nodes)

        # Centroid
        xs = [G.nodes[n]["x"] for n in comp_nodes]
        ys = [G.nodes[n]["y"] for n in comp_nodes]
        centroid = Point(np.mean(xs), np.mean(ys))

        # Settlement name: most populous node or nearest named node
        named_nodes = [n for n in comp_nodes if G.nodes[n].get("name") and not G.nodes[n]["name"].startswith("node_")]
        if named_nodes:
            best_node = max(named_nodes, key=lambda n: G.nodes[n].get("n_buildings", 0))
            cluster_name = G.nodes[best_node]["name"]
        else:
            cluster_name = f"Settlement Cluster #{idx}"

        bottlenecks = find_severed_bottlenecks(G, comp_nodes, damaged_edges)
        response = generate_suggested_response(bottlenecks, n_buildings)

        # Polygon or buffered point
        if len(comp_nodes) >= 3:
            pts = MultiPoint([Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in comp_nodes])
            geom = pts.convex_hull.buffer(50.0)
        else:
            geom = centroid.buffer(100.0)

        clusters.append(IsolatedCluster(
            cluster_id=idx,
            name=cluster_name,
            node_ids=comp_nodes,
            n_buildings=n_buildings,
            bottlenecks=bottlenecks,
            suggested_response=response,
            geometry=geom,
        ))

    return clusters


def nodes_to_geodataframe(G: nx.Graph, classifications: Dict[int, Dict[str, Any]], crs: str = "EPSG:32645") -> gpd.GeoDataFrame:
    """Convert classified nodes to GeoDataFrame."""
    records = []
    for n, data in G.nodes(data=True):
        c = classifications.get(n, {"status": "accessible", "pre_dist_km": 0, "post_dist_km": 0, "delay_ratio": 1.0})
        records.append({
            "node_id": n,
            "status": c["status"],
            "pre_dist_km": c["pre_dist_km"],
            "post_dist_km": c["post_dist_km"],
            "delay_ratio": c["delay_ratio"],
            "n_buildings": data.get("n_buildings", 0),
            "is_facility": data.get("is_facility", False),
            "facility_name": data.get("facility_name"),
            "geometry": Point(data["x"], data["y"]),
        })
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def clusters_to_geodataframe(clusters: List[IsolatedCluster], crs: str = "EPSG:32645") -> gpd.GeoDataFrame:
    """Convert IsolatedCluster list to GeoDataFrame."""
    if not clusters:
        return gpd.GeoDataFrame(columns=["cluster_id", "name", "n_buildings", "n_nodes", "suggested_response", "geometry"], crs=crs)

    records = []
    for c in clusters:
        records.append({
            "cluster_id": c.cluster_id,
            "name": c.name,
            "n_buildings": c.n_buildings,
            "n_nodes": len(c.node_ids),
            "n_bottlenecks": len(c.bottlenecks),
            "suggested_response": c.suggested_response,
            "geometry": c.geometry,
        })
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


# =============================================================================
# High-Level API
# =============================================================================

def analyze_cutoffs(G: nx.Graph, out_dir: Union[str, Path] = "./outputs/cutoff") -> Dict[str, Any]:
    """
    Run complete cut-off reachability analysis and settlement clustering.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Executing dual-run Dijkstra cut-off reachability analysis...")

    # 1. Dual Dijkstra
    pre_dists, post_dists, damaged_edges = run_dual_dijkstra(G)

    # 2. Node classifications
    classifications = classify_all_nodes(G, pre_dists, post_dists)

    # 3. Cluster isolated settlements
    clusters = cluster_isolated_settlements(G, classifications, damaged_edges)

    # 4. GeoDataFrames
    crs = G.graph.get("crs", "EPSG:32645")
    gdf_nodes = nodes_to_geodataframe(G, classifications, crs=crs)
    gdf_clusters = clusters_to_geodataframe(clusters, crs=crs)

    # 5. Save outputs
    nodes_out = out_dir / "node_reachability.geojson"
    clusters_out = out_dir / "isolated_clusters.geojson"
    summary_out = out_dir / "cutoff_summary.json"

    gdf_nodes.to_file(nodes_out, driver="GeoJSON")
    gdf_clusters.to_file(clusters_out, driver="GeoJSON")

    # 6. Summary metrics
    n_cutoff_nodes = sum(1 for c in classifications.values() if c["status"] == "cut_off")
    n_delayed_nodes = sum(1 for c in classifications.values() if c["status"] == "severely_delayed")
    n_accessible_nodes = sum(1 for c in classifications.values() if c["status"] == "accessible")
    total_cutoff_buildings = sum(c.n_buildings for c in clusters)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_nodes": G.number_of_nodes(),
        "n_cutoff_nodes": n_cutoff_nodes,
        "n_severely_delayed_nodes": n_delayed_nodes,
        "n_accessible_nodes": n_accessible_nodes,
        "n_cutoff_clusters": len(clusters),
        "total_cutoff_buildings": total_cutoff_buildings,
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "name": c.name,
                "n_buildings": c.n_buildings,
                "n_nodes": len(c.node_ids),
                "suggested_response": c.suggested_response,
            }
            for c in clusters
        ],
    }

    summary_out.write_text(json.dumps(summary, indent=2))
    logger.info(f"Cut-off analysis finished: {len(clusters)} isolated clusters found, {total_cutoff_buildings} buildings cut off")

    return {
        "nodes_gdf": gdf_nodes,
        "clusters_gdf": gdf_clusters,
        "clusters": clusters,
        "summary": summary,
        "files": {
            "nodes": nodes_out,
            "clusters": clusters_out,
            "summary": summary_out,
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut-Off Reachability Analysis")
    parser.add_argument("--out-dir", type=str, default="./outputs/cutoff")
    args = parser.parse_args()

    # Synthetic demo test
    from .osm_ingest import fetch_pre_event_osm
    from .graph_builder import build_network
    osm = fetch_pre_event_osm([85.15, 27.85, 85.45, 28.25], force_offline=True)
    G = build_network(osm["roads"], osm["bridges"], osm["facilities"], osm["buildings"])

    # Mark some bridge as affected
    for u, v, d in G.edges(data=True):
        if d.get("is_bridge"):
            d["status"] = "affected"
            break

    res = analyze_cutoffs(G, out_dir=args.out_dir)
    print("Cut-Off Summary:")
    for k, v in res["summary"].items():
        if k != "clusters":
            print(f"  {k:25s}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
