"""
src/isolation.py - Settlement Cut-Off Detection, Clustering & GeoJSON Export.

Identifies road network nodes cut off from emergency relief hubs (hospitals/hubs)
due to flood inundation, clusters them into distinct isolated communities,
and calculates physical isolation metrics:
  - Cluster ID (e.g., 'Village_Cluster_01')
  - Center coordinates (lon, lat)
  - Number of trapped road nodes
  - Estimated trapped population
  - Radius of settlement (meters)
  - Shortest geodesic / straight-line distance to nearest functioning hospital/hub
  - Name of nearest hub
  - Critical access obstacles (e.g., collapsed bridge or flooded highway)

Exports isolated communities as GeoJSON FeatureCollection of Polygons (buffered footprint)
or Points for tactical map visualization.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
from pyproj import CRS, Geod, Transformer
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon, mapping
from shapely.ops import transform as shp_transform, unary_union

from src.core_logic import cluster_isolated, find_isolated, snap_to_nodes

logger = logging.getLogger(__name__)


def _utm_crs(lon: float, lat: float) -> CRS:
    """Computes the appropriate WGS84 UTM projection CRS for a lon/lat coordinate."""
    zone = int((lon + 180) // 6) + 1
    epsg_code = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg_code)


def _format_road_name(name_val: Any, default: str = "Access Road") -> str:
    """Helper to convert OSM road name (str, list, or None) to clean str."""
    if not name_val:
        return default
    if isinstance(name_val, (list, tuple)):
        clean = [str(x) for x in name_val if x]
        return ", ".join(clean) if clean else default
    return str(name_val)


def _format_highway_type(hw_val: Any, default: str = "road") -> str:
    """Helper to convert OSM highway type (str or list) to clean str."""
    if not hw_val:
        return default
    if isinstance(hw_val, (list, tuple)):
        clean = [str(x) for x in hw_val if x]
        return clean[0] if clean else default
    return str(hw_val)


def _is_bridge_edge(edge_data: Dict[str, Any]) -> bool:
    """Determines whether an edge attribute represents a bridge or damaged bridge."""
    if edge_data.get("bridge_damaged", False):
        return True
    if edge_data.get("is_bridge", False):
        return True
    bridge = edge_data.get("bridge")
    if bridge in (True, "yes", "true", "True", "1", 1, "viaduct", "aqueduct"):
        return True
    if edge_data.get("man_made") == "bridge":
        return True
    return False


def _iter_node_incident_edges(G: nx.Graph, u: Any):
    """Safely yields (u, v, key, data) for incident edges across Graph and MultiGraph."""
    if G.is_multigraph():
        for _, v, k, d in G.edges(u, keys=True, data=True):
            yield u, v, k, d
    else:
        for _, v, d in G.edges(u, data=True):
            yield u, v, None, d


def _detect_cluster_obstacles(
    G: nx.Graph,
    cluster_nodes: Iterable[Any],
) -> Tuple[List[str], str, bool, bool, int]:
    """
    Identifies specific road cuts, submerged bridges, and highway blockages
    that severed this cluster from the reachable network.

    Returns:
        (obstacles_list, summary_str, has_bridge_cut, has_highway_cut, num_cut_edges)
    """
    comp_set = set(cluster_nodes)
    boundary_cuts: List[Dict[str, Any]] = []
    internal_cuts: List[Dict[str, Any]] = []

    # Iterate incident edges to cluster nodes
    for u in comp_set:
        for _, v, k, d in _iter_node_incident_edges(G, u):
            if not d.get("flooded", False):
                continue

            info = {
                "u": u,
                "v": v,
                "is_bridge": _is_bridge_edge(d),
                "highway": _format_highway_type(d.get("highway")),
                "name": _format_road_name(d.get("name")),
                "flood_frac": float(d.get("flood_frac", 1.0)),
                "is_boundary": (v not in comp_set),
            }

            if info["is_boundary"]:
                boundary_cuts.append(info)
            else:
                internal_cuts.append(info)

    unique_obstacles = []
    seen_keys = set()
    has_bridge = False
    has_highway = False

    # Check boundary cuts first (the primary severance), then internal cuts
    candidate_cuts = boundary_cuts if boundary_cuts else internal_cuts
    # Sort candidates so bridges and major roads come first
    candidate_cuts.sort(
        key=lambda x: (
            1 if x["is_bridge"] else 0,
            1 if x["highway"] in ("motorway", "trunk", "primary", "secondary") else 0,
            x["flood_frac"],
        ),
        reverse=True,
    )

    for c in candidate_cuts:
        is_br = c["is_bridge"]
        hw = c["highway"]
        nm = c["name"]
        frac = c["flood_frac"]
        frac_pct = int(round(frac * 100)) if not math.isnan(frac) else 100

        if is_br:
            has_bridge = True
            desc = f"Submerged bridge on {nm} ({hw}, {frac_pct}% flooded)"
        elif hw in ("motorway", "trunk", "primary", "secondary"):
            has_highway = True
            desc = f"Flooded {hw} artery on {nm} ({frac_pct}% flooded)"
        else:
            desc = f"Flooded {hw} access road on {nm} ({frac_pct}% flooded)"

        dedup_key = (is_br, hw, nm)
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            unique_obstacles.append(desc)

    total_cut_edges = len(boundary_cuts) + len(internal_cuts)
    if unique_obstacles:
        summary_str = "; ".join(unique_obstacles[:3])
    elif total_cut_edges > 0:
        summary_str = f"{total_cut_edges} flooded road access segments severed"
    else:
        summary_str = "Downstream regional road inundation (cut off at network choke point)"

    return unique_obstacles, summary_str, has_bridge, has_highway, total_cut_edges


def identify_cut_off_settlements(
    G: nx.Graph,
    hub_nodes: Union[List[Any], Set[Any], Dict[Any, Any]],
    pop_array: Optional[np.ndarray] = None,
    pop_transform: Optional[Any] = None,
    pop_crs: str = "EPSG:4326",
    graph_crs: str = "EPSG:4326",
    buffer_m: float = 300.0,
    min_nodes: int = 1,
    return_isolated: bool = True,
) -> Union[Tuple[List[Dict[str, Any]], Set[Any]], List[Dict[str, Any]]]:
    """
    Identifies newly cut-off road nodes and groups them into isolated settlement clusters.

    Calculates for each cluster:
      - Cluster ID (e.g., 'Village_Cluster_01')
      - Center coordinates (lon, lat)
      - Number of trapped road nodes
      - Estimated trapped population
      - Radius of settlement (meters)
      - Shortest geodesic / straight-line distance to nearest functioning hospital/hub
      - Name and ID of nearest hub
      - Critical access obstacles (e.g., collapsed bridge or flooded highway)

    Args:
        G: NetworkX road graph with 'flooded' edge tags (from core_logic.tag_edges).
        hub_nodes: Node IDs of functioning hospitals/relief hubs (list, set, or dict).
        pop_array: Optional 2D numpy array of population density raster.
        pop_transform: Affine transform for the population raster.
        pop_crs: Coordinate reference system of the population raster.
        graph_crs: Coordinate reference system of road graph nodes (default EPSG:4326).
        buffer_m: Search / union footprint buffer in meters around cluster nodes.
        min_nodes: Minimum road nodes to consider as a settlement cluster.
        return_isolated: If True, returns (clusters, isolated_nodes) tuple.
                         If False, returns clusters list.

    Returns:
        Tuple[List[Dict[str, Any]], Set[Any]] if return_isolated=True,
        else List[Dict[str, Any]].
    """
    # 1. Resolve hub nodes and names
    if isinstance(hub_nodes, dict):
        hub_dict = hub_nodes
        hub_id_list = list(hub_nodes.keys())
    elif isinstance(hub_nodes, (list, tuple, set)):
        hub_id_list = list(hub_nodes)
        hub_dict = {}
        for h in hub_id_list:
            if h in G.nodes:
                node_data = G.nodes[h]
                h_name = (
                    node_data.get("name")
                    or node_data.get("hub_name")
                    or node_data.get("amenity")
                    or node_data.get("healthcare")
                    or f"Relief_Hub_{h}"
                )
                hub_dict[h] = str(h_name)
            else:
                hub_dict[h] = f"Relief_Hub_{h}"
    else:
        hub_id_list = []
        hub_dict = {}

    # 2. Call core_logic.find_isolated to identify newly cut-off nodes
    isolated_nodes, after_graph = find_isolated(G, hub_id_list)

    if not isolated_nodes:
        logger.info("No isolated nodes detected. Entire road network maintains hub access.")
        return ([], isolated_nodes) if return_isolated else []

    # 3. Call core_logic.cluster_isolated to group connected components
    raw_clusters = cluster_isolated(
        G,
        isolated_nodes,
        after_graph,
        pop_array=pop_array,
        pop_transform=pop_transform,
        pop_crs=pop_crs,
        graph_crs=graph_crs,
        buffer_m=buffer_m,
        min_nodes=min_nodes,
    )

    # 4. Determine functioning hubs (present in after_graph with active connections)
    functioning_hubs = [
        h for h in hub_id_list
        if h in after_graph and (after_graph.degree(h) > 0 or G.degree(h) == 0)
    ]
    if not functioning_hubs:
        # Fallback to all hubs present in G if all hubs became isolated
        functioning_hubs = [h for h in hub_id_list if h in G.nodes]

    # Pre-extract hub coordinates for fast geodesic distance calculation
    geod = Geod(ellps="WGS84")
    hub_coords = []
    for h in functioning_hubs:
        if h in G.nodes:
            hub_coords.append((h, hub_dict.get(h, f"Hub_{h}"), G.nodes[h]["x"], G.nodes[h]["y"]))

    processed_clusters: List[Dict[str, Any]] = []

    # 5. Calculate detailed metrics for each cluster
    for idx, raw_c in enumerate(raw_clusters):
        n_nodes = int(raw_c["n_nodes"])
        cluster_id = f"Isolated_Junction_{idx + 1:02d}" if n_nodes == 1 else f"Village_Cluster_{idx + 1:02d}"
        comp = raw_c["nodes"]
        c_lon = float(raw_c["lon"])
        c_lat = float(raw_c["lat"])
        n_nodes = int(raw_c["n_nodes"])

        # Population
        pop_val = raw_c.get("population")
        if pop_val is None or math.isnan(pop_val):
            # Fallback realistic estimation if pop raster was not provided
            est_population = float(max(n_nodes * 75, 50))
        else:
            est_population = float(pop_val)

        # Settlement Radius (meters): maximum geodesic distance from centroid to any node
        node_xs = [G.nodes[n]["x"] for n in comp]
        node_ys = [G.nodes[n]["y"] for n in comp]
        if len(comp) > 1:
            lons1 = [c_lon] * len(comp)
            lats1 = [c_lat] * len(comp)
            _, _, node_dists = geod.inv(lons1, lats1, node_xs, node_ys)
            max_node_dist = float(np.max(node_dists))
            radius_m = round(max(max_node_dist, 50.0), 1)
        else:
            radius_m = 50.0  # Baseline single intersection radius

        # Distance to nearest functioning hospital/hub
        nearest_hub_name = "None Available"
        nearest_hub_id = None
        min_hub_dist_m = float("inf")

        if hub_coords:
            c_lons = [c_lon] * len(hub_coords)
            c_lats = [c_lat] * len(hub_coords)
            h_lons = [hc[2] for hc in hub_coords]
            h_lats = [hc[3] for hc in hub_coords]
            _, _, hub_dists = geod.inv(c_lons, c_lats, h_lons, h_lats)
            min_idx = int(np.argmin(hub_dists))
            min_hub_dist_m = float(hub_dists[min_idx])
            nearest_hub_id = hub_coords[min_idx][0]
            nearest_hub_name = hub_coords[min_idx][1]

        dist_to_hub_km = round(min_hub_dist_m / 1000.0, 2) if min_hub_dist_m != float("inf") else 999.9

        # Critical access obstacles
        (
            obstacles_list,
            obstacles_summary,
            has_bridge_cut,
            has_highway_cut,
            num_cut_edges,
        ) = _detect_cluster_obstacles(G, comp)

        # Polygon footprint geometry (buffered footprint in EPSG:4326)
        utm = _utm_crs(c_lon, c_lat)
        to_utm = Transformer.from_crs(graph_crs, utm, always_xy=True).transform
        to_wgs = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform

        pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in comp]
        footprint_utm = unary_union([shp_transform(to_utm, p).buffer(buffer_m) for p in pts])
        footprint_wgs = shp_transform(to_wgs, footprint_utm)

        cluster_record = {
            "cluster_id": cluster_id,
            "center_lon": round(c_lon, 6),
            "center_lat": round(c_lat, 6),
            "center_coords": (round(c_lon, 6), round(c_lat, 6)),
            "n_trapped_nodes": n_nodes,
            "trapped_nodes": n_nodes,
            "estimated_population": round(est_population, 1),
            "population": round(est_population, 1),
            "radius_m": radius_m,
            "shortest_distance_to_hub_m": round(min_hub_dist_m, 1) if min_hub_dist_m != float("inf") else -1.0,
            "dist_to_hub_km": dist_to_hub_km,
            "distance_to_nearest_hub_km": dist_to_hub_km,
            "nearest_hub": nearest_hub_name,
            "nearest_hub_id": nearest_hub_id,
            "critical_access_obstacles": obstacles_summary,
            "obstacles": obstacles_list,
            "has_bridge_cut": has_bridge_cut,
            "has_highway_cut": has_highway_cut,
            "num_cut_edges": num_cut_edges,
            "nodes": sorted(comp),
            "geometry": footprint_wgs,
            "centroid_geometry": Point(c_lon, c_lat),
        }
        processed_clusters.append(cluster_record)

    logger.info(
        f"Identified {len(processed_clusters)} isolated settlements with {len(isolated_nodes)} cut-off nodes."
    )

    if return_isolated:
        return processed_clusters, isolated_nodes
    return processed_clusters


def isolation_to_geojson(
    clusters: Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Any], gpd.GeoDataFrame, Any],
    output_path: Optional[Union[str, Path]] = None,
    as_polygons: bool = True,
) -> Dict[str, Any]:
    """
    Exports isolated clusters as a GeoJSON FeatureCollection with rich metadata.

    Supports exporting either buffered settlement footprints (Polygons) or
    centroids (Points), with properties for map visualization.

    Args:
        clusters: List of cluster dicts (or tuple returned by identify_cut_off_settlements).
        output_path: Optional file path to write GeoJSON.
        as_polygons: If True and footprint geometry is present, exports Polygons.
                     If False, exports Point centroids.

    Returns:
        GeoJSON FeatureCollection dictionary.
    """
    # Auto-unwrap tuple if caller passed return value of identify_cut_off_settlements directly
    if isinstance(clusters, tuple) and len(clusters) == 2 and isinstance(clusters[0], list):
        clusters = clusters[0]

    features_list: List[Dict[str, Any]] = []

    if isinstance(clusters, gpd.GeoDataFrame):
        geojson_str = clusters.to_json()
        geojson_dict = json.loads(geojson_str)
        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(geojson_dict, f, indent=2)
        return geojson_dict

    for c in clusters:
        # Determine geometry
        geom = None
        if as_polygons and "geometry" in c and c["geometry"] is not None:
            geom = c["geometry"]
        elif "centroid_geometry" in c and c["centroid_geometry"] is not None:
            geom = c["centroid_geometry"]
        elif "center_lon" in c and "center_lat" in c:
            geom = Point(c["center_lon"], c["center_lat"])
        elif "lon" in c and "lat" in c:
            geom = Point(c["lon"], c["lat"])
        else:
            geom = Point(0.0, 0.0)

        # Build clean properties dictionary (exclude non-serializable or large objects)
        props = {
            "cluster_id": c.get("cluster_id", "Unknown"),
            "rank": c.get("rank", c.get("Rank", 1)),
            "Rank": c.get("Rank", c.get("rank", 1)),
            "estimated_population": c.get("estimated_population", c.get("population", 0)),
            "population": c.get("population", c.get("estimated_population", 0)),
            "trapped_nodes": c.get("trapped_nodes", c.get("n_trapped_nodes", 0)),
            "radius_m": c.get("radius_m", 0.0),
            "dist_to_hub_km": c.get("dist_to_hub_km", c.get("distance_to_nearest_hub_km", 0.0)),
            "nearest_hub": c.get("nearest_hub", "Unknown"),
            "critical_access_obstacles": c.get("critical_access_obstacles", "None"),
            "has_bridge_cut": bool(c.get("has_bridge_cut", False)),
            "has_highway_cut": bool(c.get("has_highway_cut", False)),
            "center_lon": c.get("center_lon", c.get("lon", 0.0)),
            "center_lat": c.get("center_lat", c.get("lat", 0.0)),
        }

        # Include priority fields if already calculated
        if "priority_score" in c:
            props["priority_score"] = c["priority_score"]
        if "priority_tier" in c:
            props["priority_tier"] = c["priority_tier"]
        if "priority_color" in c:
            props["priority_color"] = c["priority_color"]
        if "recommended_action" in c:
            props["recommended_action"] = c["recommended_action"]

        feature = {
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": props,
        }
        features_list.append(feature)

    geojson_collection = {
        "type": "FeatureCollection",
        "features": features_list,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(geojson_collection, f, indent=2)
        logger.info(f"Saved isolation GeoJSON ({len(features_list)} features) to: {out_p}")

    return geojson_collection
