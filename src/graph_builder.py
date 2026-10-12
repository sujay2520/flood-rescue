"""
src/graph_builder.py

Production-grade graph construction module for Track B: Mapping Flood Damage from Space.
Builds a planar road network with mandatory bridge nodalization, facility connections,
and building associations. All inputs/outputs in EPSG:32645 (UTM 45N).

Author: Track B Engineering Team
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, split
from shapely.strtree import STRtree

from .osm_ingest import fetch_pre_event_osm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("graph_builder")

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class HighwaySpeeds:
    """Default free-flow speeds (km/h) by highway classification."""
    motorway: float = 100.0
    motorway_link: float = 60.0
    trunk: float = 80.0
    trunk_link: float = 50.0
    primary: float = 60.0
    primary_link: float = 40.0
    secondary: float = 50.0
    secondary_link: float = 35.0
    tertiary: float = 40.0
    tertiary_link: float = 30.0
    unclassified: float = 30.0
    residential: float = 25.0
    service: float = 15.0
    track: float = 15.0
    default: float = 30.0


HIGHWAY_SPEEDS = HighwaySpeeds()
FACILITY_CONNECTION_MAX_DIST_M = 500.0
BUILDING_ASSOCIATION_MAX_DIST_M = 300.0
BRIDGE_SPLIT_TOLERANCE_M = 1.0
NODE_MERGE_TOLERANCE_M = 1.0


def _get_speed_kmh(highway: Any) -> float:
    if isinstance(highway, list):
        highway = highway[0] if highway else "default"
    if not isinstance(highway, str):
        return HIGHWAY_SPEEDS.default
    return getattr(HIGHWAY_SPEEDS, highway, HIGHWAY_SPEEDS.default)


def _compute_free_flow_time_sec(length_m: float, speed_kmh: float) -> float:
    if speed_kmh <= 0:
        return float('inf')
    return (length_m / 1000.0) / speed_kmh * 3600.0


def _ensure_linestring(geom: Any) -> LineString:
    if geom is None or geom.is_empty:
        raise ValueError("Empty geometry")
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, LineString):
            return merged
        return max(merged.geoms, key=lambda g: g.length)
    raise TypeError(f"Expected LineString or MultiLineString, got {type(geom)}")


def _split_line_at_points(line: LineString, points: List[Point], tolerance: float = 0.01) -> List[LineString]:
    if not points:
        return [line]
    from shapely.geometry import MultiPoint
    splitter = MultiPoint(points)
    try:
        result = split(line, splitter)
        segments = [g for g in result.geoms if isinstance(g, LineString) and not g.is_empty]
        return segments if segments else [line]
    except Exception as e:
        return [line]


def _snap_points_to_line(points: List[Point], line: LineString, tolerance: float) -> List[Point]:
    snapped = []
    for pt in points:
        proj_dist = line.project(pt)
        snapped_pt = line.interpolate(proj_dist)
        if pt.distance(snapped_pt) <= tolerance:
            snapped.append(snapped_pt)
        else:
            snapped.append(pt)
    return snapped


# -----------------------------------------------------------------------------
# Road Network Builder
# -----------------------------------------------------------------------------

class RoadNetworkBuilder:
    """
    Builds a planar, topologically correct road network graph from OSM road data
    with mandatory bridge nodalization.
    """

    def __init__(
        self,
        node_merge_tolerance: float = NODE_MERGE_TOLERANCE_M,
        bridge_split_tolerance: float = BRIDGE_SPLIT_TOLERANCE_M,
    ):
        self.node_merge_tolerance = node_merge_tolerance
        self.bridge_split_tolerance = bridge_split_tolerance
        self._node_counter = 0
        self._edge_counter = 0
        self._coord_to_node: Dict[Tuple[float, float], int] = {}
        self._node_coords: Dict[int, Tuple[float, float]] = {}
        self.G = nx.Graph()
        self.G.graph['crs'] = "EPSG:32645"

    def _get_or_create_node(self, x: float, y: float) -> int:
        grid_x = round(x / self.node_merge_tolerance) * self.node_merge_tolerance
        grid_y = round(y / self.node_merge_tolerance) * self.node_merge_tolerance
        key = (grid_x, grid_y)

        if key in self._coord_to_node:
            return self._coord_to_node[key]

        node_id = self._node_counter
        self._node_counter += 1
        self._coord_to_node[key] = node_id
        self._node_coords[node_id] = (x, y)

        self.G.add_node(
            node_id,
            x=x,
            y=y,
            geometry=Point(x, y),
            n_buildings=0,
            is_facility=False,
            facility_type=None,
            facility_name=None,
        )
        return node_id

    def _add_edge(
        self,
        u: int,
        v: int,
        osm_id: Any,
        highway: Any,
        is_bridge: bool,
        geometry: LineString,
        length_m: Optional[float] = None,
    ) -> int:
        if length_m is None:
            length_m = float(geometry.length)

        speed_kmh = _get_speed_kmh(highway)
        free_flow_time = _compute_free_flow_time_sec(length_m, speed_kmh)

        edge_id = self._edge_counter
        self._edge_counter += 1

        self.G.add_edge(
            u, v,
            edge_id=edge_id,
            osm_id=osm_id,
            length_m=length_m,
            highway=highway if isinstance(highway, str) else "unclassified",
            is_bridge=bool(is_bridge),
            speed_kmh=speed_kmh,
            free_flow_time_sec=free_flow_time,
            geometry=geometry,
        )
        return edge_id

    def _process_road_geometry(
        self,
        geom: Any,
        osm_id: Any,
        highway: Any,
        is_bridge: bool,
        bridge_geoms: List[LineString],
    ) -> None:
        line = _ensure_linestring(geom)
        split_points: List[Point] = []

        if bridge_geoms:
            for bridge_geom in bridge_geoms:
                intersection = line.intersection(bridge_geom)
                if not intersection.is_empty:
                    if isinstance(intersection, Point):
                        split_points.append(intersection)
                    elif isinstance(intersection, (LineString, MultiLineString)):
                        if isinstance(intersection, LineString):
                            split_points.append(Point(intersection.coords[0]))
                            split_points.append(Point(intersection.coords[-1]))
                        else:
                            for part in intersection.geoms:
                                split_points.append(Point(part.coords[0]))
                                split_points.append(Point(part.coords[-1]))

        split_points = _snap_points_to_line(split_points, line, self.bridge_split_tolerance)
        segments = _split_line_at_points(line, split_points, self.bridge_split_tolerance)

        for seg in segments:
            if seg.length < 0.1:
                continue
            coords = list(seg.coords)
            u = self._get_or_create_node(coords[0][0], coords[0][1])
            v = self._get_or_create_node(coords[-1][0], coords[-1][1])
            if u != v:
                self._add_edge(u, v, osm_id, highway, is_bridge, seg)

    def build_from_roads_and_bridges(
        self,
        gdf_roads: gpd.GeoDataFrame,
        gdf_bridges: gpd.GeoDataFrame,
    ) -> nx.Graph:
        logger.info(f"Building road network from {len(gdf_roads)} roads and {len(gdf_bridges)} bridges")

        bridge_geoms = []
        if len(gdf_bridges) > 0:
            for _, row in gdf_bridges.iterrows():
                try:
                    bridge_geoms.append(_ensure_linestring(row.geometry))
                except Exception:
                    pass

        bridge_tree = STRtree(bridge_geoms) if bridge_geoms else None

        for idx, row in gdf_roads.iterrows():
            try:
                osm_id = row.get('osm_id', idx)
                highway = row.get('highway', 'unclassified')
                is_bridge = str(row.get('bridge', False)).lower() in ['true', 'yes', '1']

                intersecting_bridges = []
                if bridge_tree is not None:
                    road_geom = _ensure_linestring(row.geometry)
                    candidate_idx = bridge_tree.query(road_geom)
                    for bidx in candidate_idx:
                        if road_geom.intersects(bridge_geoms[bidx]):
                            intersecting_bridges.append(bridge_geoms[bidx])

                self._process_road_geometry(
                    row.geometry, osm_id, highway, is_bridge, intersecting_bridges
                )
            except Exception as e:
                logger.debug(f"Failed to process road {idx}: {e}")

        # Explicit bridge segments
        for idx, row in gdf_bridges.iterrows():
            try:
                bridge_line = _ensure_linestring(row.geometry)
                osm_id = row.get('osm_id', f"bridge_{idx}")
                highway = row.get('highway', 'bridge')
                coords = list(bridge_line.coords)
                u = self._get_or_create_node(coords[0][0], coords[0][1])
                v = self._get_or_create_node(coords[-1][0], coords[-1][1])
                if u != v:
                    self._add_edge(u, v, osm_id, highway, True, bridge_line)
            except Exception:
                pass

        logger.info(f"Road network graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G

    def connect_facilities(
        self,
        gdf_facilities: gpd.GeoDataFrame,
        max_distance: float = FACILITY_CONNECTION_MAX_DIST_M,
    ) -> nx.Graph:
        if len(gdf_facilities) == 0 or self.G.number_of_nodes() == 0:
            return self.G

        node_ids = list(self.G.nodes())
        node_coords = np.array([self._node_coords[nid] for nid in node_ids])
        tree = cKDTree(node_coords)

        connected = 0
        for idx, facility in gdf_facilities.iterrows():
            try:
                geom = facility.geometry
                pt = geom.centroid if isinstance(geom, (Polygon, MultiPolygon)) else geom
                dist, n_idx = tree.query([pt.x, pt.y], k=1)
                dist = float(dist)

                if dist <= max_distance:
                    nearest_node = node_ids[int(n_idx)]
                    self.G.nodes[nearest_node]['is_facility'] = True
                    self.G.nodes[nearest_node]['facility_type'] = facility.get('amenity', 'clinic')
                    self.G.nodes[nearest_node]['facility_name'] = facility.get('name', f"facility_{idx}")
                    connected += 1
            except Exception as e:
                logger.debug(f"Failed facility connection {idx}: {e}")

        logger.info(f"Connected {connected}/{len(gdf_facilities)} critical health facilities to network")
        return self.G

    def associate_buildings(
        self,
        gdf_buildings: gpd.GeoDataFrame,
        max_distance: float = BUILDING_ASSOCIATION_MAX_DIST_M,
    ) -> nx.Graph:
        if len(gdf_buildings) == 0 or self.G.number_of_nodes() == 0:
            return self.G

        centroids = gdf_buildings.geometry.centroid
        coords = np.column_stack((centroids.x.values, centroids.y.values))

        node_ids = list(self.G.nodes())
        node_coords = np.array([self._node_coords[n] for n in node_ids])
        tree = cKDTree(node_coords)

        distances, indices = tree.query(coords, k=1, distance_upper_bound=max_distance)
        valid = distances <= max_distance

        for idx in indices[valid]:
            node_id = node_ids[int(idx)]
            self.G.nodes[node_id]["n_buildings"] = self.G.nodes[node_id].get("n_buildings", 0) + 1

        total_bldgs = sum(d.get("n_buildings", 0) for _, d in self.G.nodes(data=True))
        logger.info(f"Associated {total_bldgs} buildings to network nodes")
        return self.G


# =============================================================================
# High-Level API Functions
# =============================================================================

def build_network(
    gdf_roads: gpd.GeoDataFrame,
    gdf_bridges: gpd.GeoDataFrame,
    gdf_facilities: gpd.GeoDataFrame,
    gdf_buildings: gpd.GeoDataFrame,
) -> nx.Graph:
    """Build complete network graph with bridges, facilities, and building counts."""
    builder = RoadNetworkBuilder()
    builder.build_from_roads_and_bridges(gdf_roads, gdf_bridges)
    builder.connect_facilities(gdf_facilities)
    builder.associate_buildings(gdf_buildings)
    return builder.G


def graph_to_geodataframes(G: nx.Graph) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert NetworkX graph to (nodes_gdf, edges_gdf) GeoDataFrames in EPSG:32645."""
    crs = G.graph.get("crs", "EPSG:32645")

    node_data = []
    for n, data in G.nodes(data=True):
        row = {"node_id": n, "geometry": Point(data["x"], data["y"])}
        row.update({k: v for k, v in data.items() if k not in ("x", "y", "geometry")})
        node_data.append(row)
    nodes_gdf = gpd.GeoDataFrame(node_data, crs=crs, geometry="geometry")

    edge_data = []
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom is None:
            u_pt = Point(G.nodes[u]["x"], G.nodes[u]["y"])
            v_pt = Point(G.nodes[v]["x"], G.nodes[v]["y"])
            geom = LineString([u_pt, v_pt])
        row = {"u": u, "v": v, "geometry": geom}
        row.update({k: v for k, v in data.items() if k != "geometry"})
        edge_data.append(row)
    edges_gdf = gpd.GeoDataFrame(edge_data, crs=crs, geometry="geometry")

    return nodes_gdf, edges_gdf


def snap_to_nearest_node(G: nx.Graph, x: float, y: float) -> Tuple[int, float]:
    """Return (nearest_node_id, distance_m) for a given coordinate in graph CRS."""
    node_ids = list(G.nodes())
    node_coords = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_ids])
    tree = cKDTree(node_coords)
    dist, idx = tree.query([x, y], k=1)
    return node_ids[int(idx)], float(dist)


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Graph Builder Pipeline (Track B Nepal)")
    parser.add_argument("--bbox", nargs=4, type=float, default=[85.15, 27.85, 85.45, 28.25], help="minx miny maxx maxy")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Graph Builder Pipeline - Nepal Trishuli")
    print("=" * 70)
    osm_layers = fetch_pre_event_osm(args.bbox, force_offline=args.offline)
    G = build_network(
        osm_layers["roads"],
        osm_layers["bridges"],
        osm_layers["facilities"],
        osm_layers["buildings"],
    )

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_bridges = sum(1 for _, _, d in G.edges(data=True) if d.get("is_bridge"))
    n_facilities = sum(1 for _, d in G.nodes(data=True) if d.get("is_facility"))
    total_buildings = sum(d.get("n_buildings", 0) for _, d in G.nodes(data=True))

    print(f"Nodes:              {n_nodes:,}")
    print(f"Edges:              {n_edges:,}")
    print(f"Bridge Edges:       {n_bridges:,}")
    print(f"Medical Facilities: {n_facilities:,}")
    print(f"Total Buildings:    {total_buildings:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
