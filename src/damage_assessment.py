"""
src/damage_assessment.py

Production-grade flood damage assessment module for Nepal Trishuli GLOF disaster.
Implements three-way road classification (affected, not_affected, not_assessable),
bridge approach severance detection (40m approach zone buffer), and building damage assessment.

Author: Track B Engineering Team
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("damage_assessment")


@dataclass
class DamageSummary:
    """Summary statistics for damage assessment."""
    total_road_km: float
    affected_road_km: float
    not_affected_road_km: float
    not_assessable_road_km: float
    total_bridges: int
    severed_bridges: int
    flooded_buildings_count: int
    total_buildings: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_crs(gdf: gpd.GeoDataFrame, target_crs: str = "EPSG:32645") -> gpd.GeoDataFrame:
    """Ensure GeoDataFrame is in target CRS."""
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    elif gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def _load_raster_mask_as_union(raster_path: Optional[Union[str, Path]], target_crs: str = "EPSG:32645"):
    """Load binary mask GeoTIFF and convert to unified Shapely geometry."""
    if not raster_path or not Path(raster_path).exists():
        return None
    try:
        from rasterio.features import shapes
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            results = [
                shape for shape, val in shapes(data, mask=(data == 1), transform=src.transform)
                if val == 1
            ]
            if not results:
                return None
            gdf = gpd.GeoDataFrame(geometry=results, crs=src.crs)
            gdf = ensure_crs(gdf, target_crs)
            return unary_union(gdf.geometry.values)
    except Exception as e:
        logger.warning(f"Failed to load raster mask {raster_path}: {e}")
        return None


def classify_road_segments(
    gdf_roads: gpd.GeoDataFrame,
    flood_gdf: gpd.GeoDataFrame,
    shadow_mask_path: Optional[Union[str, Path]] = None,
    layover_mask_path: Optional[Union[str, Path]] = None,
    min_flooded_length: float = 10.0,
) -> gpd.GeoDataFrame:
    """
    Classify each road segment into:
    - affected: intersects flood polygon OR flooded length >= min_flooded_length
    - not_assessable: within radar shadow or layover zones
    - not_affected: passable, outside flood, within observable zone
    """
    gdf = ensure_crs(gdf_roads.copy())
    flood_gdf = ensure_crs(flood_gdf.copy()) if not flood_gdf.empty else flood_gdf

    unobs_shadow = _load_raster_mask_as_union(shadow_mask_path)
    unobs_layover = _load_raster_mask_as_union(layover_mask_path)
    unobs_zones = [z for z in [unobs_shadow, unobs_layover] if z is not None]
    unobs_union = unary_union(unobs_zones) if unobs_zones else None

    flood_union = unary_union(flood_gdf.geometry.values) if len(flood_gdf) > 0 else None

    statuses = []
    flooded_lengths = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            statuses.append("not_assessable")
            flooded_lengths.append(0.0)
            continue

        # Check if in shadow/layover
        if unobs_union is not None and geom.intersects(unobs_union):
            obs_part = geom.difference(unobs_union)
            if obs_part.is_empty or obs_part.length < 1.0:
                statuses.append("not_assessable")
                flooded_lengths.append(0.0)
                continue
            assess_geom = obs_part
        else:
            assess_geom = geom

        flooded_len = 0.0
        is_affected = False

        if flood_union is not None and assess_geom.intersects(flood_union):
            inter = assess_geom.intersection(flood_union)
            if not inter.is_empty:
                flooded_len = float(inter.length)
                if flooded_len >= min_flooded_length:
                    is_affected = True

        if is_affected:
            statuses.append("affected")
        else:
            statuses.append("not_affected")

        flooded_lengths.append(flooded_len)

    gdf["status"] = statuses
    gdf["flooded_length_m"] = flooded_lengths
    gdf["length_m"] = gdf.geometry.length.astype(float)
    return gdf


def assess_bridge_approaches(
    gdf_bridges: gpd.GeoDataFrame,
    flood_gdf: gpd.GeoDataFrame,
    approach_buffer_m: float = 40.0,
) -> gpd.GeoDataFrame:
    """
    Assess bridge approach severance.
    Creates 40m buffer around each bridge endpoint.
    If flood intersects either approach buffer -> approach_severed = True.
    """
    gdf = ensure_crs(gdf_bridges.copy())
    flood_gdf = ensure_crs(flood_gdf.copy()) if not flood_gdf.empty else flood_gdf
    flood_union = unary_union(flood_gdf.geometry.values) if len(flood_gdf) > 0 else None

    approach_severed = []
    statuses = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            approach_severed.append(False)
            statuses.append("not_assessable")
            continue

        if isinstance(geom, MultiLineString):
            longest = max(geom.geoms, key=lambda x: x.length)
            coords = list(longest.coords)
        elif isinstance(geom, LineString):
            coords = list(geom.coords)
        else:
            coords = []

        if len(coords) < 2:
            approach_severed.append(False)
            statuses.append("not_assessable")
            continue

        start_pt = Point(coords[0])
        end_pt = Point(coords[-1])

        start_buf = start_pt.buffer(approach_buffer_m)
        end_buf = end_pt.buffer(approach_buffer_m)

        severed = False
        if flood_union is not None:
            if start_buf.intersects(flood_union) or end_buf.intersects(flood_union) or geom.intersects(flood_union):
                severed = True

        approach_severed.append(severed)
        statuses.append("affected" if severed else "not_affected")

    gdf["approach_severed"] = approach_severed
    gdf["status"] = statuses
    return gdf


def assess_building_damage(
    gdf_buildings: gpd.GeoDataFrame,
    flood_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assess building damage by spatial intersection with flood polygons.
    """
    gdf = ensure_crs(gdf_buildings.copy())
    flood_gdf = ensure_crs(flood_gdf.copy()) if not flood_gdf.empty else flood_gdf
    flood_union = unary_union(flood_gdf.geometry.values) if len(flood_gdf) > 0 else None

    is_flooded = []
    conf_list = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            is_flooded.append(False)
            conf_list.append(0.0)
            continue

        flooded = False
        conf = 0.0
        if flood_union is not None and geom.intersects(flood_union):
            flooded = True
            inter = geom.intersection(flood_union)
            conf = min(1.0, float(inter.area / max(geom.area, 1e-4)))

        is_flooded.append(flooded)
        conf_list.append(conf)

    gdf["is_flooded"] = is_flooded
    gdf["flood_confidence"] = conf_list
    return gdf


def compute_summary(
    gdf_roads: gpd.GeoDataFrame,
    gdf_bridges: gpd.GeoDataFrame,
    gdf_buildings: gpd.GeoDataFrame,
) -> DamageSummary:
    """Compute summary statistics from assessed layers."""
    total_road_m = float(gdf_roads["length_m"].sum()) if not gdf_roads.empty else 0.0
    affected_m = float(gdf_roads[gdf_roads["status"] == "affected"]["length_m"].sum()) if not gdf_roads.empty else 0.0
    not_affected_m = float(gdf_roads[gdf_roads["status"] == "not_affected"]["length_m"].sum()) if not gdf_roads.empty else 0.0
    not_assessable_m = float(gdf_roads[gdf_roads["status"] == "not_assessable"]["length_m"].sum()) if not gdf_roads.empty else 0.0

    total_bridges = len(gdf_bridges)
    severed_bridges = int(gdf_bridges["approach_severed"].sum()) if ("approach_severed" in gdf_bridges.columns and not gdf_bridges.empty) else 0

    total_buildings = len(gdf_buildings)
    flooded_bldgs = int(gdf_buildings["is_flooded"].sum()) if ("is_flooded" in gdf_buildings.columns and not gdf_buildings.empty) else 0

    return DamageSummary(
        total_road_km=total_road_m / 1000.0,
        affected_road_km=affected_m / 1000.0,
        not_affected_road_km=not_affected_m / 1000.0,
        not_assessable_road_km=not_assessable_m / 1000.0,
        total_bridges=total_bridges,
        severed_bridges=severed_bridges,
        flooded_buildings_count=flooded_bldgs,
        total_buildings=total_buildings,
    )


# =============================================================================
# High-Level API Function
# =============================================================================

def assess_infrastructure_damage(
    G: nx.Graph,
    gdf_roads: gpd.GeoDataFrame,
    gdf_bridges: gpd.GeoDataFrame,
    gdf_buildings: gpd.GeoDataFrame,
    flood_gdf: gpd.GeoDataFrame,
    shadow_mask_path: Optional[Union[str, Path]] = None,
    layover_mask_path: Optional[Union[str, Path]] = None,
    out_dir: Union[str, Path] = "./outputs/damage",
    approach_buffer_m: float = 40.0,
) -> Dict[str, Any]:
    """
    High-level infrastructure damage assessment coordinator.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Classify roads
    gdf_roads_assessed = classify_road_segments(
        gdf_roads, flood_gdf, shadow_mask_path, layover_mask_path
    )

    # 2. Assess bridge approaches
    gdf_bridges_assessed = assess_bridge_approaches(
        gdf_bridges, flood_gdf, approach_buffer_m
    )

    # 3. Assess buildings
    gdf_buildings_assessed = assess_building_damage(
        gdf_buildings, flood_gdf
    )

    # 4. Compute summary
    summary = compute_summary(
        gdf_roads_assessed, gdf_bridges_assessed, gdf_buildings_assessed
    )

    # 5. Update graph edge statuses
    flood_union = unary_union(flood_gdf.geometry.values) if len(flood_gdf) > 0 else None
    for u, v, data in G.edges(data=True):
        geom = data.get("geometry")
        is_bridge = data.get("is_bridge", False)
        status = "not_affected"

        if geom is not None and flood_union is not None:
            if geom.intersects(flood_union):
                status = "affected"
            elif is_bridge:
                # Check 40m approach
                coords = list(geom.coords)
                if len(coords) >= 2:
                    s_buf = Point(coords[0]).buffer(approach_buffer_m)
                    e_buf = Point(coords[-1]).buffer(approach_buffer_m)
                    if s_buf.intersects(flood_union) or e_buf.intersects(flood_union):
                        status = "affected"

        data["status"] = status

    # 6. Save GeoJSON outputs
    roads_out = out_path / "roads_assessed.geojson"
    bridges_out = out_path / "bridges_assessed.geojson"
    buildings_out = out_path / "buildings_assessed.geojson"
    summary_out = out_path / "damage_summary.json"

    gdf_roads_assessed.to_file(roads_out, driver="GeoJSON")
    gdf_bridges_assessed.to_file(bridges_out, driver="GeoJSON")
    gdf_buildings_assessed.to_file(buildings_out, driver="GeoJSON")

    summary_dict = summary.to_dict()
    summary_out.write_text(json.dumps(summary_dict, indent=2))

    logger.info(f"Damage assessment finished: {summary.affected_road_km:.2f} km roads affected, {summary.severed_bridges} bridges severed, {summary.flooded_buildings_count} buildings flooded")

    return {
        "roads_assessed": gdf_roads_assessed,
        "bridges_assessed": gdf_bridges_assessed,
        "buildings_assessed": gdf_buildings_assessed,
        "graph": G,
        "summary": summary,
        "summary_dict": summary_dict,
        "files": {
            "roads": roads_out,
            "bridges": bridges_out,
            "buildings": buildings_out,
            "summary": summary_out,
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Infrastructure Damage Assessment")
    parser.add_argument("--roads", type=str, required=True)
    parser.add_argument("--bridges", type=str, required=True)
    parser.add_argument("--buildings", type=str, required=True)
    parser.add_argument("--flood", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./outputs/damage")
    args = parser.parse_args()

    r = gpd.read_file(args.roads)
    b = gpd.read_file(args.bridges)
    bd = gpd.read_file(args.buildings)
    f = gpd.read_file(args.flood)

    from .graph_builder import build_network
    G = build_network(r, b, gpd.GeoDataFrame(), bd)

    res = assess_infrastructure_damage(G, r, b, bd, f, out_dir=args.out_dir)
    print("Damage Assessment Summary:")
    for k, v in res["summary_dict"].items():
        print(f"  {k:25s}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
