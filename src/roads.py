"""
src/roads.py - Road network flood tagging, bridge vulnerability assessment, and GeoJSON export.

Requirements implemented:
1. Import tag_edges from src.core_logic.
2. tag_road_network(G, flood_mask, transform, raster_crs, graph_crs="EPSG:4326", flooded_frac=0.3):
   Applies tag_edges to classify all edges. Returns modified graph G.
3. assess_bridges(G, flood_mask, transform, raster_crs, graph_crs="EPSG:4326"):
   Identifies bridge edges (using tags bridge=yes, man_made=bridge, tunnel, or structure attributes).
   For bridges that intersect floodwaters or connect to flooded road segments, flags them as
   likely_damaged=True with a damage risk score (0.0 to 1.0) and failure mode
   ('submerged', 'approaches_cut', 'high_hydrodynamic_load').
4. roads_to_geojson(G, output_path=None):
   Converts NetworkX road graph into a GeoJSON FeatureCollection where each LineString feature contains:
   id, name, highway, length_m, status ('passable', 'flooded', 'likely_damaged'), flood_frac,
   is_bridge, damage_score, failure_mode. Writes to disk if output_path is specified.
5. compute_road_stats(G):
   Returns dict with: total road km, flooded road km, percentage cut, damaged bridges count,
   impassable primary corridors.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform

from src.core_logic import _edge_line, _iter_edges, _utm_crs, tag_edges

__all__ = [
    "tag_road_network",
    "assess_bridges",
    "roads_to_geojson",
    "compute_road_stats",
    "is_bridge_edge",
]


def is_bridge_edge(data: dict) -> bool:
    """
    Identifies whether an edge represents a bridge, tunnel, or elevated crossing structure.
    Checks tags: bridge=yes, man_made=bridge, tunnel, or structure attributes.
    Handles single values, strings, booleans, and lists of values (common in OSMnx).
    """
    def _matches_any(val: Any, target_set: Optional[set] = None) -> bool:
        if val is None:
            return False
        if isinstance(val, (list, tuple, set)):
            return any(_matches_any(v, target_set) for v in val)
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if target_set is not None:
            return s in target_set
        return s not in ("no", "false", "none", "0", "")

    # 1. bridge tag: 'yes', 'viaduct', 'aqueduct', 'cantilever', etc.
    if _matches_any(data.get("bridge")):
        return True

    # 2. man_made tag: 'bridge'
    if _matches_any(data.get("man_made"), target_set={"bridge"}):
        return True

    # 3. tunnel tag: 'yes', 'culvert', etc.
    if _matches_any(data.get("tunnel")):
        return True

    # 4. structure attributes
    if _matches_any(data.get("structure"), target_set={"bridge", "tunnel", "viaduct", "culvert", "aqueduct"}):
        return True

    return False


def _get_node_xy(G: nx.Graph, node: Any) -> Tuple[float, float]:
    """Retrieve (x, y) or (lon, lat) coordinates from node attributes."""
    nd = G.nodes[node]
    x = nd.get("x", nd.get("lon", nd.get("longitude", 0.0)))
    y = nd.get("y", nd.get("lat", nd.get("latitude", 0.0)))
    return float(x), float(y)


def _get_edge_length_m(G: nx.Graph, u: Any, v: Any, d: dict, graph_crs: str = "EPSG:4326") -> float:
    """
    Returns edge length in meters. If 'length' is already present, returns it.
    Otherwise projects geometry to local UTM zone and computes length in meters.
    """
    raw_len = d.get("length")
    if raw_len is not None:
        try:
            val = float(raw_len)
            if not math.isnan(val) and val >= 0:
                return val
        except (ValueError, TypeError):
            pass

    line = _edge_line(G, u, v, d)
    if line.is_empty:
        return 0.0

    # If already projected metric CRS, length is directly in meters
    try:
        crs_obj = CRS.from_user_input(graph_crs)
        if crs_obj.is_projected and "metre" in crs_obj.axis_info[0].unit_name.lower():
            return float(line.length)
    except Exception:
        pass

    # Geographic CRS (e.g. EPSG:4326): project to UTM at centroid
    try:
        c = line.centroid
        utm = _utm_crs(c.x, c.y)
        to_utm = Transformer.from_crs(graph_crs, utm, always_xy=True).transform
        utm_line = shp_transform(to_utm, line)
        return float(utm_line.length)
    except Exception:
        return float(line.length) * 111_000.0


def _get_adjacent_edges(G: nx.Graph, node: Any, exclude_node: Any) -> List[dict]:
    """
    Finds edge attributes of all connected segments at `node` excluding edges
    connecting to `exclude_node`. Handles directed, undirected, and multi-graphs.
    """
    adjacent_data = []
    if G.is_directed():
        if G.is_multigraph():
            for _, nbr, _, ed in G.out_edges(node, keys=True, data=True):
                if nbr != exclude_node:
                    adjacent_data.append(ed)
            for nbr, _, _, ed in G.in_edges(node, keys=True, data=True):
                if nbr != exclude_node:
                    adjacent_data.append(ed)
        else:
            for _, nbr, ed in G.out_edges(node, data=True):
                if nbr != exclude_node:
                    adjacent_data.append(ed)
            for nbr, _, ed in G.in_edges(node, data=True):
                if nbr != exclude_node:
                    adjacent_data.append(ed)
    else:
        if G.is_multigraph():
            for _, nbr, _, ed in G.edges(node, keys=True, data=True):
                if nbr != exclude_node:
                    adjacent_data.append(ed)
        else:
            for _, nbr, ed in G.edges(node, data=True):
                if nbr != exclude_node:
                    adjacent_data.append(ed)
    return adjacent_data


def tag_road_network(
    G: nx.Graph,
    flood_mask: np.ndarray,
    transform: Union[Affine, Tuple[float, ...]],
    raster_crs: Any,
    graph_crs: str = "EPSG:4326",
    flooded_frac: float = 0.3,
) -> nx.Graph:
    """
    Applies `tag_edges` from src.core_logic to classify all edges against flood raster.

    Mutates and returns G, adding edge attributes:
      - flood_frac: float (0.0 to 1.0, or nan if off-raster)
      - flooded: bool (True if flood_frac >= flooded_frac)
      - covered: bool (True if in raster bounds)
      - status: 'flooded' or 'passable'
    """
    tag_edges(
        G=G,
        mask=flood_mask,
        transform=transform,
        raster_crs=raster_crs,
        graph_crs=graph_crs,
        flooded_frac=flooded_frac,
    )

    for _, _, _, d in _iter_edges(G):
        if d.get("flooded", False):
            d["status"] = "flooded"
        elif "status" not in d:
            d["status"] = "passable"

    return G


def assess_bridges(
    G: nx.Graph,
    flood_mask: Optional[np.ndarray] = None,
    transform: Optional[Union[Affine, Tuple[float, ...]]] = None,
    raster_crs: Optional[Any] = None,
    graph_crs: str = "EPSG:4326",
) -> nx.Graph:
    """
    Identifies bridge edges (using tags bridge=yes, man_made=bridge, tunnel, or structure attributes).
    For bridges that intersect floodwaters or connect to flooded road segments, flags them as:
      - likely_damaged = True
      - damage_score = float (0.0 to 1.0)
      - failure_mode = 'submerged' | 'approaches_cut' | 'high_hydrodynamic_load'
      - status = 'likely_damaged'

    Mutates and returns modified graph G.
    """
    # 1. If flood raster is provided, tag road network edges
    if flood_mask is not None and transform is not None and raster_crs is not None:
        tag_road_network(
            G=G,
            flood_mask=flood_mask,
            transform=transform,
            raster_crs=raster_crs,
            graph_crs=graph_crs,
        )

    # 2. First pass: mark bridge status and set safe baseline attributes on all edges
    for _, _, _, d in _iter_edges(G):
        is_br = is_bridge_edge(d)
        d["is_bridge"] = is_br
        d.setdefault("likely_damaged", False)
        d.setdefault("damage_score", 0.0)
        d.setdefault("failure_mode", "")
        if "status" not in d:
            d["status"] = "flooded" if d.get("flooded", False) else "passable"

    # 3. Second pass: evaluate bridge structural and hydraulic vulnerability
    for u, v, k, d in _iter_edges(G):
        if not d.get("is_bridge", False):
            continue

        raw_ff = d.get("flood_frac")
        if raw_ff is None or math.isnan(raw_ff):
            ff = 0.0
        else:
            ff = float(np.clip(raw_ff, 0.0, 1.0))

        is_flooded = bool(d.get("flooded", False) or ff >= 0.3)
        intersects_flood = is_flooded or (ff > 0.01)

        # Check connecting approach road segments at endpoints u and v
        def _is_seg_flooded(ed: dict) -> bool:
            if ed.get("flooded", False):
                return True
            eff = ed.get("flood_frac")
            return eff is not None and not math.isnan(eff) and eff >= 0.3

        u_approaches = _get_adjacent_edges(G, u, exclude_node=v)
        v_approaches = _get_adjacent_edges(G, v, exclude_node=u)

        u_cut = any(_is_seg_flooded(ed) for ed in u_approaches)
        v_cut = any(_is_seg_flooded(ed) for ed in v_approaches)

        approaches_cut = u_cut or v_cut
        both_approaches_cut = u_cut and v_cut

        # If bridge intersects floodwaters or connects to flooded road segments
        if intersects_flood or approaches_cut:
            d["likely_damaged"] = True
            d["bridge_damaged"] = True
            d["status"] = "damaged_bridge"

            if is_flooded:
                if ff >= 0.5:
                    # 1. Deck is submerged / overtopped by floodwaters
                    d["failure_mode"] = "submerged"
                    sub_score = 0.70 + 0.30 * min(1.0, max(0.0, (ff - 0.5) / 0.5))
                    if approaches_cut:
                        sub_score = min(1.0, sub_score + 0.05)
                    d["damage_score"] = round(float(sub_score), 2)
                else:
                    # 2. Significant floodwater on deck under high flow velocity
                    d["failure_mode"] = "high_hydrodynamic_load"
                    hydro_score = 0.50 + 0.30 * min(1.0, ff / 0.5)
                    if approaches_cut:
                        hydro_score = min(1.0, hydro_score + 0.15)
                    d["damage_score"] = round(float(hydro_score), 2)
            elif approaches_cut:
                # 3. Bridge deck itself is not flooded, but approach segments are severed
                d["failure_mode"] = "approaches_cut"
                app_score = 0.80 if both_approaches_cut else 0.60
                d["damage_score"] = round(float(app_score), 2)
            else:
                # 4. Partial edge touch with floodwaters without approach severance
                d["failure_mode"] = "high_hydrodynamic_load"
                d["damage_score"] = round(float(0.40 + 0.30 * ff), 2)
        else:
            # Undamaged bridge
            d["likely_damaged"] = False
            d["damage_score"] = 0.0
            d["failure_mode"] = ""
            d["status"] = "passable"

    return G


def roads_to_geojson(
    G: nx.Graph,
    output_path: Optional[Union[str, Path]] = None,
    graph_crs: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Converts the NetworkX road graph into a GeoJSON FeatureCollection.
    Each LineString feature contains:
      - id: edge key or osm id
      - name: road name
      - highway: road classification (primary, secondary, residential, etc.)
      - length_m: road length in meters
      - status: 'passable' (green), 'flooded' (red), 'likely_damaged' (amber/orange)
      - flood_frac: fraction of edge flooded (0.0 to 1.0)
      - is_bridge: bool
      - damage_score: float (0.0 to 1.0)
      - failure_mode: string ('submerged', 'approaches_cut', 'high_hydrodynamic_load', or '')

    If `output_path` is specified, writes the GeoJSON to disk.
    Returns the FeatureCollection dictionary.
    """
    crs_str = graph_crs or G.graph.get("crs", "EPSG:4326")
    try:
        source_crs = CRS.from_user_input(crs_str)
        wgs84 = CRS.from_epsg(4326)
        needs_transform = (source_crs != wgs84)
        transformer = (
            Transformer.from_crs(source_crs, wgs84, always_xy=True).transform
            if needs_transform
            else None
        )
    except Exception:
        transformer = None

    features: List[Dict[str, Any]] = []

    for u, v, k, d in _iter_edges(G):
        raw_line = _edge_line(G, u, v, d)
        if raw_line.is_empty:
            continue

        if transformer is not None:
            try:
                line = shp_transform(transformer, raw_line)
            except Exception:
                line = raw_line
        else:
            line = raw_line

        # Edge ID: osmid or compound node-key string
        raw_id = d.get("osmid")
        if raw_id is not None:
            if isinstance(raw_id, (list, tuple, set)):
                edge_id = str(list(raw_id)[0]) if len(raw_id) == 1 else ",".join(map(str, raw_id))
            else:
                edge_id = str(raw_id)
        else:
            edge_id = f"{u}_{v}_{k}" if k is not None else f"{u}_{v}"

        # Road name
        raw_name = d.get("name")
        if isinstance(raw_name, (list, tuple, set)):
            road_name = ", ".join(str(n) for n in raw_name)
        elif raw_name is not None:
            road_name = str(raw_name)
        else:
            road_name = "unnamed"

        # Highway classification
        raw_hw = d.get("highway")
        if isinstance(raw_hw, (list, tuple, set)):
            highway = ", ".join(str(h) for h in raw_hw)
        elif raw_hw is not None:
            highway = str(raw_hw)
        else:
            highway = "unclassified"

        # Length in meters
        length_m = round(_get_edge_length_m(G, u, v, d, graph_crs=crs_str), 2)

        # Flood fraction (ensure valid JSON float, no NaNs)
        raw_ff = d.get("flood_frac")
        if raw_ff is None or math.isnan(raw_ff):
            flood_frac = 0.0
        else:
            flood_frac = round(float(np.clip(raw_ff, 0.0, 1.0)), 4)

        # Bridge and damage attributes
        is_bridge = bool(d.get("is_bridge", False) or is_bridge_edge(d))
        likely_damaged = bool(d.get("likely_damaged", False))
        damage_score = round(float(d.get("damage_score", 0.0)), 3)
        failure_mode = str(d.get("failure_mode") or "")

        # Status: 'passable', 'flooded', 'likely_damaged'
        if likely_damaged:
            status = "likely_damaged"
        elif d.get("flooded", False) or flood_frac >= 0.3:
            status = "flooded"
        else:
            status = d.get("status", "passable")

        # GeoJSON LineString coordinates [[lon, lat], ...]
        coords = [[round(pt[0], 7), round(pt[1], 7)] for pt in line.coords]

        properties = {
            "id": edge_id,
            "name": road_name,
            "highway": highway,
            "length_m": length_m,
            "status": status,
            "flooded": bool(status == "flooded" or d.get("flooded", False) or flood_frac >= 0.3),
            "flood_frac": flood_frac,
            "is_bridge": is_bridge,
            "damage_score": damage_score,
            "failure_mode": failure_mode,
        }

        feature = {
            "type": "Feature",
            "id": edge_id,
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
            "properties": properties,
        }
        features.append(feature)

    feature_collection = {
        "type": "FeatureCollection",
        "features": features,
    }

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(feature_collection, f, indent=2)

    return feature_collection


def compute_road_stats(G: nx.Graph, graph_crs: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes summary road statistics for the road network graph G:
      - total road km
      - flooded road km
      - percentage cut
      - damaged bridges count
      - impassable primary corridors

    Returns dictionary containing both standard snake_case and label keys.
    """
    crs_str = graph_crs or G.graph.get("crs", "EPSG:4326")
    total_m = 0.0
    flooded_m = 0.0
    total_bridges_count = 0
    damaged_bridges_count = 0
    impassable_primary_corridors = 0

    primary_types = {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
    }

    for u, v, k, d in _iter_edges(G):
        length_m = _get_edge_length_m(G, u, v, d, graph_crs=crs_str)
        total_m += length_m

        raw_ff = d.get("flood_frac")
        has_ff = raw_ff is not None and not math.isnan(raw_ff)
        is_flooded = bool(d.get("flooded", False) or (has_ff and raw_ff >= 0.3))
        is_damaged = bool(d.get("likely_damaged", False))
        status = d.get("status", "passable")

        if is_flooded:
            flooded_m += length_m

        is_br = bool(d.get("is_bridge", False) or is_bridge_edge(d))
        if is_br:
            total_bridges_count += 1
            if is_damaged:
                damaged_bridges_count += 1

        # Check primary corridors
        raw_hw = d.get("highway")
        is_primary = False
        if raw_hw is not None:
            if isinstance(raw_hw, (list, tuple, set)):
                is_primary = any(str(h).strip().lower() in primary_types for h in raw_hw)
            else:
                is_primary = str(raw_hw).strip().lower() in primary_types

        impassable = is_flooded or is_damaged or status in ("flooded", "likely_damaged")
        if is_primary and impassable:
            impassable_primary_corridors += 1

    total_km = total_m / 1000.0
    flooded_km = flooded_m / 1000.0
    pct_cut = (flooded_km / total_km * 100.0) if total_km > 0.0 else 0.0

    stats = {
        "total_road_km": round(total_km, 2),
        "flooded_road_km": round(flooded_km, 2),
        "percentage_cut": round(pct_cut, 2),
        "damaged_bridges_count": int(damaged_bridges_count),
        "impassable_primary_corridors": int(impassable_primary_corridors),
        # Space-separated aliases for flexible lookup
        "total road km": round(total_km, 2),
        "flooded road km": round(flooded_km, 2),
        "percentage cut": round(pct_cut, 2),
        "damaged bridges count": int(damaged_bridges_count),
        "impassable primary corridors": int(impassable_primary_corridors),
        # Convenient short aliases
        "total_km": round(total_km, 2),
        "flooded_km": round(flooded_km, 2),
        "total_bridges": int(total_bridges_count),
        "damaged_bridges": int(damaged_bridges_count),
    }

    return stats
