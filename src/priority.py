"""
src/priority.py - Multi-Factor Rescue Priority Ranking & Dispatch Manifest.

Computes a standardized Rescue Priority Score (0 to 100) for cut-off settlements
evaluating:
  1. Population at risk (default weight 0.40)
  2. Medical isolation distance to nearest functioning hospital/hub (default weight 0.25)
  3. Settlement physical isolation size / trapped nodes (default weight 0.20)
  4. Critical vulnerability factor (e.g. collapsed bridges, highway severance) (default weight 0.15)

Assigns standardized Priority Tiers:
  - CRITICAL (Score >= 75, red)
  - HIGH     (Score 50-74, orange)
  - MEDIUM   (Score 25-49, yellow)
  - LOW      (Score < 25, blue)

Generates tactical rescue manifests sorted by urgency with operational action directives
(e.g., amphibious rescue, boat evacuation, road detour possible) and exports to CSV and GeoJSON.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Standard weights according to specification
DEFAULT_WEIGHTS: Dict[str, float] = {
    "population": 0.40,
    "distance": 0.25,
    "trapped_nodes": 0.20,
    "vulnerability": 0.15,
}

# Color codes for mapping & dashboard UI
TIER_COLORS: Dict[str, str] = {
    "CRITICAL": "#D32F2F",  # Red
    "HIGH": "#F57C00",      # Orange
    "MEDIUM": "#FBC02D",    # Yellow
    "LOW": "#1976D2",       # Blue
}


def _assign_tier_and_color(score: float) -> Tuple[str, str]:
    """Assigns priority tier and corresponding color code based on score."""
    if score >= 75.0:
        return "CRITICAL", TIER_COLORS["CRITICAL"]
    elif score >= 50.0:
        return "HIGH", TIER_COLORS["HIGH"]
    elif score >= 25.0:
        return "MEDIUM", TIER_COLORS["MEDIUM"]
    else:
        return "LOW", TIER_COLORS["LOW"]


def _determine_recommended_action(
    tier: str,
    score: float,
    pop: float,
    dist_km: float,
    has_bridge_cut: bool,
    has_highway_cut: bool,
) -> str:
    """
    Synthesizes tactical dispatch recommendation based on urgency and physical access constraints.

    Matches domain operational requirements:
      - 'Airdrop supplies / Amphibious rescue'
      - 'Boat evacuation'
      - 'Road detour possible'
    """
    if tier == "CRITICAL":
        if has_bridge_cut or dist_km >= 8.0:
            return "Airdrop supplies / Amphibious rescue"
        elif pop >= 500:
            return "Immediate boat evacuation & medical air-evac"
        else:
            return "Airdrop supplies / Amphibious rescue"
    elif tier == "HIGH":
        if pop >= 150 or has_bridge_cut:
            return "Boat evacuation"
        elif dist_km >= 6.0:
            return "Amphibious transport & emergency supply drop"
        else:
            return "Boat evacuation"
    elif tier == "MEDIUM":
        if dist_km <= 5.0 and not has_bridge_cut:
            return "Road detour possible / High-clearance vehicle rescue"
        else:
            return "Shallow-draft boat evacuation / monitoring"
    else:  # LOW
        return "Road detour possible / Monitor flood levels"


def _normalize_feature(values: np.ndarray, reference_cap: float) -> np.ndarray:
    """
    Normalizes a numerical feature into [0.0, 1.0] by blending relative min-max scaling
    with absolute benchmark capacity. This prevents divide-by-zero on single-cluster or
    uniform-value inputs while maintaining discrimination across clusters.
    """
    if len(values) == 0:
        return values
    v_clean = np.nan_to_num(values, nan=0.0)
    v_min = float(np.min(v_clean))
    v_max = float(np.max(v_clean))

    # Absolute benchmark scaling
    abs_scaled = np.clip(v_clean / reference_cap, 0.0, 1.0)

    if v_max > v_min:
        # Relative min-max
        rel_scaled = (v_clean - v_min) / (v_max - v_min)
        # Blend relative ranking (65%) with absolute scale (35%)
        combined = 0.65 * rel_scaled + 0.35 * abs_scaled
    else:
        # If all values are identical or only 1 item
        combined = abs_scaled

    return np.clip(combined, 0.0, 1.0)


def calculate_priority(
    clusters: Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Any], pd.DataFrame, gpd.GeoDataFrame],
    weights: Optional[Dict[str, float]] = None,
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Computes a standardized Rescue Priority Score (0 to 100) for each isolated settlement:
      - Population at risk (default weight 0.40)
      - Complete medical isolation distance (default weight 0.25)
      - Settlement physical isolation size / trapped nodes (default weight 0.20)
      - Critical vulnerability factor (default weight 0.15)

    Assigns Priority Tiers:
      - CRITICAL (Score >= 75, red)
      - HIGH     (Score 50-74, orange)
      - MEDIUM   (Score 25-49, yellow)
      - LOW      (Score < 25, blue)

    Args:
        clusters: List of settlement dicts (from src.isolation), or DataFrame,
                  or tuple (clusters, isolated_nodes).
        weights: Optional dictionary of component weights.
                 Keys: 'population', 'distance', 'trapped_nodes', 'vulnerability'.

    Returns:
        Clusters with priority score, tier, color, and component scores added,
        sorted in descending order of urgency.
    """
    # Auto-unwrap tuple if passed directly from identify_cut_off_settlements
    if isinstance(clusters, tuple) and len(clusters) == 2 and isinstance(clusters[0], list):
        clusters = clusters[0]

    is_df_input = isinstance(clusters, (pd.DataFrame, gpd.GeoDataFrame))
    if is_df_input:
        if clusters.empty:
            return clusters
        cluster_list = clusters.to_dict(orient="records")
    else:
        cluster_list = [dict(c) for c in clusters]

    if not cluster_list:
        return clusters

    # Normalize weights to ensure sum == 1.0
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k, v in weights.items():
            if k in w:
                w[k] = float(v)
            elif k in ("pop", "estimated_population"):
                w["population"] = float(v)
            elif k in ("dist", "distance_km", "hub_dist"):
                w["distance"] = float(v)
            elif k in ("nodes", "trapped_nodes", "nodes_count"):
                w["trapped_nodes"] = float(v)
            elif k in ("vuln", "vulnerability_factor", "critical_vulnerability"):
                w["vulnerability"] = float(v)

    total_w = sum(w.values())
    if total_w > 0:
        w = {k: v / total_w for k, v in w.items()}
    else:
        w = dict(DEFAULT_WEIGHTS)

    n = len(cluster_list)
    pops = np.zeros(n, dtype=float)
    dists = np.zeros(n, dtype=float)
    nodes = np.zeros(n, dtype=float)
    vulns = np.zeros(n, dtype=float)

    for i, c in enumerate(cluster_list):
        # Extract population
        p = c.get("estimated_population", c.get("population", 0.0))
        pops[i] = float(p) if p is not None and not math.isnan(p) else 0.0

        # Extract distance in km
        d = c.get("dist_to_hub_km", c.get("distance_to_nearest_hub_km"))
        if d is None or math.isnan(d):
            m = c.get("shortest_distance_to_hub_m", c.get("distance_to_nearest_hub_m"))
            d = (float(m) / 1000.0) if m is not None and m >= 0 else 10.0
        dists[i] = max(float(d), 0.0)

        # Extract trapped nodes count
        nd = c.get("trapped_nodes", c.get("n_trapped_nodes", 1))
        nodes[i] = max(float(nd) if nd is not None else 1.0, 1.0)

        # Compute Critical Vulnerability Factor (0.0 to 1.0)
        has_bridge = bool(c.get("has_bridge_cut", False))
        has_highway = bool(c.get("has_highway_cut", False))
        num_cuts = int(c.get("num_cut_edges", 0))

        # Check obstacle text if flags were not present
        obs_text = str(c.get("critical_access_obstacles", "")).lower()
        if "bridge" in obs_text or "viaduct" in obs_text:
            has_bridge = True
        if "highway" in obs_text or "motorway" in obs_text or "trunk" in obs_text or "primary" in obs_text:
            has_highway = True

        # Base vulnerability for any completely cut off settlement
        v_score = 0.10
        if has_bridge:
            v_score += 0.40  # Collapsed/submerged bridge removes road crossing capability
        if has_highway:
            v_score += 0.25  # Primary arterial cut removes high-capacity evacuation route
        if num_cuts >= 3:
            v_score += 0.15
        elif num_cuts >= 1:
            v_score += 0.08
        if pops[i] >= 300 and dists[i] >= 5.0:
            v_score += 0.10  # Compounding crisis: large population deeply isolated

        vulns[i] = min(v_score, 1.0)

    # Standardize components
    norm_pops = _normalize_feature(pops, reference_cap=2500.0)
    norm_dists = _normalize_feature(dists, reference_cap=20.0)
    norm_nodes = _normalize_feature(nodes, reference_cap=80.0)
    norm_vulns = vulns  # Already scaled [0, 1]

    # Calculate final scores (0 to 100)
    raw_scores = (
        w["population"] * norm_pops
        + w["distance"] * norm_dists
        + w["trapped_nodes"] * norm_nodes
        + w["vulnerability"] * norm_vulns
    ) * 100.0

    scores = np.round(np.clip(raw_scores, 0.0, 100.0), 1)

    for i, c in enumerate(cluster_list):
        score_val = float(scores[i])
        tier, color = _assign_tier_and_color(score_val)
        c["priority_score"] = score_val
        c["priority_tier"] = tier
        c["priority_color"] = color
        c["Priority_Score"] = score_val
        c["Priority_Tier"] = tier
        c["Priority_Color"] = color
        c["component_scores"] = {
            "population_norm": round(float(norm_pops[i]), 3),
            "distance_norm": round(float(norm_dists[i]), 3),
            "trapped_nodes_norm": round(float(norm_nodes[i]), 3),
            "vulnerability_factor": round(float(norm_vulns[i]), 3),
        }
        # Add recommended action
        c["recommended_action"] = _determine_recommended_action(
            tier=tier,
            score=score_val,
            pop=pops[i],
            dist_km=dists[i],
            has_bridge_cut=bool(c.get("has_bridge_cut", False)),
            has_highway_cut=bool(c.get("has_highway_cut", False)),
        )
        c["Recommended_Action"] = c["recommended_action"]

    # Sort descending by priority score, then population
    cluster_list.sort(
        key=lambda x: (
            x["priority_score"],
            x.get("estimated_population", x.get("population", 0.0)),
        ),
        reverse=True,
    )

    for rank_idx, c in enumerate(cluster_list, start=1):
        c["rank"] = rank_idx
        c["Rank"] = rank_idx

    if is_df_input:
        res_df = pd.DataFrame(cluster_list)
        if isinstance(clusters, gpd.GeoDataFrame) and "geometry" in clusters.columns:
            return gpd.GeoDataFrame(res_df, geometry="geometry", crs=clusters.crs)
        return res_df

    return cluster_list



def generate_rescue_manifest(
    clusters_df: Union[pd.DataFrame, gpd.GeoDataFrame, List[Dict[str, Any]], Tuple[Any, Any]],
) -> pd.DataFrame:
    """
    Returns a clean, standardized pandas DataFrame ready for rescue coordinators,
    sorted by Rank (Rank 1 = Highest Priority).

    Columns:
      [Rank, Cluster_ID, Priority_Tier, Priority_Score, Estimated_Population,
       Trapped_Nodes, Dist_to_Hub_km, Nearest_Hub, Centroid_Lat, Centroid_Lon,
       Recommended_Action]

    Args:
        clusters_df: List of cluster dicts or DataFrame.

    Returns:
        Structured rescue manifest DataFrame sorted by Rank.
    """
    # Auto-unwrap tuple if caller passed return value of identify_cut_off_settlements
    if isinstance(clusters_df, tuple) and len(clusters_df) == 2 and isinstance(clusters_df[0], list):
        clusters_df = clusters_df[0]

    manifest_cols = [
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

    # If empty input
    if (isinstance(clusters_df, (pd.DataFrame, gpd.GeoDataFrame)) and clusters_df.empty) or (
        isinstance(clusters_df, list) and len(clusters_df) == 0
    ):
        return pd.DataFrame(columns=manifest_cols)

    # Convert to list of dicts for uniform processing
    if isinstance(clusters_df, (pd.DataFrame, gpd.GeoDataFrame)):
        records = clusters_df.to_dict(orient="records")
    else:
        records = [dict(c) for c in clusters_df]

    # If priority calculation has not been executed yet, compute it now
    if not records or "Priority_Score" not in records[0] and "priority_score" not in records[0]:
        records = calculate_priority(records)

    # Extract geometries if present for preservation in attrs
    geom_map = {}
    for r in records:
        cid = r.get("cluster_id", r.get("Cluster_ID"))
        if "geometry" in r and r["geometry"] is not None:
            geom_map[cid] = r["geometry"]

    # Sort descending by priority score
    records.sort(
        key=lambda x: (
            float(x.get("Priority_Score", x.get("priority_score", 0.0))),
            float(x.get("Estimated_Population", x.get("estimated_population", 0.0))),
        ),
        reverse=True,
    )

    rows = []
    for rank_idx, r in enumerate(records, start=1):
        cid = str(r.get("Cluster_ID", r.get("cluster_id", f"Village_Cluster_{rank_idx:02d}")))
        tier = str(r.get("Priority_Tier", r.get("priority_tier", "LOW")))
        score = float(r.get("Priority_Score", r.get("priority_score", 0.0)))

        pop_val = r.get("Estimated_Population", r.get("estimated_population", r.get("population", 0)))
        pop_int = int(round(float(pop_val))) if pop_val is not None and not math.isnan(pop_val) else 0

        trapped_val = r.get("Trapped_Nodes", r.get("trapped_nodes", r.get("n_trapped_nodes", 0)))
        trapped_int = int(trapped_val) if trapped_val is not None else 0

        dist_val = r.get("Dist_to_Hub_km", r.get("dist_to_hub_km", r.get("distance_to_nearest_hub_km", 0.0)))
        dist_float = round(float(dist_val), 2) if dist_val is not None and not math.isnan(dist_val) else 0.0

        hub_str = str(r.get("Nearest_Hub", r.get("nearest_hub", "Unknown")))

        lat_val = r.get("Centroid_Lat", r.get("center_lat", r.get("lat", 0.0)))
        lon_val = r.get("Centroid_Lon", r.get("center_lon", r.get("lon", 0.0)))

        # Determine action if not already assigned
        action = r.get("Recommended_Action", r.get("recommended_action"))
        if not action:
            has_br = bool(r.get("has_bridge_cut", False))
            has_hw = bool(r.get("has_highway_cut", False))
            action = _determine_recommended_action(tier, score, pop_int, dist_float, has_br, has_hw)

        row = {
            "Rank": rank_idx,
            "Cluster_ID": cid,
            "Priority_Tier": tier,
            "Priority_Score": round(score, 1),
            "Estimated_Population": pop_int,
            "Trapped_Nodes": trapped_int,
            "Dist_to_Hub_km": dist_float,
            "Nearest_Hub": hub_str,
            "Centroid_Lat": round(float(lat_val), 6),
            "Centroid_Lon": round(float(lon_val), 6),
            "Recommended_Action": action,
        }
        rows.append(row)

    manifest_df = pd.DataFrame(rows, columns=manifest_cols)
    if geom_map:
        manifest_df.attrs["geometry_map"] = geom_map

    return manifest_df


def export_rescue_plan(
    ranked_df: pd.DataFrame,
    csv_path: Union[str, Path],
    json_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Optional[str]]:
    """
    Exports the rescue manifest to CSV and GeoJSON for field teams and tactical GIS.

    Args:
        ranked_df: pandas DataFrame from generate_rescue_manifest.
        csv_path: Path to output CSV file.
        json_path: Optional path to output GeoJSON file.

    Returns:
        Dictionary containing paths written: {'csv_path': str, 'json_path': Optional[str]}
    """
    out_csv = Path(csv_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked_df.to_csv(out_csv, index=False)
    logger.info(f"Saved rescue manifest CSV to: {out_csv}")

    out_json_str = None
    if json_path is not None:
        out_json = Path(json_path)
        out_json.parent.mkdir(parents=True, exist_ok=True)

        if ranked_df.empty:
            empty_geojson = {"type": "FeatureCollection", "features": []}
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(empty_geojson, f, indent=2)
            out_json_str = str(out_json)
        else:
            geom_map = getattr(ranked_df, "attrs", {}).get("geometry_map", {})
            geometries = []

            for _, row in ranked_df.iterrows():
                cid = row.get("Cluster_ID")
                if cid in geom_map:
                    geometries.append(geom_map[cid])
                elif "geometry" in row and row["geometry"] is not None:
                    geometries.append(row["geometry"])
                else:
                    lon = row.get("Centroid_Lon", 0.0)
                    lat = row.get("Centroid_Lat", 0.0)
                    geometries.append(Point(lon, lat))

            # Build GeoDataFrame with EPSG:4326 CRS
            gdf = gpd.GeoDataFrame(ranked_df.copy(), geometry=geometries, crs="EPSG:4326")
            gdf.to_file(out_json, driver="GeoJSON")
            out_json_str = str(out_json)
            logger.info(f"Saved rescue manifest GeoJSON to: {out_json}")

    return {
        "csv_path": str(out_csv),
        "json_path": out_json_str,
    }
