"""
src/ingest.py - Data ingestion and realistic disaster scenario generator.

Provides:
  1. load_sar: Loads pre- and post-flood SAR GeoTIFFs using rasterio, handling
     polarization selection (VV preferred), type conversion (float32), and CRS alignment.
  2. load_roads: Fetches OpenStreetMap road networks via OSMnx from a bounding box
     (north, south, east, west), with caching and resilient offline fallback generation.
  3. load_hubs: Queries emergency facilities (hospitals, clinics, fire stations, relief centers)
     from OSMnx / Overpass or cached data, returning [(lon, lat, name, type)].
  4. load_population: Reads a WorldPop GeoTIFF clipped to a bounding box, or generates
     a realistic population density raster with rural hamlets and dense urban settlements.
  5. generate_demo_event: Produces a fully self-contained, high-fidelity 20x20 km
     disaster scenario for Dadu District, Sindh, Pakistan (2022 Floods) for 100% offline
     reproducibility and testing.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import rasterio
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
from shapely.geometry import LineString, Point, mapping

logger = logging.getLogger(__name__)


class Hub(NamedTuple):
    """
    Representation of an emergency relief hub or medical facility.
    Tuples can be unpacked directly as (lon, lat, name, type).
    """
    lon: float
    lat: float
    name: str
    type: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert hub to dictionary format."""
        return {
            "name": self.name,
            "lon": self.lon,
            "lat": self.lat,
            "type": self.type,
        }


def _normalize_bbox(
    bbox: Sequence[float],
) -> Tuple[float, float, float, float]:
    """
    Normalize bounding box coordinates into (north, south, east, west).

    Handles both:
      - (north, south, east, west) [Geographic standard / prompt specification]
      - (west, south, east, north) [GeoJSON / GIS standard / OSMnx 2.x]

    Returns:
        (north, south, east, west) where north > south and east > west.
    """
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with 4 coordinates, got {len(bbox)}: {bbox}")

    b0, b1, b2, b3 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Check whether pair (b0, b2) is longitudes (west, east) or (b0, b1) is latitudes (north, south)
    # In a local bounding box, coordinate span along each axis is small (< 5 deg)
    if abs(b0 - b2) < abs(b0 - b1):
        # Format is (west, south, east, north)
        west = min(b0, b2)
        east = max(b0, b2)
        south = min(b1, b3)
        north = max(b1, b3)
    else:
        # Format is (north, south, east, west)
        north = max(b0, b1)
        south = min(b0, b1)
        east = max(b2, b3)
        west = min(b2, b3)

    return north, south, east, west


def _bbox_to_osmnx(
    north: float, south: float, east: float, west: float
) -> Tuple[float, float, float, float]:
    """Convert (north, south, east, west) to OSMnx 2.x bbox format (left, bottom, right, top)."""
    return (west, south, east, north)


# ----------------------------------------------------------------------------
# 1. SAR GeoTIFF Ingestion
# ----------------------------------------------------------------------------

def _select_sar_band_index(src: rasterio.io.DatasetReader) -> int:
    """
    Find the 1-based band index for SAR co-polarization (VV), or default to band 1.
    """
    if src.count == 1:
        return 1

    # Check band descriptions (Sentinel-1 GRD usually lists 'VV', 'VH')
    if src.descriptions:
        for idx, desc in enumerate(src.descriptions, start=1):
            if desc and "vv" in str(desc).strip().lower():
                return idx

    # Check tags on each band
    for idx in range(1, src.count + 1):
        tags = src.tags(idx)
        for k, v in tags.items():
            if "vv" in str(v).lower() or "polarization=vv" in str(v).lower():
                return idx

    # Check dataset tags
    tags = src.tags()
    if "polarization" in tags and "vv" in str(tags["polarization"]).lower():
        return 1

    return 1


def load_sar(
    pre_path: Union[str, Path, os.PathLike],
    post_path: Union[str, Path, os.PathLike],
) -> Tuple[np.ndarray, np.ndarray, Affine, Any, Dict[str, Any]]:
    """
    Load pre- and post-flood SAR GeoTIFF images using rasterio.

    Features:
      - Reads multi-band or single-band imagery (prioritizes VV polarization).
      - Converts values to float32 and replaces invalid/nodata values.
      - Aligns post-flood raster grid to pre-flood raster grid if dimensions or CRS differ.

    Args:
        pre_path: Filepath to pre-flood SAR GeoTIFF.
        post_path: Filepath to post-flood SAR GeoTIFF.

    Returns:
        (pre_arr, post_arr, transform, crs, meta)
        - pre_arr: 2D numpy float32 array of pre-flood backscatter.
        - post_arr: 2D numpy float32 array of post-flood backscatter (aligned to pre_arr).
        - transform: Affine georeferencing transform.
        - crs: Coordinate Reference System (e.g. EPSG:4326 or UTM).
        - meta: Rasterio metadata dictionary.
    """
    pre_p = Path(pre_path)
    post_p = Path(post_path)

    if not pre_p.exists():
        raise FileNotFoundError(f"Pre-flood SAR GeoTIFF not found: {pre_p.resolve()}")
    if not post_p.exists():
        raise FileNotFoundError(f"Post-flood SAR GeoTIFF not found: {post_p.resolve()}")

    with rasterio.open(pre_p) as src_pre:
        pre_band_idx = _select_sar_band_index(src_pre)
        pre_arr = src_pre.read(pre_band_idx).astype(np.float32)
        transform = src_pre.transform
        crs = src_pre.crs
        meta = src_pre.meta.copy()

        # Handle pre nodata
        if src_pre.nodata is not None and not np.isnan(src_pre.nodata):
            pre_arr = np.where(pre_arr == src_pre.nodata, np.nan, pre_arr)

    with rasterio.open(post_p) as src_post:
        post_band_idx = _select_sar_band_index(src_post)
        post_raw = src_post.read(post_band_idx).astype(np.float32)

        # Handle post nodata
        if src_post.nodata is not None and not np.isnan(src_post.nodata):
            post_raw = np.where(post_raw == src_post.nodata, np.nan, post_raw)

        # If dimensions, transform, or CRS differ, reproject post to match pre
        if (
            src_post.shape != src_pre.shape
            or src_post.crs != src_pre.crs
            or src_post.transform != src_pre.transform
        ):
            logger.info("Aligning post-flood SAR raster to pre-flood raster grid...")
            post_aligned = np.empty_like(pre_arr, dtype=np.float32)
            reproject(
                source=post_raw,
                destination=post_aligned,
                src_transform=src_post.transform,
                src_crs=src_post.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
            post_arr = post_aligned
        else:
            post_arr = post_raw

    meta.update(
        count=1,
        dtype="float32",
        transform=transform,
        crs=crs,
    )

    return pre_arr, post_arr, transform, crs, meta


# ----------------------------------------------------------------------------
# 2. Road Network Ingestion & Resilient Offline Fallback
# ----------------------------------------------------------------------------

def _generate_synthetic_roads(
    north: float, south: float, east: float, west: float, crs: str = "EPSG:4326"
) -> nx.MultiDiGraph:
    """
    Generate a high-fidelity, topologically connected synthetic road network
    spanning the given bounding box. Used when offline or Overpass API fails.
    """
    G = nx.MultiDiGraph(crs=crs)
    grid_rows, grid_cols = 10, 10
    lats = np.linspace(south + 0.012, north - 0.012, grid_rows)
    lons = np.linspace(west + 0.012, east - 0.012, grid_cols)

    node_matrix: Dict[Tuple[int, int], int] = {}
    node_id = 1

    # Place nodes
    for r in range(grid_rows):
        for c in range(grid_cols):
            nid = node_id
            node_matrix[(r, c)] = nid
            G.add_node(
                nid,
                x=float(lons[c]),
                y=float(lats[r]),
                osmid=nid,
            )
            node_id += 1

    # Connect edges
    for r in range(grid_rows):
        for c in range(grid_cols):
            u = node_matrix[(r, c)]
            # West-East road connection
            if c + 1 < grid_cols:
                v = node_matrix[(r, c + 1)]
                u_x, u_y = G.nodes[u]["x"], G.nodes[u]["y"]
                v_x, v_y = G.nodes[v]["x"], G.nodes[v]["y"]
                geom = LineString([(u_x, u_y), (v_x, v_y)])

                # Approximate length in meters
                dy = (v_y - u_y) * 111_139.0
                dx = (v_x - u_x) * 111_139.0 * math.cos(math.radians((u_y + v_y) / 2.0))
                length_m = float(math.sqrt(dx * dx + dy * dy))

                is_bridge = (c == 4 or c == 5) and (r in (2, 5, 8))
                hw_type = "primary" if r == 5 else ("secondary" if r in (2, 8) else "residential")

                edge_attrs = {
                    "geometry": geom,
                    "highway": hw_type,
                    "length": length_m,
                    "oneway": False,
                    "bridge": "yes" if is_bridge else "no",
                    "name": f"East-West Link {r}",
                }
                G.add_edge(u, v, 0, **edge_attrs)
                G.add_edge(v, u, 0, **edge_attrs)

            # South-North road connection
            if r + 1 < grid_rows:
                v = node_matrix[(r + 1, c)]
                u_x, u_y = G.nodes[u]["x"], G.nodes[u]["y"]
                v_x, v_y = G.nodes[v]["x"], G.nodes[v]["y"]
                geom = LineString([(u_x, u_y), (v_x, v_y)])

                dy = (v_y - u_y) * 111_139.0
                dx = (v_x - u_x) * 111_139.0 * math.cos(math.radians((u_y + v_y) / 2.0))
                length_m = float(math.sqrt(dx * dx + dy * dy))

                # Indus Highway corridor (approx c = 4 or 5)
                hw_type = "primary" if c in (4, 5) else ("secondary" if c in (2, 8) else "residential")

                edge_attrs = {
                    "geometry": geom,
                    "highway": hw_type,
                    "length": length_m,
                    "oneway": False,
                    "bridge": "no",
                    "name": f"North-South Corridor {c}",
                }
                G.add_edge(u, v, 0, **edge_attrs)
                G.add_edge(v, u, 0, **edge_attrs)

    return G


def load_roads(
    bbox: Tuple[float, float, float, float],
    network_type: str = "drive",
    cache_path: Optional[Union[str, Path, os.PathLike]] = None,
) -> nx.MultiDiGraph:
    """
    Fetch road network from OpenStreetMap within bbox or load from cached file.

    If network connection fails or Overpass times out, provides a graceful fallback
    to prevent pipeline interruptions in offline disaster response environments.

    Args:
        bbox: Bounding box tuple (north, south, east, west) in WGS84 degrees.
        network_type: OSMnx network type ('drive', 'all', 'walk', etc.).
        cache_path: Optional path to GraphML cache file.

    Returns:
        nx.MultiDiGraph representing the road network with coordinate attributes.
    """
    north, south, east, west = _normalize_bbox(bbox)

    # 1. Check cache first
    if cache_path is not None:
        c_path = Path(cache_path)
        if c_path.exists():
            try:
                import osmnx as ox
                logger.info(f"Loading road network from cache: {c_path}")
                G = ox.load_graphml(c_path)
                return G
            except Exception as e:
                logger.warning(f"Failed to load cached GraphML ({e}), attempting query/generation...")

    # 2. Attempt online download via OSMnx
    try:
        import osmnx as ox
        ox.settings.requests_timeout = 6
        ox_bbox = _bbox_to_osmnx(north, south, east, west)
        logger.info(f"Fetching road network from OSM for bbox={ox_bbox}...")
        G = ox.graph_from_bbox(
            bbox=ox_bbox,
            network_type=network_type,
            simplify=True,
        )
        if G is not None and len(G.nodes) > 0:
            if cache_path is not None:
                c_path = Path(cache_path)
                c_path.parent.mkdir(parents=True, exist_ok=True)
                ox.save_graphml(G, c_path)
            return G
    except Exception as e:
        logger.warning(f"OSMnx Overpass query failed or network offline ({e}).")

    # 3. Check alternative local cache locations before generating synthetic
    fallback_cache_candidates = [
        Path("data/demo_sindh/roads.graphml"),
        Path("../data/demo_sindh/roads.graphml"),
    ]
    for candidate in fallback_cache_candidates:
        if candidate.exists():
            try:
                import osmnx as ox
                logger.info(f"Using existing offline demo road network: {candidate}")
                return ox.load_graphml(candidate)
            except Exception:
                pass

    # 4. Graceful synthetic fallback
    logger.info("Generating high-fidelity synthetic road network fallback for bounding box...")
    G = _generate_synthetic_roads(north, south, east, west)

    if cache_path is not None:
        try:
            import osmnx as ox
            c_path = Path(cache_path)
            c_path.parent.mkdir(parents=True, exist_ok=True)
            ox.save_graphml(G, c_path)
        except Exception as e:
            logger.warning(f"Could not save fallback graph to cache ({e}).")

    return G


# ----------------------------------------------------------------------------
# 3. Emergency Relief Hubs Ingestion
# ----------------------------------------------------------------------------

def _get_sindh_dadu_fallback_hubs(
    north: float, south: float, east: float, west: float
) -> List[Hub]:
    """Return authentic emergency facilities for Dadu District, Sindh (2022 flood)."""
    return [
        Hub(lon=67.7785, lat=26.7342, name="DHQ Civil Hospital & Trauma Center Dadu", type="hospital"),
        Hub(lon=67.7650, lat=26.7280, name="Pakistan Red Crescent Relief Hub Dadu", type="emergency_center"),
        Hub(lon=67.7720, lat=26.7380, name="Rescue 1122 Sindh Emergency Station", type="fire_station"),
        Hub(lon=67.7200, lat=26.7910, name="Rural Health Center Makhdoom Bilawal", type="clinic"),
        Hub(lon=67.8105, lat=26.6748, name="Basic Health Unit Pir Goth Clinic", type="clinic"),
        Hub(lon=67.7560, lat=26.6860, name="Rural Health Post Khudabad", type="clinic"),
    ]


def _generate_generic_fallback_hubs(
    north: float, south: float, east: float, west: float
) -> List[Hub]:
    """Generate realistic emergency hubs positioned within the bounding box."""
    c_lon = (west + east) / 2.0
    c_lat = (south + north) / 2.0
    d_lon = (east - west) * 0.25
    d_lat = (north - south) * 0.25

    return [
        Hub(lon=float(c_lon), lat=float(c_lat), name="District Central Emergency Hospital", type="hospital"),
        Hub(lon=float(c_lon - d_lon), lat=float(c_lat - d_lat), name="Western Relief & Evacuation Hub", type="emergency_center"),
        Hub(lon=float(c_lon + d_lon), lat=float(c_lat + d_lat), name="Eastern Rapid Response Station", type="fire_station"),
        Hub(lon=float(c_lon - d_lon), lat=float(c_lat + d_lat), name="Northern Community Health Clinic", type="clinic"),
        Hub(lon=float(c_lon + d_lon), lat=float(c_lat - d_lat), name="Southern Emergency First Aid Post", type="clinic"),
    ]


def load_hubs(
    bbox: Tuple[float, float, float, float],
    G: Optional[nx.Graph] = None,
    cache_path: Optional[Union[str, Path, os.PathLike]] = None,
    return_node_ids: bool = False,
) -> Union[List[Hub], List[int]]:
    """
    Fetch emergency medical hubs, hospitals, and rescue centers within bbox.

    Queries OpenStreetMap Overpass API via OSMnx, with resilient caching and
    fallback mechanisms.

    Args:
        bbox: Bounding box tuple (north, south, east, west) in WGS84 degrees.
        G: Optional road network graph. If provided and return_node_ids=True,
           snaps hubs to the nearest road network node IDs.
        cache_path: Optional path to cached GeoJSON or JSON file.
        return_node_ids: If True and G is provided, returns snapped node IDs.

    Returns:
        List of Hub instances: [(lon, lat, name, type)], or list of node IDs if
        return_node_ids=True and G is provided.
    """
    north, south, east, west = _normalize_bbox(bbox)
    hubs: List[Hub] = []

    # 1. Check cache file if provided
    if cache_path is not None:
        c_path = Path(cache_path)
        if c_path.exists():
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "features" in data:
                    for f in data["features"]:
                        coords = f["geometry"]["coordinates"]
                        props = f.get("properties", {})
                        hubs.append(
                            Hub(
                                lon=float(coords[0]),
                                lat=float(coords[1]),
                                name=str(props.get("name", "Emergency Medical Hub")),
                                type=str(props.get("amenity", props.get("type", "hospital"))),
                            )
                        )
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, (list, tuple)) and len(item) >= 4:
                            hubs.append(Hub(lon=float(item[0]), lat=float(item[1]), name=str(item[2]), type=str(item[3])))
                        elif isinstance(item, dict):
                            hubs.append(
                                Hub(
                                    lon=float(item["lon"]),
                                    lat=float(item["lat"]),
                                    name=str(item.get("name", "Emergency Hub")),
                                    type=str(item.get("type", "hospital")),
                                )
                            )
                if hubs:
                    logger.info(f"Loaded {len(hubs)} hubs from cache: {c_path}")
            except Exception as e:
                logger.warning(f"Failed to parse hubs cache ({e}), proceeding with query...")

    # 2. Query OSMnx / Overpass if not loaded from cache
    if not hubs:
        try:
            import osmnx as ox
            ox.settings.requests_timeout = 6
            ox_bbox = _bbox_to_osmnx(north, south, east, west)
            tags = {
                "amenity": ["hospital", "clinic", "doctors", "fire_station", "emergency_service"],
                "healthcare": ["hospital", "clinic", "centre"],
            }
            gdf = ox.features_from_bbox(bbox=ox_bbox, tags=tags)
            if gdf is not None and not gdf.empty:
                for idx, row in gdf.iterrows():
                    geom = row.geometry
                    pt = geom.centroid if hasattr(geom, "centroid") else geom
                    name = row.get("name")
                    if name is None or (isinstance(name, float) and math.isnan(name)):
                        name = f"Medical Center {idx}"
                    hub_type = row.get("amenity") or row.get("healthcare") or "hospital"
                    hubs.append(
                        Hub(
                            lon=float(pt.x),
                            lat=float(pt.y),
                            name=str(name),
                            type=str(hub_type),
                        )
                    )
        except Exception as e:
            logger.warning(f"OSMnx emergency amenities query failed or offline ({e}).")

    # 3. Fallback hubs
    if not hubs:
        # Check if coordinates correspond to Sindh, Pakistan
        if 24.0 <= south <= 28.5 and 66.0 <= west <= 70.0:
            hubs = _get_sindh_dadu_fallback_hubs(north, south, east, west)
        else:
            hubs = _generate_generic_fallback_hubs(north, south, east, west)

    # 4. Optional snapping to road graph nodes
    if return_node_ids and G is not None:
        from src.core_logic import snap_to_nodes
        coords = [(h.lon, h.lat) for h in hubs]
        return snap_to_nodes(G, coords)

    return hubs


# ----------------------------------------------------------------------------
# 4. Population Density Ingestion & Synthetic Generation
# ----------------------------------------------------------------------------

def load_population(
    bbox: Tuple[float, float, float, float],
    pop_raster_path: Optional[Union[str, Path, os.PathLike]] = None,
    shape: Tuple[int, int] = (400, 400),
) -> Tuple[np.ndarray, Affine, Any, Dict[str, Any]]:
    """
    Load a WorldPop GeoTIFF clipped to bbox, or generate a realistic population density raster.

    Args:
        bbox: Bounding box tuple (north, south, east, west) in WGS84 degrees.
        pop_raster_path: Optional path to WorldPop GeoTIFF.
        shape: Output raster shape (height, width) when generating synthetic raster.

    Returns:
        (pop_arr, transform, crs, meta)
        - pop_arr: 2D numpy float32 array of population counts/density per pixel.
        - transform: Affine georeferencing transform.
        - crs: Coordinate Reference System (EPSG:4326).
        - meta: Rasterio metadata dictionary.
    """
    north, south, east, west = _normalize_bbox(bbox)

    # 1. Read existing WorldPop GeoTIFF if provided
    if pop_raster_path is not None:
        p_path = Path(pop_raster_path)
        if p_path.exists():
            try:
                with rasterio.open(p_path) as src:
                    # If whole raster fits or needs windowing
                    pop_arr = src.read(1).astype(np.float32)
                    transform = src.transform
                    crs = src.crs or CRS.from_epsg(4326)
                    meta = src.meta.copy()

                    if src.nodata is not None and not np.isnan(src.nodata):
                        pop_arr = np.where(pop_arr == src.nodata, 0.0, pop_arr)
                    pop_arr = np.nan_to_num(pop_arr, nan=0.0, posinf=0.0, neginf=0.0)

                    meta.update(
                        count=1,
                        dtype="float32",
                        crs=crs,
                        transform=transform,
                    )
                    return pop_arr, transform, crs, meta
            except Exception as e:
                logger.warning(f"Error reading population raster {p_path}: {e}. Generating synthetic density...")

    # 2. Generate realistic WorldPop-style population density raster
    h, w = shape
    transform = from_bounds(west, south, east, north, w, h)
    crs = CRS.from_epsg(4326)

    # Base agrarian rural dispersion (0.1 to 2.0 persons/pixel)
    rng = np.random.default_rng(seed=42)
    pop_arr = rng.gamma(shape=0.6, scale=1.5, size=(h, w)).astype(np.float32)

    # Define high-density settlements and rural villages
    settlements = [
        # (center_y_ratio, center_x_ratio, radius_px, peak_population)
        (0.55, 0.48, 45, 320.0),  # Dadu District Urban Core
        (0.25, 0.22, 28, 140.0),  # Makhdoom Bilawal Village Cluster
        (0.82, 0.65, 25, 110.0),  # Pir Goth Agricultural Settlement
        (0.78, 0.38, 22, 95.0),   # Khudabad Historic Settlement
        (0.42, 0.15, 20, 80.0),   # Phulji Station Rural Deh
        (0.35, 0.72, 18, 75.0),   # Indus Riparian Village Cluster
        (0.68, 0.85, 16, 65.0),   # Floodplain Hamlet
    ]

    y_grid, x_grid = np.ogrid[:h, :w]
    for cy_r, cx_r, r_px, peak in settlements:
        cy = int(h * cy_r)
        cx = int(w * cx_r)
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        blob = peak * np.exp(-dist_sq / (2.0 * (r_px / 2.2) ** 2))
        pop_arr += blob.astype(np.float32)

    # Ensure non-negative and finite
    pop_arr = np.clip(pop_arr, 0.0, None)

    meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "lzw",
    }

    return pop_arr, transform, crs, meta


# ----------------------------------------------------------------------------
# 5. Self-Contained 20x20 km Demo Disaster Generator (Dadu, Sindh)
# ----------------------------------------------------------------------------

def generate_demo_event(
    output_dir: Union[str, Path, os.PathLike] = "data/demo_sindh",
    force: bool = False,
) -> Dict[str, Any]:
    """
    Generate a fully self-contained, realistic 20x20 km flood disaster dataset
    for Dadu District, Sindh, Pakistan (2022 Floods).

    Outputs generated:
      - pre_sar.tif: Pre-flood Sentinel-1 SAR (Indus River channel, dry agricultural fields, towns)
      - post_sar.tif: Post-flood Sentinel-1 SAR (flooded floodplains, breached levee, submerged roads)
      - roads.graphml: Road network with bridges and highway classifications
      - roads.geojson: Vectorized roads for Folium/Leaflet visualization
      - hubs.geojson / hubs.json: Emergency medical centers and relief dispatch hubs
      - population.tif: WorldPop-style population density GeoTIFF

    Args:
        output_dir: Directory where datasets will be saved.
        force: If True, regenerates existing files.

    Returns:
        Dictionary mapping resource names to their file paths and bounding box.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dadu District 20x20 km bounding box
    # South: 26.65 N, North: 26.85 N, West: 67.68 E, East: 67.88 E
    north = 26.85
    south = 26.65
    east = 67.88
    west = 67.68
    bbox = (north, south, east, west)

    # File paths
    pre_path = out_dir / "pre_sar.tif"
    post_path = out_dir / "post_sar.tif"
    roads_graph_path = out_dir / "roads.graphml"
    roads_geojson_path = out_dir / "roads.geojson"
    hubs_json_path = out_dir / "hubs.json"
    hubs_geojson_path = out_dir / "hubs.geojson"
    pop_tif_path = out_dir / "population.tif"

    required_files = [
        pre_path,
        post_path,
        roads_graph_path,
        roads_geojson_path,
        hubs_json_path,
        hubs_geojson_path,
        pop_tif_path,
    ]

    # Return cached paths if already generated
    if not force and all(p.exists() for p in required_files):
        logger.info(f"Demo event files already present in {out_dir.resolve()}.")
        return {
            "pre_sar": str(pre_path),
            "post_sar": str(post_path),
            "roads_graph": str(roads_graph_path),
            "roads_graphml": str(roads_graph_path),
            "roads_geojson": str(roads_geojson_path),
            "hubs": str(hubs_json_path),
            "hubs_json": str(hubs_json_path),
            "hubs_geojson": str(hubs_geojson_path),
            "population": str(pop_tif_path),
            "population_tif": str(pop_tif_path),
            "bbox": bbox,
        }

    logger.info(f"Generating realistic Dadu, Sindh flood disaster dataset in {out_dir.resolve()}...")

    h, w = 500, 500  # ~40m pixel resolution across 20 km
    transform = from_bounds(west, south, east, north, w, h)
    crs = CRS.from_epsg(4326)
    rng = np.random.default_rng(seed=101)

    # ---------------------------------------------------------
    # A. Synthesize Sentinel-1 SAR Backscatter (Linear Intensity)
    # ---------------------------------------------------------
    # Physical backscatter calibration:
    #   - Dry soil / vegetation: ~ -11.5 dB (linear ~ 0.071)
    #   - Water (specular calm): ~ -22.5 dB (linear ~ 0.0056)
    #   - Urban structures:      ~ -1.0 dB  (linear ~ 0.794)
    # Realistic SAR multiplicative speckle is modelled using Gamma distribution.

    pre_db = rng.normal(loc=-11.5, scale=1.5, size=(h, w)).astype(np.float32)
    post_db = pre_db.copy() + rng.normal(loc=0.0, scale=0.6, size=(h, w)).astype(np.float32)

    # 1. Indus River permanent channel (winding on eastern side: ~ 67.83 to 67.86 E)
    y_coords = np.arange(h)
    river_center_x = (w * 0.78 + 35 * np.sin(y_coords / 50.0) + 12 * np.cos(y_coords / 22.0)).astype(int)
    for i in range(h):
        rcx = river_center_x[i]
        x0 = max(0, rcx - 7)
        x1 = min(w, rcx + 8)
        # Permanent river is dark in BOTH pre and post (-23 dB)
        pre_db[i, x0:x1] = rng.normal(-23.0, 0.8, size=x1 - x0)
        post_db[i, x0:x1] = rng.normal(-23.2, 0.8, size=x1 - x0)

    # 2. Main Nara Valley (MNV) drainage channel on western flank (~ 67.71 E)
    canal_center_x = (w * 0.16 + 10 * np.sin(y_coords / 70.0)).astype(int)
    for i in range(h):
        ccx = canal_center_x[i]
        x0 = max(0, ccx - 3)
        x1 = min(w, ccx + 3)
        pre_db[i, x0:x1] = rng.normal(-22.0, 0.8, size=x1 - x0)
        post_db[i, x0:x1] = rng.normal(-22.2, 0.8, size=x1 - x0)

    # 3. Towns and settlements (bright corner reflectors in both pre and post)
    settlement_locs = [
        (int(h * 0.55), int(w * 0.48), 24),  # Dadu City
        (int(h * 0.25), int(w * 0.22), 14),  # Makhdoom Bilawal
        (int(h * 0.82), int(w * 0.65), 12),  # Pir Goth
        (int(h * 0.78), int(w * 0.38), 12),  # Khudabad
        (int(h * 0.42), int(w * 0.15), 10),  # Phulji Station
    ]
    y_grid, x_grid = np.ogrid[:h, :w]
    for sy, sx, srad in settlement_locs:
        dist_sq = (x_grid - sx) ** 2 + (y_grid - sy) ** 2
        urban_mask = dist_sq <= (srad ** 2)
        # Urban double-bounce backscatter (-2 to +1 dB)
        urban_db = rng.normal(-1.5, 1.2, size=h * w).reshape(h, w).astype(np.float32)
        pre_db[urban_mask] = urban_db[urban_mask]
        post_db[urban_mask] = urban_db[urban_mask]

    # 4. 2022 Flood Inundation Zones
    # Massive Indus overflow inundating eastern agricultural basin
    for i in range(80, 480):
        rcx = river_center_x[i]
        # Inundation spreads westward from the Indus towards N-55 highway
        spread = int(120 + 40 * np.sin(i / 40.0) + 15 * np.cos(i / 15.0))
        f_start = max(0, rcx - spread)
        f_end = min(w, rcx - 2)
        if f_end > f_start:
            flood_noise = rng.normal(-22.5, 1.0, size=f_end - f_start).astype(np.float32)
            # Water covers fields except elevated urban mounds
            sub_mask = ~(urban_mask[i, f_start:f_end])
            post_db[i, f_start:f_end][sub_mask] = flood_noise[sub_mask]

    # Western breach around MNV drain / Makhdoom Bilawal depression
    for i in range(90, 240):
        ccx = canal_center_x[i]
        spread_west = int(35 + 20 * np.sin(i / 25.0))
        w0 = max(0, ccx - 10)
        w1 = min(w, ccx + spread_west)
        sub_mask = ~(urban_mask[i, w0:w1])
        flood_noise = rng.normal(-21.8, 1.0, size=w1 - w0).astype(np.float32)
        post_db[i, w0:w1][sub_mask] = flood_noise[sub_mask]

    # Convert dB to linear backscatter: linear = 10^(dB / 10)
    pre_linear = (10.0 ** (pre_db / 10.0)).astype(np.float32)
    post_linear = (10.0 ** (post_db / 10.0)).astype(np.float32)

    # Save SAR GeoTIFFs
    sar_meta = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }
    with rasterio.open(pre_path, "w", **sar_meta) as dst:
        dst.write(pre_linear, 1)
        dst.set_band_description(1, "VV Linear Backscatter (Pre-Flood)")

    with rasterio.open(post_path, "w", **sar_meta) as dst:
        dst.write(post_linear, 1)
        dst.set_band_description(1, "VV Linear Backscatter (Post-Flood)")

    # ---------------------------------------------------------
    # B. Synthesize Road Network (NetworkX MultiDiGraph)
    # ---------------------------------------------------------
    G = nx.MultiDiGraph(crs="EPSG:4326")
    grid_rows, grid_cols = 10, 10
    lats = np.linspace(south + 0.015, north - 0.015, grid_rows)
    lons = np.linspace(west + 0.015, east - 0.015, grid_cols)

    node_matrix: Dict[Tuple[int, int], int] = {}
    node_id = 1
    for r in range(grid_rows):
        for c in range(grid_cols):
            nid = node_id
            node_matrix[(r, c)] = nid
            G.add_node(nid, x=float(lons[c]), y=float(lats[r]), osmid=nid)
            node_id += 1

    # Connect grid edges
    for r in range(grid_rows):
        for c in range(grid_cols):
            u = node_matrix[(r, c)]
            # Eastbound connection
            if c + 1 < grid_cols:
                v = node_matrix[(r, c + 1)]
                geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
                is_bridge = (c == 6 or c == 7) and (r in (2, 5, 8))
                hw_type = "primary" if r == 5 else ("secondary" if r in (2, 8) else "residential")
                length_m = float(geom.length * 111_139.0)

                d = {
                    "geometry": geom,
                    "highway": hw_type,
                    "length": length_m,
                    "bridge": "yes" if is_bridge else "no",
                    "oneway": False,
                    "name": f"Road {r}-{c}",
                }
                G.add_edge(u, v, 0, **d)
                G.add_edge(v, u, 0, **d)

            # Northbound connection
            if r + 1 < grid_rows:
                v = node_matrix[(r + 1, c)]
                geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])])
                # N-55 Indus Highway along column 4
                hw_type = "primary" if c == 4 else ("secondary" if c in (2, 7) else "residential")
                length_m = float(geom.length * 111_139.0)

                d = {
                    "geometry": geom,
                    "highway": hw_type,
                    "length": length_m,
                    "bridge": "no",
                    "oneway": False,
                    "name": f"Corridor {r}-{c}",
                }
                G.add_edge(u, v, 0, **d)
                G.add_edge(v, u, 0, **d)

    # Save GraphML
    import osmnx as ox
    ox.save_graphml(G, roads_graph_path)

    # Save Road GeoJSON
    road_features = []
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry", LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]))
        road_features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "u": u,
                "v": v,
                "highway": d.get("highway", "unclassified"),
                "bridge": d.get("bridge", "no"),
                "length": d.get("length", 1000.0),
                "name": d.get("name", "Road"),
            },
        })
    with open(roads_geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": road_features}, f, indent=2)

    # ---------------------------------------------------------
    # C. Emergency Hubs (Hospitals & Relief Centers)
    # ---------------------------------------------------------
    # Facilities located in protected urban center (Dadu) and district wings
    hub_facilities = [
        {
            "name": "DHQ Civil Hospital & Trauma Center Dadu",
            "node_id": node_matrix[(5, 4)],
            "lon": float(G.nodes[node_matrix[(5, 4)]]["x"]),
            "lat": float(G.nodes[node_matrix[(5, 4)]]["y"]),
            "type": "hospital",
            "beds": 250,
        },
        {
            "name": "Pakistan Red Crescent Disaster Relief HQ",
            "node_id": node_matrix[(5, 3)],
            "lon": float(G.nodes[node_matrix[(5, 3)]]["x"]),
            "lat": float(G.nodes[node_matrix[(5, 3)]]["y"]),
            "type": "emergency_center",
            "beds": 100,
        },
        {
            "name": "Rescue 1122 Sindh Central Station",
            "node_id": node_matrix[(6, 4)],
            "lon": float(G.nodes[node_matrix[(6, 4)]]["x"]),
            "lat": float(G.nodes[node_matrix[(6, 4)]]["y"]),
            "type": "fire_station",
            "beds": 30,
        },
        {
            "name": "Rural Health Center Makhdoom Bilawal",
            "node_id": node_matrix[(7, 2)],
            "lon": float(G.nodes[node_matrix[(7, 2)]]["x"]),
            "lat": float(G.nodes[node_matrix[(7, 2)]]["y"]),
            "type": "clinic",
            "beds": 25,
        },
        {
            "name": "Basic Health Unit Pir Goth Medical Post",
            "node_id": node_matrix[(2, 6)],
            "lon": float(G.nodes[node_matrix[(2, 6)]]["x"]),
            "lat": float(G.nodes[node_matrix[(2, 6)]]["y"]),
            "type": "clinic",
            "beds": 20,
        },
    ]

    # Save hubs JSON
    with open(hubs_json_path, "w", encoding="utf-8") as f:
        json.dump(hub_facilities, f, indent=2)

    # Save hubs GeoJSON
    hub_features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [h_item["lon"], h_item["lat"]]},
            "properties": {
                "name": h_item["name"],
                "node_id": h_item["node_id"],
                "amenity": h_item["type"],
                "beds": h_item["beds"],
            },
        }
        for h_item in hub_facilities
    ]
    with open(hubs_geojson_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": hub_features}, f, indent=2)

    # ---------------------------------------------------------
    # D. WorldPop Population Density GeoTIFF
    # ---------------------------------------------------------
    pop_arr, pop_transform, pop_crs, pop_meta = load_population(
        bbox=bbox, shape=(h, w)
    )

    # Zero out water channels in population raster
    for i in range(h):
        rcx = river_center_x[i]
        pop_arr[i, max(0, rcx - 7):min(w, rcx + 8)] = 0.0

    # Ensure villages in the flooded sector have significant vulnerable populations
    pop_arr[int(h * 0.20):int(h * 0.30), int(w * 0.18):int(w * 0.26)] += 420.0
    pop_arr[int(h * 0.75):int(h * 0.88), int(w * 0.60):int(w * 0.70)] += 380.0

    with rasterio.open(pop_tif_path, "w", **pop_meta) as dst:
        dst.write(pop_arr, 1)

    logger.info(f"Successfully generated Dadu, Sindh demo event in: {out_dir.resolve()}")

    return {
        "pre_sar": str(pre_path),
        "post_sar": str(post_path),
        "roads_graph": str(roads_graph_path),
        "roads_graphml": str(roads_graph_path),
        "roads_geojson": str(roads_geojson_path),
        "hubs": str(hubs_json_path),
        "hubs_json": str(hubs_json_path),
        "hubs_geojson": str(hubs_geojson_path),
        "population": str(pop_tif_path),
        "population_tif": str(pop_tif_path),
        "bbox": bbox,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    paths = generate_demo_event()
    print("\nGenerated Demo Event Dataset:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
