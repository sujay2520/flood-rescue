"""
FloodRescue AI Package
Autonomous SAR Inundation & Topological Isolation Routing
"""

__version__ = "1.0.0"

from src.core_logic import (
    baseline_flood_mask,
    cluster_isolated,
    find_isolated,
    snap_to_nodes,
    tag_edges,
    to_db,
)
from src.ingest import (
    Hub,
    generate_demo_event,
    load_hubs,
    load_population,
    load_roads,
    load_sar,
)

from src.isolation import identify_cut_off_settlements, isolation_to_geojson
from src.priority import calculate_priority, export_rescue_plan, generate_rescue_manifest
from src.flood import (
    create_baseline_mask,
    save_geotiff,
    mask_to_polygons,
    unet_inference,
    calculate_pixel_area_m2,
    calculate_flood_area_km2,
    calculate_flood_percentage,
    get_flood_summary,
    calculate_flood_stats,
)

__all__ = [
    "to_db",
    "baseline_flood_mask",
    "create_baseline_mask",
    "save_geotiff",
    "mask_to_polygons",
    "unet_inference",
    "calculate_pixel_area_m2",
    "calculate_flood_area_km2",
    "calculate_flood_percentage",
    "get_flood_summary",
    "calculate_flood_stats",
    "tag_edges",
    "snap_to_nodes",
    "find_isolated",
    "cluster_isolated",
    "load_sar",
    "load_roads",
    "load_hubs",
    "load_population",
    "generate_demo_event",
    "Hub",
    "identify_cut_off_settlements",
    "isolation_to_geojson",
    "calculate_priority",
    "generate_rescue_manifest",
    "export_rescue_plan",
]

