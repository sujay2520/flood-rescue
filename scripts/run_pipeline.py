"""
scripts/run_pipeline.py - Headless CLI Runner for End-to-End Flood Rescue Analysis.

Executes the entire flood response pipeline:
  1. Ingestion: Load or generate SAR, OSM roads, hubs, and population.
  2. Flood Masking: Run SAR change detection (baseline) or U-Net model.
  3. Road Tagging: Overlay flood mask on OSM road graph and flag bridges.
  4. Isolation Analysis: Graph cut-off detection (before vs after flood).
  5. Population & Ranking: Cluster isolated nodes, sum population, compute rescue priority.
  6. Output Generation: Export GeoTIFFs, Road GeoJSON, Rescue Manifest CSV & GeoJSON.
"""

import argparse
import os
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingest import load_sar, generate_demo_event
from src.flood import create_baseline_mask, save_geotiff
from src.roads import tag_road_network, assess_bridges, roads_to_geojson, compute_road_stats
from src.isolation import identify_cut_off_settlements, isolation_to_geojson
from src.priority import calculate_priority, generate_rescue_manifest, export_rescue_plan


def run_pipeline(
    pre_path: str,
    post_path: str,
    roads_graph_path: str,
    hubs_path: str,
    pop_path: str,
    output_dir: str = "output",
    water_db: float = -18.0,
    drop_db: float = 3.0,
    road_flood_frac: float = 0.3,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("FLOOD RESCUE AI - END-TO-END RAPID ASSESSMENT PIPELINE")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Ingestion
    # ---------------------------------------------------------
    print("\n[PHASE 0/1] Loading Sentinel-1 SAR imagery...")
    pre_sar, post_sar, transform, crs, meta = load_sar(pre_path, post_path)
    print(f" -> SAR raster size: {pre_sar.shape} | CRS: {crs}")

    # ---------------------------------------------------------
    # 2. Flood Detection (Phase 1)
    # ---------------------------------------------------------
    print("\n[PHASE 1] Generating SAR Change Detection Flood Mask...")
    flood_mask = create_baseline_mask(
        pre_sar, post_sar, is_db=False, water_db=water_db, drop_db=drop_db
    )
    from src.flood import get_flood_summary
    flood_stats = get_flood_summary(flood_mask, transform=transform, crs=crs)
    flood_area_km2 = flood_stats["flood_area_km2"]
    flooded_pixels = flood_stats["flood_pixels"]
    print(f" -> Flooded Area: {flood_area_km2:.2f} km² ({flooded_pixels:,} pixels, {flood_stats['flood_percentage']:.1f}% of AOI)")

    mask_tif_path = os.path.join(output_dir, "flood_mask.tif")
    save_geotiff(flood_mask.astype("uint8"), transform, crs, mask_tif_path)
    print(f" -> Saved flood mask GeoTIFF: {mask_tif_path}")

    # ---------------------------------------------------------
    # 3. Infrastructure & Bridge Analysis (Phase 2)
    # ---------------------------------------------------------
    print("\n[PHASE 2] Intersecting with Road Network & Tagging Bridges...")
    import osmnx as ox
    if os.path.exists(roads_graph_path):
        G = ox.load_graphml(roads_graph_path)
    else:
        raise FileNotFoundError(f"Road graph not found: {roads_graph_path}")

    raster_crs_str = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    G = tag_road_network(G, flood_mask, transform, raster_crs_str, flooded_frac=road_flood_frac)
    G = assess_bridges(G, flood_mask, transform, raster_crs_str)

    road_stats = compute_road_stats(G)
    print(f" -> Total Road Network: {road_stats.get('total_km', 0):.1f} km")
    flooded_pct = road_stats.get('percentage_cut', road_stats.get('flooded_pct', 0.0))
    print(f" -> Submerged/Impassable Roads: {road_stats.get('flooded_km', 0):.1f} km ({flooded_pct:.1f}%)")
    print(f" -> High-Risk / Cut Bridges: {road_stats.get('damaged_bridges_count', 0)}")

    roads_geojson_path = os.path.join(output_dir, "roads_status.geojson")
    roads_to_geojson(G, output_path=roads_geojson_path)
    print(f" -> Saved road status GeoJSON: {roads_geojson_path}")

    # ---------------------------------------------------------
    # 4. "Who is Cut Off" (Phase 3)
    # ---------------------------------------------------------
    print("\n[PHASE 3] Computing Connected Components & Lost Hospital Access...")
    import json
    with open(hubs_path, "r", encoding="utf-8") as f:
        hubs_data = json.load(f)
    
    # Extract hub node IDs or coordinates
    hub_nodes = []
    if "features" in hubs_data:
        from src.core_logic import snap_to_nodes
        coords = [f["geometry"]["coordinates"] for f in hubs_data["features"]]
        hub_nodes = snap_to_nodes(G, coords)
    elif isinstance(hubs_data, list):
        hub_nodes = hubs_data

    # Load Population Raster
    import rasterio
    with rasterio.open(pop_path) as pop_src:
        pop_arr = pop_src.read(1)
        pop_transform = pop_src.transform
        pop_crs = pop_src.crs.to_string() if pop_src.crs else "EPSG:4326"

    clusters, isolated_nodes = identify_cut_off_settlements(
        G, hub_nodes, pop_array=pop_arr, pop_transform=pop_transform, pop_crs=pop_crs
    )
    print(f" -> Newly Cut-Off Road Intersections: {len(isolated_nodes):,}")
    print(f" -> Trapped Communities / Clusters: {len(clusters)}")

    clusters_geojson_path = os.path.join(output_dir, "isolated_settlements.geojson")
    isolation_to_geojson(clusters, output_path=clusters_geojson_path)

    # ---------------------------------------------------------
    # 5. Rescue Priority Dispatch Manifest
    # ---------------------------------------------------------
    print("\n[PHASE 4] Calculating Multi-Factor Rescue Priority Ranking...")
    ranked_clusters = calculate_priority(clusters)
    manifest_df = generate_rescue_manifest(ranked_clusters)

    manifest_csv_path = os.path.join(output_dir, "rescue_manifest.csv")
    manifest_json_path = os.path.join(output_dir, "rescue_manifest.geojson")
    export_rescue_plan(manifest_df, manifest_csv_path, manifest_json_path)

    total_trapped_pop = int(manifest_df["Estimated_Population"].sum()) if not manifest_df.empty else 0
    print(f"\n=======================================================")
    print(f"RAPID ASSESSMENT RESULTS SUMMARY")
    print(f"=======================================================")
    print(f"Total Trapped Population at Risk : {total_trapped_pop:,} people")
    print(f"Critical Priority Clusters       : {len(manifest_df[manifest_df['Priority_Tier'] == 'CRITICAL'])}")
    print(f"High Priority Clusters           : {len(manifest_df[manifest_df['Priority_Tier'] == 'HIGH'])}")
    print(f"Total Flooded Roads              : {road_stats.get('flooded_km', 0):.1f} km")
    print(f"Damaged / Cut Bridges            : {road_stats.get('damaged_bridges_count', 0)}")
    print(f"Rescue Manifest CSV Saved        : {manifest_csv_path}")
    print(f"Pipeline Execution Time          : {time.time() - start_time:.2f} seconds")
    print(f"=======================================================\n")

    if not manifest_df.empty:
        print("TOP 5 RESCUE TARGETS:")
        print(manifest_df[["Rank", "Cluster_ID", "Priority_Tier", "Priority_Score", "Estimated_Population", "Nearest_Hub", "Recommended_Action"]].head(5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run headless Flood Rescue pipeline.")
    parser.add_argument("--demo", action="store_true", help="Generate and run on synthetic demo event (Sindh, Pakistan)")
    parser.add_argument("--pre", type=str, help="Path to Pre-flood SAR GeoTIFF")
    parser.add_argument("--post", type=str, help="Path to Post-flood SAR GeoTIFF")
    parser.add_argument("--roads", type=str, help="Path to Roads GraphML")
    parser.add_argument("--hubs", type=str, help="Path to Hubs GeoJSON")
    parser.add_argument("--pop", type=str, help="Path to Population GeoTIFF")
    parser.add_argument("--water_db", type=float, default=-18.0, help="Water backscatter threshold in dB")
    parser.add_argument("--drop_db", type=float, default=3.0, help="Temporal backscatter drop threshold in dB")
    parser.add_argument("--road_flood_frac", type=float, default=0.3, help="Fraction of road length flooded to classify as impassable")
    parser.add_argument("--output", type=str, default="output", help="Output directory")

    args = parser.parse_args()

    if args.demo or not (args.pre and args.post and args.roads and args.hubs and args.pop):
        print("[*] Running in Self-Contained Demo Mode (Generating Dadu, Sindh flood dataset)...")
        demo_paths = generate_demo_event(output_dir="data/demo_sindh")
        run_pipeline(
            pre_path=demo_paths["pre_sar"],
            post_path=demo_paths["post_sar"],
            roads_graph_path=demo_paths["roads_graph"],
            hubs_path=demo_paths["hubs_geojson"],
            pop_path=demo_paths["population_tif"],
            output_dir=args.output,
            water_db=args.water_db,
            drop_db=args.drop_db,
            road_flood_frac=args.road_flood_frac,
        )
    else:
        run_pipeline(
            pre_path=args.pre,
            post_path=args.post,
            roads_graph_path=args.roads,
            hubs_path=args.hubs,
            pop_path=args.pop,
            output_dir=args.output,
            water_db=args.water_db,
            drop_db=args.drop_db,
            road_flood_frac=args.road_flood_frac,
        )
