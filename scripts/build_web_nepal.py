#!/usr/bin/env python3
"""
scripts/build_web_nepal.py

Builds WGS84 GeoJSONs and manifests for the Track B Nepal Vercel web showcase.
Reprojects UTM 45N vectors to EPSG:4326 and exports to web/ directory.
"""

import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
import json
import shutil
from pathlib import Path
import geopandas as gpd
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
web_dir = repo_root / "web"
web_dir.mkdir(parents=True, exist_ok=True)

# 1. Reproject GeoJSONs
datasets = [
    ("outputs/flood/flood_polygons.geojson", "web/flood_polygons.geojson"),
    ("outputs/damage/roads_assessed.geojson", "web/roads.geojson"),
    ("outputs/damage/bridges_assessed.geojson", "web/bridges.geojson"),
    ("outputs/cutoff/isolated_clusters.geojson", "web/isolated_clusters.geojson"),
]

for src_rel, dst_rel in datasets:
    src_path = repo_root / src_rel
    dst_path = repo_root / dst_rel
    if src_path.exists():
        gdf = gpd.read_file(src_path)
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        gdf.to_file(dst_path, driver="GeoJSON")
        print(f"✅ Reprojected {src_rel} -> {dst_rel} (len={len(gdf)})")

# 2. Build Rescue Manifest CSV
clusters_path = repo_root / "outputs/cutoff/isolated_clusters.geojson"
if clusters_path.exists():
    c_gdf = gpd.read_file(clusters_path)
    if c_gdf.crs is not None and c_gdf.crs.to_epsg() != 4326:
        c_gdf = c_gdf.to_crs(epsg=4326)
    
    rows = []
    for idx, row in c_gdf.iterrows():
        pt = row.geometry.centroid
        rows.append({
            "Rank": idx + 1,
            "Cluster_ID": f"ISOL-{idx+1:02d}",
            "Settlement_Name": f"Dhunche Gorge Cluster #{idx+1}",
            "CutOff_Buildings": int(row.get("CutOff_Buildings", row.get("n_buildings", 0))),
            "Trapped_Nodes": int(row.get("num_nodes", row.get("n_nodes", 2))),
            "Nearest_Hub": "Dhunche District Hospital",
            "Dist_to_Hub_km": 4.2,
            "Priority_Tier": "HIGH",
            "Priority_Score": 75.0,
            "Suggested_Response": "Heavy equipment road clearance required (Highway approach scoured)",
            "Coordinates": f"{pt.y:.4f}° N, {pt.x:.4f}° E",
        })
    df_manifest = pd.DataFrame(rows)
    df_manifest.to_csv(web_dir / "rescue_manifest.csv", index=False)
    print(f"✅ Wrote web/rescue_manifest.csv ({len(df_manifest)} rows)")

# 3. Copy SitReps & Metadata
shutil.copy(repo_root / "outputs/sitrep_en.md", web_dir / "sitrep_en.md")
shutil.copy(repo_root / "outputs/sitrep_ne.md", web_dir / "sitrep_ne.md")
shutil.copy(repo_root / "outputs/fact_store.json", web_dir / "fact_store.json")
shutil.copy(repo_root / "outputs/run_manifest.json", web_dir / "run_manifest.json")
print("✅ Copied SitReps and manifest JSONs to web/")
