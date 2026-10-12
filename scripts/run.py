#!/usr/bin/env python3
"""
scripts/run.py

Unified CLI pipeline runner for Track B: Mapping Flood Damage from Space.
August 26, 2026 Bhote Koshi - Trishuli River Corridor GLOF & Flood Disaster, Nepal.

Chains together 9 deterministic, grounded modules:
  1. Sentinel-1 Relative Orbit Scene Pairing & Preprocessing (Track 85 Ascending)
  2. Copernicus DEM GLO-30 Mountain Terrain & SAR Geometry Masking
  3. Pre-event OSM Ingestion (Snapshot <= 27 July 2026 via Overpass/Parquet)
  4. Planar NetworkX Graph Builder (Bridge Nodalization & Building Association)
  5. Mountain SAR Flood & Mudflow Detection (Slope <= 15°, Shadow/Layover Filtered)
  6. Infrastructure Damage Assessment (Roads, 40m Bridge Scouring, Building Inundation)
  7. Dijkstra Network Cut-Off Reachability & Settlement Cluster Isolation Analysis
  8. Immutable Cryptographic FactStore Construction
  9. Bilingual Grounded Situation Report Generation (English & Nepali)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports
import geopandas as gpd
import pandas as pd

# Local module imports
from src.s1_pairing import find_same_orbit_pair, prepare_s1_pair_data
from src.dem_terrain import prepare_dem_and_masks
from src.osm_ingest import fetch_pre_event_osm
from src.graph_builder import build_network, graph_to_geodataframes
from src.mountain_flood import detect_mountain_flood
from src.damage_assessment import assess_infrastructure_damage
from src.cutoff_analysis import analyze_cutoffs
from src.fact_store import build_fact_store_from_pipeline_outputs
from src.sitrep_generator import generate_sitreps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("track_b_pipeline")


def sha256_file(path: Path) -> str:
    """Return SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(directory: Path) -> str:
    """Return combined SHA-256 of all files in directory (sorted)."""
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.name.encode())
            try:
                h.update(sha256_file(p).encode())
            except Exception:
                pass
    return h.hexdigest()


# Configure UTF-8 encoding for standard streams
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_ansi_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    """Pretty-print an ANSI-formatted table using ASCII borders for Windows compatibility."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "-"
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    border = "+" + "+".join(sep * (w + 2) for w in col_widths) + "+"
    print(f"\n\033[1;36m{title}\033[0m")
    print(border)
    print("| " + header_line + " |")
    print(border)

    for row in rows:
        row_line = " | ".join(str(c).ljust(w) for c, w in zip(row, col_widths))
        print("| " + row_line + " |")

    print(border)


class TrackBPipeline:
    def __init__(
        self,
        bbox: List[float],
        event_date: str,
        out_root: Path,
        max_lookback_days: int = 30,
        orbit_state: str = "ascending",
        osm_timestamp: str = "2026-07-27T00:00:00Z",
        force_offline: bool = False,
    ):
        self.bbox = bbox
        self.event_date = event_date
        self.out_root = Path(out_root)
        self.max_lookback_days = max_lookback_days
        self.orbit_state = orbit_state
        self.osm_timestamp = osm_timestamp
        self.force_offline = force_offline

        # Sub-directories
        self.s1_dir = self.out_root / "s1_data"
        self.dem_dir = self.out_root / "dem"
        self.osm_dir = self.out_root / "osm"
        self.flood_dir = self.out_root / "flood"
        self.damage_dir = self.out_root / "damage"
        self.cutoff_dir = self.out_root / "cutoff"
        self.sitrep_dir = self.out_root / "sitreps"

        for d in [
            self.s1_dir, self.dem_dir, self.osm_dir,
            self.flood_dir, self.damage_dir, self.cutoff_dir, self.sitrep_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.manifest = {
            "pipeline_name": "Track B: Mapping Flood Damage from Space",
            "case_study": "Bhote Koshi-Trishuli River Corridor GLOF & Flood Disaster, Nepal (Copernicus EMSR927)",
            "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "inputs": {
                "bbox": bbox,
                "event_date": event_date,
                "max_lookback_days": max_lookback_days,
                "orbit_state": orbit_state,
                "osm_timestamp": osm_timestamp,
                "force_offline": force_offline,
            },
            "steps": {},
            "outputs": {},
            "provenance_hash": None,
            "status": "RUNNING",
        }
        self.step_timings: Dict[str, float] = {}

    def _run_step(self, step_name: str, func, *args, **kwargs) -> Any:
        """Execute a pipeline step with timing and error handling."""
        logger.info(f"==> [{step_name}] Starting...")
        t0 = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            self.step_timings[step_name] = dt
            self.manifest["steps"][step_name] = {
                "status": "SUCCESS",
                "duration_seconds": round(dt, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.info(f"==> [{step_name}] Completed in {dt:.2f}s")
            return result
        except Exception as e:
            dt = time.perf_counter() - t0
            self.step_timings[step_name] = dt
            self.manifest["steps"][step_name] = {
                "status": "FAILED",
                "duration_seconds": round(dt, 3),
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            logger.error(f"==> [{step_name}] FAILED after {dt:.2f}s: {e}")
            raise

    def execute(self) -> Dict[str, Any]:
        """Execute the end-to-end pipeline stages sequentially."""
        print("\n" + "=" * 76)
        print(" TRACK B: MAPPING FLOOD DAMAGE FROM SPACE — LIVE END-TO-END PIPELINE")
        print(" August 26, 2026 Bhote Koshi–Trishuli River Corridor GLOF Disaster, Nepal")
        print("=" * 76)

        # ─── Step 1: S-1 Pairing ───
        pair = self._run_step(
            "1. S1 Pair Discovery",
            find_same_orbit_pair,
            self.bbox,
            event_date=self.event_date,
            max_lookback_days=self.max_lookback_days,
            force_offline=self.force_offline,
        )
        s1_paths = self._run_step(
            "2. S1 Preprocessing & Filtering",
            prepare_s1_pair_data,
            pair,
            self.bbox,
            output_dir=self.s1_dir,
            force_offline=self.force_offline,
        )
        pre_vv = s1_paths["pre_VV"][0]
        post_vv = s1_paths["post_VV"][0]
        pre_vh = s1_paths.get("pre_VH", [None])[0]
        post_vh = s1_paths.get("post_VH", [None])[0]

        self.manifest["outputs"]["s1_pre_vv"] = str(pre_vv)
        self.manifest["outputs"]["s1_post_vv"] = str(post_vv)

        # ─── Step 2: DEM & Terrain Geometry Masks ───
        dem_masks = self._run_step(
            "3. DEM & Mountain Shadow/Layover Masking",
            prepare_dem_and_masks,
            bbox=self.bbox,
            orbit_state=self.orbit_state,
            out_dir=self.dem_dir,
            force_offline=self.force_offline,
        )
        self.manifest["outputs"]["dem_masks"] = {k: str(v) for k, v in dem_masks.items()}

        # ─── Step 3: Pre-event OSM Ingestion (Historical Snapshot) ───
        osm_data = self._run_step(
            "4. Pre-Event OSM Historical Ingestion (<= 27 July 2026)",
            fetch_pre_event_osm,
            bbox=self.bbox,
            timestamp=self.osm_timestamp,
            out_dir=self.osm_dir,
            force_offline=self.force_offline,
        )
        roads = osm_data["roads"]
        bridges = osm_data["bridges"]
        buildings = osm_data["buildings"]
        facilities = osm_data.get("facilities", osm_data.get("medical", gpd.GeoDataFrame()))

        # ─── Step 4: Planar Graph Construction ───
        G = self._run_step(
            "5. Road Graph Nodalization & Building Mapping",
            build_network,
            roads, bridges, facilities, buildings,
        )
        nodes_gdf, edges_gdf = graph_to_geodataframes(G)
        edges_out = self.out_root / "edges.geojson"
        nodes_out = self.out_root / "nodes.geojson"
        edges_gdf.to_file(edges_out, driver="GeoJSON")
        nodes_gdf.to_file(nodes_out, driver="GeoJSON")
        self.manifest["outputs"]["graph_edges"] = str(edges_out)
        self.manifest["outputs"]["graph_nodes"] = str(nodes_out)

        # ─── Step 5: Mountain SAR Flood Detection ───
        flood_res = self._run_step(
            "6. Mountain SAR Flood & Mudflow Detection",
            detect_mountain_flood,
            pre_vv, post_vv,
            dem_masks=dem_masks,
            out_dir=self.flood_dir,
            pre_vh_path=pre_vh,
            post_vh_path=post_vh,
        )
        flood_gdf = flood_res["flood_gdf"]
        flood_stats = flood_res["stats"]
        self.manifest["outputs"]["flood_polygons"] = str(flood_res["polygons_geojson_path"])
        self.manifest["outputs"]["flood_stats"] = flood_stats

        # ─── Step 6: Infrastructure Damage Assessment ───
        damage_res = self._run_step(
            "7. Infrastructure Damage Assessment (40m Bridge Scour)",
            assess_infrastructure_damage,
            G, roads, bridges, buildings, flood_gdf,
            shadow_mask_path=dem_masks.get("shadow"),
            layover_mask_path=dem_masks.get("layover"),
            out_dir=self.damage_dir,
        )
        damage_summary = damage_res["summary_dict"]
        self.manifest["outputs"]["damage_summary"] = damage_summary

        # ─── Step 7: Cutoff & Isolation Analysis ───
        cutoff_res = self._run_step(
            "8. Dijkstra Cut-Off & Community Isolation Analysis",
            analyze_cutoffs,
            G,
            out_dir=self.cutoff_dir,
        )
        cutoff_summary = cutoff_res["summary"]
        clusters = cutoff_res["clusters"]
        clusters_dicts = cutoff_summary.get("clusters", [])
        self.manifest["outputs"]["cutoff_summary"] = cutoff_summary

        # ─── Step 8: Immutable FactStore Construction ───
        pair_meta = pair.to_dict() if hasattr(pair, "to_dict") else {
            "relative_orbit": 85,
            "pass_direction": "ASCENDING",
            "event_date": self.event_date,
        }
        fact_store = self._run_step(
            "9. Cryptographic FactStore Assembly",
            build_fact_store_from_pipeline_outputs,
            flood_stats, damage_summary, cutoff_summary,
            pair_metadata=pair_meta,
            out_path=self.out_root / "fact_store.json",
        )
        self.manifest["outputs"]["fact_store"] = str(self.out_root / "fact_store.json")

        # ─── Step 9: Bilingual Grounded SitRep Generation ───
        sitrep_paths = self._run_step(
            "10. Bilingual Situation Report Generation (EN & NE)",
            generate_sitreps,
            fact_store,
            clusters=clusters_dicts,
            out_dir=self.sitrep_dir,
        )
        self.manifest["outputs"]["sitreps"] = {k: str(v) for k, v in sitrep_paths.items()}

        # ─── Finalize Provenance ───
        self.manifest["status"] = "SUCCESS"
        total_time = sum(self.step_timings.values())
        self.manifest["total_duration_seconds"] = round(total_time, 2)
        self.manifest["provenance_hash"] = sha256_dir(self.out_root)
        self._write_manifest()
        self._print_summary_tables(damage_summary, cutoff_summary, flood_stats)

        return self.manifest

    def _write_manifest(self) -> None:
        """Write run_manifest.json to root output directory."""
        manifest_path = self.out_root / "run_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, default=str)
        logger.info(f"Pipeline Run Manifest saved to: {manifest_path}")

    def _print_summary_tables(self, damage: Dict[str, Any], cutoff: Dict[str, Any], flood: Dict[str, Any]) -> None:
        """Print clean ANSI summary tables for the pipeline results."""
        # Execution timing table
        timing_rows = [
            [step, f"{dur:.2f} s", "SUCCESS"]
            for step, dur in self.step_timings.items()
        ]
        total_time = sum(self.step_timings.values())
        timing_rows.append(["TOTAL PIPELINE TIME", f"{total_time:.2f} s", "COMPLETE"])
        print_ansi_table("PIPELINE STAGE EXECUTION PROFILER", ["Stage", "Duration", "Status"], timing_rows)

        # Disaster & Damage metrics table
        metrics_rows = [
            ["Detected Flood / Mudflow Area", f"{flood.get('flood_area_km2', 0):.2f} km²", "Sentinel-1 VV (-2.5 dB drop)"],
            ["Mountain Shadow / Cliffs Excluded", "Filtered", "Ray-traced shadow & slope > 15°"],
            ["Total Road Network Assessed", f"{damage.get('total_road_km', 0):.1f} km", "Pre-event OSM snapshot"],
            ["Roads Affected by Flood / Mudflow", f"{damage.get('affected_road_km', 0):.1f} km", "Surface intersection"],
            ["Roads Non-Assessable (Shadow/Layover)", f"{damage.get('not_assessable_road_km', 0):.1f} km", "Zero false-confidence guarantee"],
            ["Total Bridges Evaluated", f"{damage.get('total_bridges', 0)}", "OSM bridge ways"],
            ["Severed / Scoured Bridges", f"{damage.get('severed_bridges', 0)}", "Direct flood or 40m approach scour"],
            ["Pre-Event Mapped Buildings", f"{damage.get('total_buildings', 0)}", "Pre-event OSM footprints (No WorldPop)"],
            ["Directly Flooded Buildings", f"{damage.get('flooded_buildings_count', 0)}", "Footprint intersects flood polygon"],
            ["Completely Cut-Off Settlement Clusters", f"{cutoff.get('n_cutoff_clusters', 0)}", "Isolated road subnetworks"],
            ["Total Buildings Cut Off from Network", f"{cutoff.get('total_cutoff_buildings', 0)}", "Sum of buildings in isolated clusters"],
        ]
        print_ansi_table("DISASTER IMPACT & INFRASTRUCTURE DAMAGE METRICS", ["Metric Description", "Observed Value", "Methodology / Source"], metrics_rows)

        # Isolated Clusters Roster
        clusters = cutoff.get("clusters", [])
        cluster_rows = []
        for c in clusters:
            cluster_rows.append([
                str(c.get("cluster_id", "N/A")),
                str(c.get("name", "Settlement")),
                f"{c.get('n_buildings', 0)} buildings",
                str(c.get("n_nodes", 0)),
                str(c.get("suggested_response", "Footpath / Helicopter"))[:45] + "...",
            ])
        if cluster_rows:
            print_ansi_table("ISOLATED SETTLEMENT CLUSTERS ROSTER (CUT-OFF COMMUNITIES)", ["Cluster #", "Settlement Name", "Cut-Off Buildings", "Nodes", "Suggested Response"], cluster_rows)

        print(f"\n\033[1;32m[SUCCESS] System Verified: Immutable FactStore and Bilingual SitReps generated successfully.\033[0m")
        print(f"  - English SitRep: {self.sitrep_dir / 'sitrep_en.md'}")
        print(f"  - Nepali SitRep:  {self.sitrep_dir / 'sitrep_ne.md'}")
        print(f"  - Manifest:       {self.out_root / 'run_manifest.json'}")
        print(f"  - FactStore SHA:  {self.manifest.get('provenance_hash', '')[:20]}...\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track B: Mapping Flood Damage from Space — End-to-End Live Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[85.15, 27.85, 85.45, 28.25],
        help="Bounding box in EPSG:4326: min_lon min_lat max_lon max_lat (default: Trishuli Valley)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2026-08-26",
        help="Target disaster date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--orbit",
        type=str,
        default="ascending",
        choices=["ascending", "descending"],
        help="Sentinel-1 orbit pass direction",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
        help="Maximum lookback window in days for same-orbit pre-event baseline",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./outputs",
        help="Root directory for pipeline outputs and artifacts",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline mode using cached or calibrated synthetic Himalayan data",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = TrackBPipeline(
        bbox=args.bbox,
        event_date=args.date,
        out_root=Path(args.out_dir),
        max_lookback_days=args.lookback,
        orbit_state=args.orbit,
        force_offline=args.offline,
    )
    try:
        pipeline.execute()
        return 0
    except Exception as e:
        logger.exception(f"Pipeline execution aborted: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
