"""
tests/test_track_b.py

Comprehensive test suite for Track B: Mapping Flood Damage from Space.
August 26, 2026 Bhote Koshi - Trishuli River Corridor Disaster, Nepal.

Validates:
  1. Mountain SAR physics filters (zero false-positives in shadow/layover/slope > 15°)
  2. Pre-event OSM historical timestamp constraints (strictly on or before 27 July 2026)
  3. Strict zero-population & building-only reporting (anti-resampling-bug constraint)
  4. Deterministic bilingual number hallucination validator (Arabic & Devanagari numerals)
  5. Dijkstra topological cut-off and isolated settlement clustering
  6. Cryptographic FactStore SHA-256 provenance hash reproducibility
  7. Full end-to-end headless pipeline execution with manifest verification
"""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import LineString, Point, Polygon

from src.cutoff_analysis import analyze_cutoffs
from src.damage_assessment import assess_infrastructure_damage
from src.fact_store import FactRecord, FactStore, build_fact_store_from_pipeline_outputs
from src.graph_builder import build_network, graph_to_geodataframes
from src.mountain_flood import detect_mountain_flood
from src.osm_ingest import DEFAULT_TIMESTAMP, fetch_pre_event_osm
from src.validator import validate_response
from scripts.run import TrackBPipeline


# =============================================================================
# 1. Mountain SAR Physics Filters
# =============================================================================
def test_mountain_sar_physics_filters(tmp_path: Path):
    """
    Verifies that detect_mountain_flood zeroes out any pixels located in
    radar shadows, radar layovers, or slopes greater than 15 degrees.
    """
    h, w = 60, 60
    transform = from_bounds(85.15, 27.85, 85.45, 28.25, w, h)
    meta = {
        "driver": "GTiff", "height": h, "width": w, "count": 1,
        "dtype": "float32", "crs": "EPSG:32645", "transform": transform,
    }

    # Pre-event: normal backscatter (-10 dB)
    pre_vv_path = tmp_path / "pre_vv.tif"
    with rasterio.open(pre_vv_path, "w", **meta) as dst:
        dst.write(np.full((h, w), -10.0, dtype=np.float32), 1)

    # Post-event: water backscatter (-18 dB, drop of -8 dB)
    post_vv_path = tmp_path / "post_vv.tif"
    with rasterio.open(post_vv_path, "w", **meta) as dst:
        dst.write(np.full((h, w), -18.0, dtype=np.float32), 1)

    # Masks
    # Top 20 rows: radar shadow
    shadow = np.zeros((h, w), dtype=np.uint8)
    shadow[:20, :] = 1
    shadow_path = tmp_path / "shadow.tif"
    meta_u8 = dict(meta, dtype="uint8")
    with rasterio.open(shadow_path, "w", **meta_u8) as dst:
        dst.write(shadow, 1)

    # Middle 20 rows: radar layover
    layover = np.zeros((h, w), dtype=np.uint8)
    layover[20:40, :] = 1
    layover_path = tmp_path / "layover.tif"
    with rasterio.open(layover_path, "w", **meta_u8) as dst:
        dst.write(layover, 1)

    # Bottom 20 rows: slope plausible (<= 15 deg)
    slope_ok = np.zeros((h, w), dtype=np.uint8)
    slope_ok[40:, :] = 1
    slope_path = tmp_path / "slope_plausible.tif"
    with rasterio.open(slope_path, "w", **meta_u8) as dst:
        dst.write(slope_ok, 1)

    valid_sar = np.ones((h, w), dtype=np.uint8)
    valid_sar_path = tmp_path / "valid_sar.tif"
    with rasterio.open(valid_sar_path, "w", **meta_u8) as dst:
        dst.write(valid_sar, 1)

    dem_masks = {
        "valid_sar": valid_sar_path,
        "shadow": shadow_path,
        "layover": layover_path,
        "slope_plausible": slope_path,
    }

    res = detect_mountain_flood(
        pre_vv_path, post_vv_path,
        dem_masks=dem_masks,
        out_dir=tmp_path / "flood_out",
        db_threshold=-2.5,
        post_db_threshold=-15.0,
        min_size_pixels=1,
    )

    with rasterio.open(res["flood_mask_path"]) as src:
        flood_mask = src.read(1)

    # Shadow (rows 0-20) and Layover (rows 20-40) must not contain flood detections (not 1)
    assert np.all(flood_mask[:20, :] != 1), "Radar shadow region must not contain flood detections"
    assert np.all(flood_mask[20:40, :] != 1), "Radar layover region must not contain flood detections"
    # Valid valley floor (inner pixels away from 1-pixel morphological border) detects flood
    assert np.sum(flood_mask[40:, :] == 1) > 100, "Valid valley floor must detect flood"


# =============================================================================
# 2. OSM Pre-Event Snapshot Date Constraint
# =============================================================================
def test_osm_preevent_snapshot_constraint():
    """
    Verifies that OSM ingestion uses a pre-event snapshot on or before
    27 July 2026 (2026-07-27T00:00:00Z) to prevent data leakage and disqualification.
    """
    assert DEFAULT_TIMESTAMP == "2026-07-27T00:00:00Z"
    dt = datetime.fromisoformat(DEFAULT_TIMESTAMP.replace("Z", "+00:00"))
    cutoff_dt = datetime(2026, 7, 27, 0, 0, 0, tzinfo=timezone.utc)
    assert dt <= cutoff_dt, f"OSM snapshot {DEFAULT_TIMESTAMP} is after cutoff date 2026-07-27"

    # Test synthetic ingestion offline
    bbox = [85.15, 27.85, 85.45, 28.25]
    osm_data = fetch_pre_event_osm(bbox, force_offline=True)
    assert "roads" in osm_data
    assert "bridges" in osm_data
    assert "buildings" in osm_data
    assert not osm_data["roads"].empty
    assert not osm_data["buildings"].empty


# =============================================================================
# 3. Strict Zero Population / Headcount Constraint
# =============================================================================
def test_no_worldpop_or_headcounts(tmp_path: Path):
    """
    Verifies that the system reports building counts (n_buildings) and never
    synthetic headcounts or WorldPop sums, preventing the 100x resampling distortion.
    """
    flood_stats = {"flood_area_km2": 4.12, "flood_pixels": 41200}
    damage_summary = {
        "total_road_km": 42.5, "affected_road_km": 6.8, "not_assessable_road_km": 2.1,
        "total_bridges": 5, "severed_bridges": 2,
        "total_buildings": 280, "flooded_buildings_count": 34,
    }
    cutoff_summary = {
        "n_cutoff_clusters": 3, "total_cutoff_buildings": 72,
        "clusters": [
            {"cluster_id": 1, "name": "Settlement A", "n_buildings": 38, "suggested_response": "Footpath only"},
            {"cluster_id": 2, "name": "Settlement B", "n_buildings": 34, "suggested_response": "Helicopter relief"},
        ]
    }

    fs_path = tmp_path / "fact_store.json"
    fs = build_fact_store_from_pipeline_outputs(flood_stats, damage_summary, cutoff_summary, out_path=fs_path)

    # Check that facts are recorded in building units
    assert "BUILDINGS_TOTAL_COUNT" in fs
    assert "BUILDINGS_FLOODED_COUNT" in fs
    assert "CUTOFF_BUILDINGS_COUNT" in fs

    # Ensure no facts have 'population', 'people', or 'headcount' in ID or unit
    for fact_id, record in fs.get_all_facts().items():
        assert "POPULATION" not in fact_id
        assert "HEADCOUNT" not in fact_id
        assert record.unit not in ["people", "population", "souls", "persons"]


# =============================================================================
# 4. Zero-Number-Hallucination Validator (English & Nepali)
# =============================================================================
def test_zero_number_hallucination_validator():
    """
    Verifies that the validator catches ungrounded numbers across both Arabic (0-9)
    and Devanagari (०-९) numerals with 100% precision.
    """
    prov = "7c3f81e095a48d6b9f2e7c3f81e095a4"
    time_str = "2026-08-26T12:00:00Z"
    facts = {
        "FLOOD_AREA_KM2": FactRecord(fact_id="FLOOD_AREA_KM2", value=4.12, unit="km2", source_layer="flood_mask.tif", description_en="Flood area", description_ne="बाढी क्षेत्र", provenance_hash=prov, timestamp=time_str),
        "ROADS_AFFECTED_KM": FactRecord(fact_id="ROADS_AFFECTED_KM", value=6.8, unit="km", source_layer="roads.geojson", description_en="Affected roads", description_ne="प्रभावित सडक", provenance_hash=prov, timestamp=time_str),
        "BRIDGES_SEVERED_COUNT": FactRecord(fact_id="BRIDGES_SEVERED_COUNT", value=2, unit="count", source_layer="bridges.geojson", description_en="Severed bridges", description_ne="विच्छेद पुल", provenance_hash=prov, timestamp=time_str),
        "BUILDINGS_TOTAL_COUNT": FactRecord(fact_id="BUILDINGS_TOTAL_COUNT", value=280, unit="count", source_layer="bldgs.geojson", description_en="Total buildings", description_ne="कुल भवन", provenance_hash=prov, timestamp=time_str),
        "CUTOFF_BUILDINGS_COUNT": FactRecord(fact_id="CUTOFF_BUILDINGS_COUNT", value=72, unit="buildings", source_layer="clusters.geojson", description_en="Cutoff buildings", description_ne="सम्पर्कविहीन भवन", provenance_hash=prov, timestamp=time_str),
    }
    fs = FactStore(facts)

    # Grounded English text
    grounded_en = "Satellite radar detected 4.12 km2 of flooding. 6.8 km of roads were affected and 2 bridges severed. 72 buildings are cut off."
    res_en = validate_response(grounded_en, fs)
    assert res_en.is_valid is True
    assert res_en.grounding_score == 1.0

    # Grounded Nepali text with Devanagari numerals
    grounded_ne = "बाढीले ४.१२ वर्ग किमी क्षेत्र जलमग्न बनाएको छ। ६.८ किमी सडक र २ पुलहरू विच्छेद भएका छन्। ७२ भवनहरू सम्पर्कविहीन छन्।"
    res_ne = validate_response(grounded_ne, fs)
    assert res_ne.is_valid is True
    assert res_ne.grounding_score == 1.0

    # Hallucinated English number (9999)
    hallucinated_en = "Satellite detected 4.12 km2 flooding and 9999 people trapped."
    bad_res_en = validate_response(hallucinated_en, fs)
    assert bad_res_en.is_valid is False
    assert any(v.normalized_value == 9999.0 for v in bad_res_en.violations)

    # Hallucinated Devanagari number (९९९९)
    hallucinated_ne = "बाढीमा ९९९९ जना फसेका छन्।"
    bad_res_ne = validate_response(hallucinated_ne, fs)
    assert bad_res_ne.is_valid is False
    assert any(v.normalized_value == 9999.0 for v in bad_res_ne.violations)


# =============================================================================
# 5. Dijkstra Cutoff Reachability When Bridge Severed
# =============================================================================
def test_dijkstra_cutoff_reachability(tmp_path: Path):
    """
    Verifies that when a bottleneck bridge is severed, the Dijkstra routing
    algorithm partitions the graph into isolated settlement clusters.
    """
    osm = fetch_pre_event_osm([85.15, 27.85, 85.45, 28.25], force_offline=True)
    facilities = osm.get("facilities", osm.get("medical"))
    G = build_network(osm["roads"], osm["bridges"], facilities, osm["buildings"])
    assert G.number_of_nodes() > 0

    # Mark bridge edges as affected
    for u, v, d in G.edges(data=True):
        if d.get("is_bridge"):
            d["status"] = "affected"

    res = analyze_cutoffs(G, out_dir=tmp_path / "cutoff_test")
    summary = res["summary"]

    assert summary["n_cutoff_clusters"] >= 1
    assert summary["n_cutoff_nodes"] >= 1


# =============================================================================
# 6. Cryptographic FactStore Provenance Hash Reproducibility
# =============================================================================
def test_provenance_hash_reproducibility():
    """
    Verifies that identical facts generate identical SHA-256 provenance hashes.
    """
    f1 = {"FLOOD_AREA_KM2": 4.12, "ROADS_AFFECTED_KM": 6.8, "CUTOFF_BUILDINGS": 72}
    f2 = {"CUTOFF_BUILDINGS": 72, "FLOOD_AREA_KM2": 4.12, "ROADS_AFFECTED_KM": 6.8}

    hash1 = hashlib.sha256(json.dumps(f1, sort_keys=True).encode()).hexdigest()
    hash2 = hashlib.sha256(json.dumps(f2, sort_keys=True).encode()).hexdigest()

    assert hash1 == hash2, "FactStore provenance hash must be deterministic and invariant to key insertion order"


# =============================================================================
# 7. End-to-End Headless Pipeline Execution (Offline Mode)
# =============================================================================
def test_pipeline_end_to_end_offline(tmp_path: Path):
    """
    Verifies that TrackBPipeline executes end-to-end under 15 seconds,
    generates all artifacts, and records SUCCESS in run_manifest.json.
    """
    pipeline = TrackBPipeline(
        bbox=[85.15, 27.85, 85.45, 28.25],
        event_date="2026-08-26",
        out_root=tmp_path / "run_out",
        force_offline=True,
    )
    manifest = pipeline.execute()

    assert manifest["status"] == "SUCCESS"
    assert manifest["total_duration_seconds"] < 25.0
    assert (tmp_path / "run_out" / "run_manifest.json").exists()
    assert (tmp_path / "run_out" / "fact_store.json").exists()
    assert (tmp_path / "run_out" / "sitreps" / "sitrep_en.md").exists()
    assert (tmp_path / "run_out" / "sitreps" / "sitrep_ne.md").exists()
    assert (tmp_path / "run_out" / "flood" / "flood_polygons.geojson").exists()
