"""
src/fact_store.py

Production-grade Fact Store Module for Track B: Mapping Flood Damage from Space (Nepal Trishuli Disaster).
Provides an immutable, cryptographically verifiable Fact Store for disaster metrics.

Author: Track B Engineering Team
License: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Pydantic Models
# =============================================================================

class FactRecord(BaseModel):
    """
    Immutable record representing a single verified fact or metric.
    Frozen to ensure immutability and cryptographic integrity.
    """
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )

    fact_id: str = Field(
        ...,
        description="Unique identifier in UPPER_SNAKE_CASE (e.g., FLOOD_AREA_KM2)",
        min_length=1,
        max_length=64,
    )
    value: Union[float, int, str] = Field(
        ...,
        description="Metric value (numeric or string)",
    )
    unit: str = Field(
        ...,
        description="Unit of measurement (e.g., 'km2', 'km', 'count', 'date')",
        min_length=1,
        max_length=32,
    )
    source_layer: str = Field(
        ...,
        description="Source data layer or artifact filename",
        min_length=1,
        max_length=256,
    )
    description_en: str = Field(
        ...,
        description="Human-readable description in English",
        min_length=1,
        max_length=512,
    )
    description_ne: str = Field(
        ...,
        description="Human-readable description in Nepali",
        min_length=1,
        max_length=512,
    )
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hex digest of the source file or payload",
        min_length=16,
        max_length=64,
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 UTC timestamp when fact was registered",
    )

    @field_validator("fact_id", mode="before")
    @classmethod
    def _normalize_fact_id(cls, v: str) -> str:
        if not isinstance(v, str):
            raise TypeError("fact_id must be a string")
        return v.strip().upper().replace("-", "_").replace(" ", "_")


# =============================================================================
# Fact Store Class
# =============================================================================

class FactStore:
    """
    Immutable fact store holding verified disaster metrics.
    Keyed by uppercase fact_id for fast lookup.
    """

    def __init__(self, facts: Optional[Dict[str, FactRecord]] = None) -> None:
        self._facts: Dict[str, FactRecord] = {}
        if facts:
            for fact_id, record in facts.items():
                if not isinstance(record, FactRecord):
                    raise TypeError(f"Value for {fact_id} is not a FactRecord")
                self._facts[record.fact_id] = record

    def register_fact(self, fact: FactRecord) -> FactStore:
        """Return a new FactStore with the new fact added."""
        new_facts = self._facts.copy()
        new_facts[fact.fact_id] = fact
        return FactStore(new_facts)

    def get_fact(self, fact_id: str) -> Optional[FactRecord]:
        """Retrieve a fact by its ID."""
        return self._facts.get(fact_id.upper().strip())

    def get_value(self, fact_id: str) -> Any:
        """Retrieve just the value of a fact."""
        fact = self.get_fact(fact_id)
        return fact.value if fact else None

    def get_all_numeric_values(self) -> Dict[str, float]:
        """Return all numeric facts as {fact_id: float_value} for hallucination validation."""
        return {
            fact_id: float(record.value)
            for fact_id, record in self._facts.items()
            if isinstance(record.value, (int, float))
        }

    def get_all_facts(self) -> Dict[str, FactRecord]:
        return self._facts.copy()

    def __len__(self) -> int:
        return len(self._facts)

    def __contains__(self, fact_id: str) -> bool:
        return fact_id.upper().strip() in self._facts

    def __iter__(self):
        return iter(self._facts.values())

    def save_to_json(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "fact_count": len(self._facts),
            "facts": {fid: record.model_dump(mode="json") for fid, record in self._facts.items()},
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> FactStore:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fact store not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        facts_data = data.get("facts", {})
        facts = {fid: FactRecord(**fdata) for fid, fdata in facts_data.items()}
        return cls(facts)


# =============================================================================
# Helper & Factory Functions
# =============================================================================

def compute_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 of file or return hash of non-existent string."""
    path = Path(filepath)
    if not path.exists():
        return hashlib.sha256(str(filepath).encode()).hexdigest()
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_fact_store_from_pipeline_outputs(
    flood_stats: Dict[str, Any],
    damage_summary: Dict[str, Any],
    cutoff_summary: Dict[str, Any],
    pair_metadata: Optional[Dict[str, Any]] = None,
    out_path: Union[str, Path] = "./outputs/fact_store.json",
) -> FactStore:
    """
    Build a FactStore from live pipeline in-memory dictionaries and artifact paths.
    """
    out_path = Path(out_path)
    timestamp = utc_now_iso()
    meta_hash = hashlib.sha256(json.dumps(flood_stats, default=str).encode()).hexdigest()

    facts = {}

    def add_fact(fid: str, val: Any, unit: str, src: str, en: str, ne: str):
        facts[fid] = FactRecord(
            fact_id=fid,
            value=val,
            unit=unit,
            source_layer=src,
            description_en=en,
            description_ne=ne,
            provenance_hash=meta_hash,
            timestamp=timestamp,
        )

    # 1. Flood Extent Facts
    flood_area_km2 = round(float(flood_stats.get("flood_area_km2", flood_stats.get("flood_area_m2", 0) / 1e6)), 2)
    flood_px = int(flood_stats.get("flood_pixels", 0))
    add_fact("FLOOD_AREA_KM2", flood_area_km2, "km2", "flood_mask.tif",
             "Total flooded/debris area mapped from satellite radar",
             "उपग्रह रडारबाट नक्शा गरिएको कुल बाढी तथा गेग्रान ग्रस्त क्षेत्रफल")
    add_fact("FLOOD_PIXELS_COUNT", flood_px, "pixels", "flood_mask.tif",
             "Count of 10m satellite pixels classified as flooded",
             "बाढी ग्रस्त वर्गीकरण गरिएका १० मिटर उपग्रह पिक्सेलको संख्या")

    # 2. Road Damage Facts
    tot_road_km = round(float(damage_summary.get("total_road_km", 0.0)), 2)
    aff_road_km = round(float(damage_summary.get("affected_road_km", 0.0)), 2)
    unobs_road_km = round(float(damage_summary.get("not_assessable_road_km", 0.0)), 2)

    add_fact("ROADS_TOTAL_KM", tot_road_km, "km", "roads_assessed.geojson",
             "Total road length in observed area",
             "अध्ययन क्षेत्रमा सडकको कुल लम्बाई")
    add_fact("ROADS_AFFECTED_KM", aff_road_km, "km", "roads_assessed.geojson",
             "Length of roads affected or cut by floodwaters/debris",
             "बाढी वा पहिरोले प्रभावित वा अवरुद्ध भएको सडकको लम्बाई")
    add_fact("ROADS_NOT_ASSESSABLE_KM", unobs_road_km, "km", "roads_assessed.geojson",
             "Road length in radar shadow or layover unobservable from space",
             "रडार छायाँ वा लेओभरका कारण अन्तरिक्षबाट अवलोकन गर्न नसकिएको सडक लम्बाई")

    # 3. Bridge Damage Facts
    tot_bridges = int(damage_summary.get("total_bridges", 0))
    sev_bridges = int(damage_summary.get("severed_bridges", 0))
    add_fact("BRIDGES_TOTAL_COUNT", tot_bridges, "count", "bridges_assessed.geojson",
             "Total bridge crossings in the study corridor",
             "अध्ययन करिडोरमा रहेका कुल पुलहरूको संख्या")
    add_fact("BRIDGES_SEVERED_COUNT", sev_bridges, "count", "bridges_assessed.geojson",
             "Count of bridges severed directly or via 40m approach washouts",
             "प्रत्यक्ष वा ४० मिटर पहुँच सडक कटानका कारण सम्पर्कविच्छेद भएका पुलहरूको संख्या")

    # 4. Building Damage Facts
    tot_buildings = int(damage_summary.get("total_buildings", 0))
    fld_buildings = int(damage_summary.get("flooded_buildings_count", 0))
    add_fact("BUILDINGS_TOTAL_COUNT", tot_buildings, "count", "buildings_assessed.geojson",
             "Total pre-event OSM mapped buildings in corridor",
             "करिडोरमा रहेका कुल पूर्व-घटना ओएसएम भवनहरू")
    add_fact("BUILDINGS_FLOODED_COUNT", fld_buildings, "count", "buildings_assessed.geojson",
             "Buildings directly intersecting flood or mudflow extent",
             "प्रत्यक्ष रूपमा बाढी वा लेदो बहावमा परेका भवनहरूको संख्या")

    # 5. Cut-Off Community Facts
    n_cutoff_clusters = int(cutoff_summary.get("n_cutoff_clusters", 0))
    cutoff_bldgs = int(cutoff_summary.get("total_cutoff_buildings", 0))
    add_fact("CUTOFF_CLUSTERS_COUNT", n_cutoff_clusters, "clusters", "isolated_clusters.geojson",
             "Number of distinct settlement clusters completely cut off from road network",
             "सडक सञ्जालबाट पूर्ण रूपमा सम्पर्कविच्छेद भएका बस्ती समूहहरूको संख्या")
    add_fact("CUTOFF_BUILDINGS_COUNT", cutoff_bldgs, "buildings", "isolated_clusters.geojson",
             "Total number of buildings completely cut off from road network",
             "सडक सञ्जालबाट पूर्ण रूपमा सम्पर्कविच्छेद भएका भवनहरूको कुल संख्या")
    # 6. Per-cluster building facts
    if "clusters" in cutoff_summary:
        for c in cutoff_summary["clusters"]:
            cid = c.get("cluster_id", 1)
            nb = c.get("n_buildings", 0)
            cname = c.get("name", f"Cluster {cid}")
            add_fact(f"CUTOFF_CLUSTER_{cid}_BUILDINGS", nb, "buildings", "isolated_clusters.geojson",
                     f"Buildings in isolated cluster {cname}", f"{cname} सम्पर्कविहीन बस्तीमा रहेका भवनहरू")

    # 7. Satellite & Disaster Context
    if pair_metadata:
        orbit = pair_metadata.get("relative_orbit", 85)
        add_fact("SATELLITE_RELATIVE_ORBIT", orbit, "orbit_number", "pair_metadata.json",
                 "Sentinel-1 relative orbit track",
                 "सेन्टिनेल-१ उपग्रहको सापेक्षिक कक्ष ट्र्याक")

    store = FactStore(facts)
    store.save_to_json(out_path)
    return store


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Fact Store Management")
    parser.add_argument("--test", action="store_true", help="Run test creation")
    args = parser.parse_args()

    dummy_flood = {"flood_area_km2": 4.12, "flood_pixels": 41200}
    dummy_damage = {
        "total_road_km": 42.5, "affected_road_km": 6.8, "not_assessable_road_km": 2.1,
        "total_bridges": 5, "severed_bridges": 2, "total_buildings": 280, "flooded_buildings_count": 34
    }
    dummy_cutoff = {"n_cutoff_clusters": 3, "total_cutoff_buildings": 72}

    store = build_fact_store_from_pipeline_outputs(dummy_flood, dummy_damage, dummy_cutoff)
    print("=" * 70)
    print(f"FactStore Built with {len(store)} Facts:")
    for f in store:
        print(f"  {f.fact_id:25s}: {f.value} {f.unit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
