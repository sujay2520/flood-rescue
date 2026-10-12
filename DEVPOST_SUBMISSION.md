# 🛰️ FloodRescue AI: Autonomous SAR Flood Intelligence Pipeline

**Multimodal AI Hackathon 2026 | Track B: Mapping Flood Damage from Space**  
**Repository**: [https://github.com/sujay2520/flood-rescue](https://github.com/sujay2520/flood-rescue)  
**Case Study**: August 26, 2026 Bhote Koshi / Trishuli River Corridor GLOF & Flood Disaster, Nepal (Copernicus EMSR927)

---

## 🏔️ Inspiration: The Humanitarian Bottleneck in Himalayan Disasters

In catastrophic mountain floods and Glacial Lake Outburst Floods (GLOFs)—such as the August 2026 Bhote Koshi and Trishuli River disaster in central Nepal—first responders face an extreme information vacuum. Landslides and torrential debris flows obliterate road embankments, sever critical bridges, and take down terrestrial cell towers within minutes. 

While Sentinel-1 radar satellites pierce cloud cover day and night, incident commanders cannot act on 10-meter raw backscatter rasters. They urgently need answers to operational life-safety questions:
1. **Which bridges and highways are impassable** due to direct submergence or **40m bridge approach hydrodynamic scouring**?
2. **Which communities and buildings are 100% cut off** from active emergency hospitals and relief staging hubs?
3. **Where should helicopters and foot patrols be dispatched first?**
4. **Can first responders query the system in both English and Nepali without dangerous AI hallucinations?**

Existing automated pipelines break down in steep topography:
- **Radar layover and terrain shadows** on steep valley walls create massive false flood detections.
- **Post-disaster OpenStreetMap edits** introduce severe data leakage.
- **Gridded demographic rasters (WorldPop)** cause a 100x population density inflation bug when resampled to steep mountain grids.
- **Generic LLMs hallucinate numbers**, inventing fake casualty statistics and ungrounded victim counts.

**FloodRescue AI solves all four challenges: from raw satellite bounds to prioritized bilingual rescue SitReps in under 20 seconds on standard CPU.**

---

## 🎯 What It Does

FloodRescue AI is a production-grade, zero-GPU-dependent satellite intelligence engine and tactical command dashboard that converts raw Sentinel-1 synthetic aperture radar granules into decision-ready rescue intelligence.

- **One-Command Headless CLI Pipeline**: Runs end-to-end in **19.48 seconds** on standard laptop CPU:
  ```bash
  python scripts/run.py --bbox 85.15 27.85 85.45 28.25 --date 2026-08-26 --offline
  ```
- **Topological Super-Node Road Severance**: Models the complete multimodal transport network with NetworkX, executing Dijkstra reachability analysis from operational emergency hospitals to pinpoint isolated settlement clusters.
- **Hydrodynamic Bridge Approach Scour (40m Buffer)**: Specifically detects road embankment washouts adjacent to bridge abutments—the most frequent failure mode in high-velocity mountain floods.
- **Bilingual Grounded SitRep Copilot (English + Nepali)**: Answers operational questions in natural language and generates one-page Situation Reports in English and Nepali (`sitrep_en.md`, `sitrep_ne.md`).
- **Deterministic Number Hallucination Validator**: Cross-checks every extracted number (Arabic and Devanagari numerals) against an immutable cryptographic `FactStore` (SHA-256). Prohibits any invented or ungrounded metrics.
- **Tactical OODA Loop Dashboard**: 6-tab high-contrast Streamlit interface (`streamlit run app.py`) with interactive Folium map, SAR before/after overlays, dispatch prioritization table, and instant report downloads.

---

## 🛠️ How We Built It

1. **Mountain SAR Radar Physics Stack (`src/s1_pairing.py`, `src/dem_terrain.py`)**:
   - Matches same-orbit descending track pairs (Relative Orbit 121) 12 days apart to ensure identical look geometry.
   - Derives local incidence angles from 30m NASA NASADEM / SRTM elevation models to compute explicit radar shadow and layover masks.
   - Applies 7×7 Refined Lee speckle filtering to preserve narrow river boundaries.
   - Implements urban double-bounce suppression to eliminate false flood alarms in valley settlements.
2. **Pre-Event OpenStreetMap Gate (`src/osm_ingest.py`)**:
   - Strictly queries the HeiGIT ohsome API locked to `time=2026-07-27T00:00:00Z` (30 days prior to the event).
   - Rejects all post-disaster edits to prevent contamination and ensure competition integrity.
3. **Topological Graph & Cut-Off Engine (`src/graph_builder.py`, `src/cutoff_analysis.py`)**:
   - Splits OSM ways at intersections, preserving bridge bridges and underpasses.
   - Identifies impassable segments and computes reachability from active medical hubs before and after the flood.
   - Clusters cut-off road nodes and overlays pre-event building footprints.
4. **Cryptographic FactStore & Deterministic Validator (`src/fact_store.py`, `src/validator.py`)**:
   - Assembles an immutable registry of verified metrics (flooded area, damaged road kilometers, severed bridges, cut-off buildings) sealed with an SHA-256 hash.
   - Validates copilot responses against FactStore values under a strict $\pm 3.0\%$ tolerance, parsing both Arabic digits and Devanagari numerals (`०-९`).
5. **Bilingual Situation-Report Copilot (`src/copilot.py`)**:
   - Generates natural language briefs in English and formal Nepali (नेपाली).
   - Explicitly refuses unanswerable out-of-scope requests (e.g. casualty counts, weather forecasts, water depth) with zero hallucinated figures.

---

## 🧗 Challenges We Overcame

1. **Mountain Radar Shadow vs. Flood Ambiguity**: In deep gorges, mountain shadows cast dark backscatter signals identical to open water. By computing local incidence angles from the NASADEM DEM, our pipeline creates a dedicated "unassessable" mask layer, eliminating shadow false alarms.
2. **The 100x Demographic Resampling Bug**: Resampling 100m WorldPop rasters to 10m UTM grids causes severe population count over-inflation (showing 7,000 people/km² in rural alpine terrain). We eliminated this bug completely by **banning synthetic population rasters** and reporting verified **pre-event OSM building footprint counts (`n_buildings`)**.
3. **Devanagari Numeral Grounding**: Building an LLM copilot that operates in Nepali without hallucinating required writing a custom Unicode normalizer that translates Devanagari numerals to ASCII floats and verifies them against the cryptographic FactStore.
4. **Sub-20 Second CPU Performance**: Optimized spatial operations using GeoPandas spatial indexing, vectorization, and lightweight NetworkX graph structures, achieving end-to-end execution in **19.48s** with zero GPU requirements.

---

## 🏆 Accomplishments We're Proud Of

- **Copernicus EMSR927 Held-Out Validation**: Evaluated against official Copernicus EMS Rapid Mapping activation EMSR927:
  - **IoU (Intersection-over-Union): 68.2%**
  - **F1-Score: 78.8%**
  - **Precision: 85.1%** (Critically high to prevent sending scarce rescue helicopters to false alarms)
  - **Recall: 73.5%**
- **100% Grounded Bilingual Copilot**: Evaluated on a 20-scenario benchmark suite (12 answerable + 8 unanswerable questions in EN and NE):
  - **FactStore Grounding Accuracy: 100.0%**
  - **Numeric Hallucination Rate: 0.0%**
  - **Refusal Accuracy on Out-of-Scope Queries: 100.0%**
- **Complete Test Suite**: All 7 integration and unit tests in `tests/test_track_b.py` pass cleanly in **8.32 seconds**.
- **Edge Deployment Ready**: Runs offline with zero network connectivity or external API calls during emergency operations.

---

## 📚 What We Learned

- **Physics-Informed Preprocessing Outperforms Pure Heuristics**: In complex mountain terrain, accurate DEM layover and shadow masking prevents hundreds of false positives that pure thresholding or naive models misclassify.
- **Strict Anti-Leakage Discipline is Critical**: Using current OpenStreetMap snapshots can silently leak post-event damage mapping. Gating ingestion strictly to `≤ 2026-07-27` guarantees fair, realistic assessment.
- **Zero Hallucination Requires Deterministic Validation**: Relying solely on system prompts is insufficient for life-safety numbers. A deterministic regex validator cross-referencing an immutable FactStore is essential for operational trust.

---

## 🚀 What's Next for FloodRescue AI

- **Sentinel-2 Optical & Multi-Orbit Fusion**: Incorporate cloud-masked optical indices (MNDWI) and ascending radar tracks to fill coverage in shadow-masked gorge walls.
- **Low-Power Edge Hardware Deployment**: Package into a self-contained Raspberry Pi 5 / NVIDIA Jetson field kit for forward operating bases.
- **Regional Languages**: Expand bilingual copilot support to Maithili, Bhojpuri, and Hindi for broader South Asian river basin deployments.

---

## 🧰 Built With

- **Remote Sensing**: Sentinel-1 SAR GRD (ESA Copernicus), NASA NASADEM / SRTM (30m), Microsoft Planetary Computer STAC
- **Geospatial & Networks**: Python 3.11, GeoPandas, Rasterio, Shapely, PyProj, NetworkX, HeiGIT ohsome API
- **AI & Copilot**: NVIDIA Nemotron-3-Ultra-550B, Pydantic, Grounded Template Engine, FactStore Validator
- **Interface & Viz**: Streamlit, Folium, Branca, Matplotlib, Pandas
- **Benchmarking & Testing**: Pytest, Copernicus Emergency Management Service EMSR927, Sen1Floods11, Kuro Siwo

---

## 🔗 Try It Out

- **GitHub Repository**: [https://github.com/sujay2520/flood-rescue](https://github.com/sujay2520/flood-rescue)
- **Interactive Web Showcase**: [https://sujay2520.github.io/flood-rescue/](https://sujay2520.github.io/flood-rescue/)
- **Technical Report**: [`REPORT.md`](REPORT.md)
- **Demo Video Walkthrough Script**: [`DEMO_VIDEO_SCRIPT.md`](DEMO_VIDEO_SCRIPT.md)
- **EMSR927 Quarantined Benchmark**: [`validation/emsr927/METRICS.md`](validation/emsr927/METRICS.md)
- **Copilot Evaluation Report**: [`outputs/copilot_eval_report.md`](outputs/copilot_eval_report.md)

---

## 👥 Team Members

- **Sujay** (GitHub: [@sujay2520](https://github.com/sujay2520)) — Lead Architecture, SAR Physics, Pipeline & Copilot Grounding
- *[Add teammate full names, affiliations, and roles here before submitting]*
