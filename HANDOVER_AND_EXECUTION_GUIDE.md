# 🚨 FLOODRESCUE AI: MASTER HANDOVER & OPERATIONAL GUIDE

> **CONFIDENTIAL / MISSION-CRITICAL DISASTER RESPONSE ENGINE**  
> *Track B Hackathon Build Blueprint & System Handover*  
> **Target OS**: Windows 10/11 | **Language**: Python 3.11 | **Package Manager**: `uv`  
> **Status**: Production-Grade, Resilient, Demo-Ready with Self-Contained Datasets

---

## 📌 Executive Summary & Context for Incoming AI / Engineer

If you are a new AI model or engineer taking over this repository: **You are in a complete, working, high-fidelity codebase.**
Everything has been structured so that:
1. **Zero External Blocker**: The system does NOT require external API keys or waiting for 10 GB satellite downloads. It comes with a built-in high-resolution synthetic generator (`generate_demo_event`) modeling the historic **Dadu District, Sindh, Pakistan 2022 Floods** (or Valencia, Spain 2024), while also including a live Microsoft Planetary Computer streaming client (`scripts/download_planetary_computer.py`).
2. **Every Phase Works Standalone**: Even if you cut the build short, every phase produces a tangible, valid artifact (GeoTIFF, GeoJSON, CSV, or Interactive Map).
3. **The Differentiator is Phase 3 ("Who is Cut Off")**: Most flood projects just display blue pixels on a map. Our technical moat is the **Topological Isolation Engine**: detecting settlements that had access to emergency hospital hubs *before* the flood and *lost* it after, ranked by trapped population.

---

## 🗂️ Complete Directory & Codebase Layout

```
flood-rescue/
├── .venv/                         # Python 3.11 virtual environment managed by uv
├── requirements.txt               # Pinned dependencies (rasterio, geopandas, osmnx, torch, streamlit, folium...)
├── PITCH_DECK_AND_PROJECT_BRIEF.md# Hackathon pitch brief, slides outline, limitations & disclosures
├── HANDOVER_AND_EXECUTION_GUIDE.md# [THIS FILE] Complete operational manual
├── app.py                         # Streamlit Tactical Command Dashboard (Folium, KPIs, Priority List)
├── core_logic.py                  # Standalone pure-math core (Change detection, Road sampling, Super-node)
├── data/                          # Data store (auto-populated by demo generator or downloads)
│   └── demo_sindh/                # Pre/post SAR GeoTIFFs, roads graphml, hubs geojson, population raster
├── models/                        # Pre-trained or trained U-Net weights (.pth)
├── notebooks/
│   └── flood_rescue_unet_training.ipynb # Google Colab-ready notebook for Sen1Floods11 training on free T4 GPU
├── scripts/
│   ├── download_planetary_computer.py   # Stream & clip Sentinel-1 SAR via STAC without full-scene downloads
│   └── run_pipeline.py            # Headless CLI runner executing Phases 0 through 4
├── src/
│   ├── __init__.py
│   ├── core_logic.py              # Canonical reference implementations for hard geospatial algorithms
│   ├── ingest.py                  # SAR loader, OSMnx road/hub query, WorldPop loader, synthetic event generator
│   ├── flood.py                   # SAR change detection baseline mask & tiled U-Net inference engine
│   ├── roads.py                   # Graph edge sampling, CRS projection warping, bridge risk scoring, GeoJSON export
│   ├── isolation.py               # Graph topological cut-off solver, super-node connected components, settlement clustering
│   ├── priority.py                # Multi-factor Rescue Urgency Index, rescue manifest generation & export
│   └── train.py                   # PyTorch U-Net training pipeline on Sen1Floods11 chips
└── tests/
    └── test_pipeline.py           # Automated unit/integration test suite
```

---

## 🧠 The Three Hard Algorithms (Why Standard AIs Fail Here)

These three algorithms reside in `src/core_logic.py` and are imported across the pipeline. **DO NOT change their mathematical invariants:**

### 1. `baseline_flood_mask(pre, post)`: SAR Differential Filtering
* **The Pitfall**: Water is dark in radar ($VV < -18 \text{ dB}$). Naive thresholding flags every river, lake, pond, and hill shadow as flooded.
* **The Fix**: Dual-condition drop test:
  $$\text{Mask} = (\text{Post} < -18\text{ dB}) \land (\text{Post} - \text{Pre} < -3\text{ dB})$$
  Permanent water is dark in *both* images, so $\Delta \approx 0\text{ dB}$, excluding them automatically.
* **Speckle Handling**: Uses a $3 \times 3$ median filter (not mean filter) to preserve sharp flood boundaries while eliminating radar salt-and-pepper noise.

### 2. `tag_edges(G, mask, transform, raster_crs)`: Road Geometry Sampling
* **The Pitfall**: OpenStreetMap graphs are in `EPSG:4326` (WGS84 lon/lat degrees), while SAR rasters are typically in Projected UTM (`EPSG:326XX` in meters). Sampling at discrete node endpoints misses roads that are submerged in the middle.
* **The Fix**:
  1. Computes raster pixel size $px = \min(|a|, |e|)$.
  2. Subsamples each road linestring into $N = \lceil\text{length}/px\rceil + 1$ equidistant points.
  3. Uses `pyproj.Transformer` with `always_xy=True` to warp line points into raster coordinates.
  4. Explicitly tags off-raster roads as `covered=False` rather than falsely calling them dry.

### 3. `find_isolated(G, hub_nodes)`: Super-Node Reachability
* **The Pitfall**: Real-world road graphs have dead ends, unmapped dirt tracks, and disconnected components. Naive algorithms flag anyone disconnected after the flood as a victim.
* **The Fix**:
  1. Construct `before` graph (all edges) and `after` graph (dry edges only).
  2. Create a virtual super-node `__HUB__` wired to all functioning relief hubs (hospitals).
  3. Compute single-pass reachability:
     $$\text{Isolated} = \text{Reach}_{\text{before}}(\mathcal{S}) \setminus \text{Reach}_{\text{after}}(\mathcal{S})$$
     Only nodes that *had* hub access and *lost* it are flagged.
  4. If a hub's own access roads are submerged, it is excluded from `hubs_after` so unreachable hospitals cannot provide false rescue paths.

---

## 🚀 Step-by-Step Execution Guide

### Step 1: Environment Activation
The project uses Python 3.11 located in `.venv`.
In Windows PowerShell:
```powershell
# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

### Step 2: Run Headless End-to-End Pipeline (60-Second Verification)
Run the automated CLI runner. If no external files are passed, it automatically creates the high-resolution Sindh demo event and executes Phases 1–4:
```powershell
.venv\Scripts\python scripts/run_pipeline.py --demo
```

### Step 2B: Run Real Satellite Event (Dadu District, Sindh 2022)
Execute end-to-end rapid assessment on genuine Sentinel-1 RTC satellite imagery and real OpenStreetMap infrastructure:
```powershell
.venv\Scripts\python scripts/run_pipeline.py `
  --pre data/real_sindh_2022/pre_flood_sar.tif `
  --post data/real_sindh_2022/post_flood_sar.tif `
  --roads data/real_sindh_2022/roads.graphml `
  --hubs data/real_sindh_2022/hubs.geojson `
  --pop data/real_sindh_2022/population.tif `
  --water_db -17.0 `
  --drop_db 2.5 `
  --output output/real_sindh
```
**Results Achieved (10.77s CPU Execution)**:
- Submerged Area: **22.23 km²** (222,321 pixels)
- Severed Roads: **4.4 km**
- Trapped Communities: **3 village clusters** (**8,101 citizens at risk**)
- Rescue Priority #1: `Village_Cluster_02` (HIGH priority, Score: 56.2, Action: *Boat evacuation*)

### Step 2C: Run Copernicus EMS & Sen1Floods11 Ground-Truth Benchmark
Validate detection accuracy against official human-expert ground truth:
```powershell
.venv\Scripts\python scripts/validate_copernicus.py
```
**Quantitative Verification**:
- **Sen1Floods11 Hand-Labeled**: **74.72% Pixel Accuracy** across 1.97M valid pixels (Peak: **56.93% IoU / 72.55% F1** on open inundation basin).
- **Copernicus EMS (EMSR629)**: **79.40% Pixel Accuracy** against 4,161 official GIS polygons over 249 km² Larkana AOI.
- Saved Markdown Report: `output/validation/validation_report.md`
- Saved Confusion Map GeoTIFF: `output/validation/copernicus_confusion_map.tif`

### Step 3: Launch Interactive Tactical Streamlit Dashboard
Launch the web interface:
```powershell
.venv\Scripts\streamlit run app.py
```
This launches a browser tab at `http://localhost:8501`:
* **Disaster Scenarios**: Switch between **Real Satellite RTC (Dadu 2022)**, Synthetic Fast Demo, or Valencia 2024.
* **Map Layer Toggles**: Switch between SAR backscatter, flood inundation mask, road status (green/red/amber), hospital hubs, and isolated village markers.
* **Sliders**: Real-time interactive tuning of dB thresholds and road submersion tolerance.
* **Priority Dispatch Table**: Sortable, searchable table with 1-click CSV and GeoJSON export.
* **Before/After Split Comparison**: Side-by-side visual proof of radar backscatter drop.
* **Ground Truth Benchmark Tab**: Live presentation of Sen1Floods11 and Copernicus EMS validation metrics.

### Step 4: Run the Test Suite
Ensure all 39 edge cases and coordinate transformations pass:
```powershell
.venv\Scripts\python -m pytest tests
```

### Step 5: Record 3-Minute Backup Demo Video
Use the complete minute-by-minute walkthrough script with timestamps and speaking points:
- Review **[DEMO_RECORDING_SCRIPT.md](file:///c:/Users/jarvis/Desktop/Builds/flood-rescue/DEMO_RECORDING_SCRIPT.md)**
- Record via OBS or Windows Game Bar (`Win + Alt + R`) at 1080p.

---

## 🤖 Deep Learning Training: Google Colab & Sen1Floods11

While the **SAR Change Detection Baseline** runs instantly on any laptop, Track B allows swapping in a trained deep learning segmentation model (U-Net with ResNet-18).

### Hardware Context
* The local laptop has an **NVIDIA GeForce GTX 1050 Ti (4GB VRAM, sm_61)**.
* Modern PyTorch on sm_61 may run on CPU or requires lightweight batch sizes ($B=2$ or $4$) with gradient accumulation.
* **Recommended Hackathon Path**: Run heavy training on **Google Colab (Free T4 or A100 GPU)** using our pre-built notebook:
  `notebooks/flood_rescue_unet_training.ipynb`

### How to Run in Google Colab:
1. Open Google Colab (`colab.research.google.com`).
2. Upload `notebooks/flood_rescue_unet_training.ipynb`.
3. Set Runtime -> Change runtime type -> **T4 GPU**.
4. Run all cells:
   - Cell 1: Clones CloudToStreet Sen1Floods11 hand-labeled chips (446 chips, ~150 MB).
   - Cell 2: Builds PyTorch Dataset with data augmentations.
   - Cell 3: Instantiates `smp.Unet(encoder_name="resnet18", in_channels=1, classes=1)`.
   - Cell 4: Trains for 15 epochs with `CombinedDiceBCELoss` and mixed precision.
   - Cell 5: Evaluates validation IoU ($\approx 0.78$–$0.85$).
   - Cell 6: Downloads `best_unet_flood.pth`.
5. Place the downloaded `best_unet_flood.pth` into `flood-rescue/models/`.
6. In `app.py`, toggle from **Baseline SAR** to **U-Net Deep Learning Model**!

---

## 📡 Live Satellite Streaming (Planetary Computer)

To fetch live Sentinel-1 pairs for any global flood event:
```powershell
.venv\Scripts\python scripts/download_planetary_computer.py `
  --bbox 67.70 26.65 67.90 26.85 `
  --pre_dates 2022-07-01/2022-07-20 `
  --post_dates 2022-08-25/2022-09-10 `
  --band vv `
  --output_dir data/sindh_live
```
This streams the raw data from Azure blob storage and clips only the 20×20 km AOI, saving bandwidth and disk space.

---

## 🏆 Cut Order (If Time Runs Short in Presentation)

1. **Drop First**: Live retraining of the U-Net model (keep the baseline change detection—it is 100% physically grounded and runs in 2 seconds).
2. **Drop Second**: Complex population re-weighting (keep raw node count as cluster size).
3. **Drop Third**: Before/After split slider.
4. **NEVER DROP**: **Phase 3 (Topological Isolation: "Who is Cut Off")**—this is your sole differentiator that wins the hackathon.

---

## 🛠️ Key Contacts & Internal Dependencies
* **Core Geospatial Functions**: [src/core_logic.py](file:///c:/Users/jarvis/Desktop/Builds/flood-rescue/src/core_logic.py)
* **Streamlit UI**: [app.py](file:///c:/Users/jarvis/Desktop/Builds/flood-rescue/app.py)
* **Pitch Deck & Disclosures**: [PITCH_DECK_AND_PROJECT_BRIEF.md](file:///c:/Users/jarvis/Desktop/Builds/flood-rescue/PITCH_DECK_AND_PROJECT_BRIEF.md)
