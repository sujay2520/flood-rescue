# 🌊 FloodRescue AI
> **Autonomous Sentinel-1 SAR Inundation Mapping & Topological Road Isolation Routing**  
> *From Satellite Pixels to Rescue Manifests in Under 11 Seconds.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-39%2F39%20passed-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/GPU-0%20Required%20(CPU%20Edge)-orange.svg)]()

---

## 🚨 The Humanitarian Problem
During catastrophic flood events, first responders receive satellite flood maps showing inundated pixels. But incident commanders need to answer three life-or-death operational questions:
1. **Which roads and bridges are actually cut off?**
2. **Which settlements have lost all road access to the nearest emergency hospital?**
3. **Where are the trapped communities who need evacuation boats first?**

**FloodRescue AI** answers this in **10.7 seconds** on a standard laptop CPU with zero GPU required.

```
+------------------------------------+          +------------------------------------+
|       WHAT RESCUE AGENCIES GET      |          |       WHAT RESCUE AGENCIES NEED    |
+------------------------------------+          +------------------------------------+
| • 10-meter raw SAR backscatter     |          | • Exact impassable road segments   |
| • 2 million flooded pixels (GeoTIFF)|   VS     | • Communities 100% cut off from hubs|
| • Static PDF reports 3 days late   |          | • Trapped population headcounts    |
| • No connection to road networks   |          | • Ranked dispatch list for boats   |
+------------------------------------+          +------------------------------------+
```

---

## ⚡ Core Technical Innovations

### 1. Temporal SAR Change Detection (Ignores Permanent Rivers)
Water acts as a specular reflector for C-band radar ($\sigma^0_{VV} < -17.0\text{ dB}$). Naive thresholding misclassifies every permanent river, lake, and radar shadow as a flood. We apply dual-condition temporal differencing:
$$\text{Flooded}(x, y) = (\text{Post}_{VV} < -17.0\text{ dB}) \land (\text{Post}_{VV} - \text{Pre}_{VV} < -2.5\text{ dB})$$
Because perennial water bodies are dark in both passes, their drop is $\approx 0\text{ dB}$. Only newly inundated terrain is flagged.

### 2. Topological Super-Node Reachability (Phase 3 Differentiator)
Naive algorithms flag dead-end dirt roads and pre-existing cul-de-sacs as flood victims. We construct a road graph with NetworkX and wire all active relief hospitals to a virtual super-node $\mathcal{S}$. A community is marked cut off if and only if:
$$\mathcal{V}_{\text{isolated}} = \text{Reach}_{\text{before}}(\mathcal{S}) \setminus \text{Reach}_{\text{after}}(\mathcal{S})$$

### 3. Multi-Factor Rescue Urgency Index
Trapped road nodes are clustered via post-flood graph connectivity, buffered, and overlaid on high-resolution WorldPop rasters to compute a composite rescue urgency score ($S \in [0, 100]$):
$$S_c = 0.40 \cdot \min(100, 20 \log_{10}(P_c + 1)) + 0.25 \cdot \min\left(100, \frac{D_{\text{hub}}}{500}\right) + 0.20 \cdot \min(100, 5 N_{\text{nodes}}) + 0.15 \cdot B_{\text{severance}}$$

---

## 📊 Quantitative Benchmarks & Physical Validation

```
HEADLINE: A fast, label-free baseline. Strong on open water, weak on crops.

[BENCHMARK — Sen1Floods11, 11 Held-Out Pakistan Chips (June 2017), 1.97M Valid Pixels]
Pooled Baseline:  IoU 22.4%  |  F1 36.6%  |  Precision 28.2%  |  Recall 51.9%

Where it works and where it doesn't:
  • Open-water basin (Pak_94095)      : IoU 56.9% | F1 72.6% | Precision 66.7% | Recall 79.6%
  • Riverine breach  (Pak_1027214)    : IoU 42.3% | F1 59.5% | Precision 48.1% | Recall 77.9%
  • Flooded crops    (Pak_849790)     : IoU 32.5% | F1 49.0% | Precision 89.1% | Recall 33.8%
                                        (High precision, but misses emergent crops via double-bounce)

[SAME PIPELINE, TWO ZONES — Sentinel-1 RTC, September 2022 Flood Peak]
  • Eastern Indus Corridor (levee-protected) :  22.2 km² flooded (5.0% of AOI) | 4.4 km roads cut
  • Western Johi / Manchar Breach Basin      : 143.9 km² flooded (32.0% of AOI) | Catastrophic breach
  -> The detector tracks the physical footprint: levees held in the east; basin drowned in the west.
```

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/sujay2520/flood-rescue.git
cd flood-rescue
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Interactive Command Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` to view the interactive Folium tactical map, priority lists, and before/after satellite imagery.

### 3. Run Headless Assessment via CLI
```bash
# On Real Pakistan 2022 Satellite Granules:
python scripts/run_pipeline.py \
  --pre data/real_sindh_2022/pre_flood_sar.tif \
  --post data/real_sindh_2022/post_flood_sar.tif \
  --roads data/real_sindh_2022/roads.graphml \
  --hubs data/real_sindh_2022/hubs.geojson \
  --pop data/real_sindh_2022/population.tif \
  --water_db -17.0 \
  --drop_db 2.5 \
  --output output/real_sindh

# Self-Contained Instant Demo (3.1s):
python scripts/run_pipeline.py --demo
```

### 4. Run Test Suite
```bash
pytest tests
```
All **39 test cases** pass with zero failures.

---

## 📁 Repository Structure
```text
flood-rescue/
├── app.py                     # Streamlit tactical command dashboard
├── requirements.txt           # Python dependencies
├── packages.txt               # System GDAL libraries for cloud deployment
├── src/
│   ├── core_logic.py          # Core SAR change detection & graph severance invariants
│   ├── flood.py               # Flood mask generation & GeoTIFF raster I/O
│   ├── roads.py               # Sub-pixel road geometry sampling & bridge risk
│   ├── isolation.py           # Super-node reachability & cut-off cluster analysis
│   ├── priority.py            # WorldPop integration & Multi-factor urgency index
│   └── train.py               # PyTorch Sen1Floods11 U-Net training pipeline
├── scripts/
│   ├── run_pipeline.py        # Headless automated CLI runner
│   ├── validate_copernicus.py # Ground-truth benchmark evaluator
│   └── tune_thresholds.py     # Unsupervised SAR radiometry calibrator
├── tests/                     # 39 passing unit & integration tests
├── PITCH_DECK_AND_PROJECT_BRIEF.md # Comprehensive deck brief & formulas
└── DEMO_RECORDING_SCRIPT.md   # 3-minute video backup walkthrough script
```

---

## 📜 Operational Disclosures & Limitations
1. **Emergent Crops**: C-band radar specular thresholding misses flooded vegetation due to stalk-water double bounce. Next milestone: positive change detection class (+2 dB) and dual-pol U-Net.
2. **Bridge Structural Load**: Satellite radar detects submerged approach roads, not underwater pier scour or foundation cavitation. Flagged bridges require UAV drone verification.
3. **Field Deployment**: Validation was grounded on 11 historical chips (June 2017) and Copernicus EMSR629. Field operations require local radiometry calibration.

---

## 📄 License
MIT License. Open source for humanitarian disaster relief.
