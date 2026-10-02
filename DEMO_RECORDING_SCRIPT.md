# 🎬 FloodRescue AI: 3-Minute Backup Pitch Demo Video Script

> **Purpose**: A tight, high-impact screen recording and spoken walkthrough to serve as the official backup demo video for hackathon submissions and live judge presentations.  
> **Target Duration**: 2 minutes 45 seconds – 3 minutes 00 seconds  
> **Suggested Tools**: OBS Studio or Windows Game Bar (`Win + Alt + R`) at 1080p (1920×1080), 60 FPS.  

---

## 🛠️ Pre-Recording Checklist (60 Seconds Setup)

1. **Start Dashboard**:
   ```powershell
   .venv\Scripts\streamlit run app.py
   ```
2. **Browser Setup**:
   - Navigate to `http://localhost:8501`.
   - Set zoom to **100%** or **110%** for clear visibility of map labels.
   - Collapse the Streamlit hamburger menu (top right).
3. **Initial State**:
   - Sidebar -> Disaster Scenario: Select `"Real Satellite RTC: Dadu District, Sindh (2022 Flood Peak - Sentinel-1)"`.
   - Inundation Detection Engine: `"Baseline SAR Change Detection (Otsu & Drop Filter)"`.
   - Tab: Select **`🗺️ Interactive Operations Map`**.

---

## ⏱️ Minute-by-Minute Recording Timeline

| Timestamp | Visual Screen Action | Spoken Voiceover Script | Key Judge Takeaway |
| :--- | :--- | :--- | :--- |
| **00:00 – 00:30** | Show dashboard header, red KPI banner (`8,101 Trapped Population`), and zoom in on the Folium map. | *"When catastrophic monsoon rains hit Sindh, Pakistan in 2022, one-third of the country went underwater. Emergency teams received thousands of distress calls, but traditional flood maps only showed blue pixels over satellite photos. They didn't answer the only question that matters: **Who is cut off, and who dies first if we don't dispatch boats right now?** This is FloodRescue AI."* | Defines the critical humanitarian problem and system purpose. |
| **00:30 – 01:10** | In Sidebar, toggle layers: show SAR radar void, cyan flood extent, and red severed roads. Slide the dB slider slightly to show sub-second reactivity. | *"Optical satellites are blind under monsoon storm clouds. FloodRescue AI ingests all-weather Sentinel-1 C-band radar. Water acts as a specular mirror, scattering microwave pulses away from the sensor. By computing temporal change detection against pre-flood baselines, we automatically filter out permanent rivers and detect 22.2 square kilometers of active inundation in Dadu District in under 3.5 seconds on standard laptop hardware—zero GPU required."* | Proves all-weather physics, sub-second latency, and zero-GPU edge execution. |
| **01:10 – 01:50** | Click on Tab 1 map markers showing isolated villages. Hover over red road lines and severed bridges. | *"Here is our core technical differentiator: **Topological Isolation Analysis**. Most flood algorithms falsely report every dead-end dirt track as a disaster victim. FloodRescue AI builds a mathematical road network graph with NetworkX and evaluates a virtual super-node reachability invariant before versus after the flood. A settlement is flagged if and only if it had road access to an emergency hospital before the flood and lost it after."* | **THE WINNING DIFFERENTIATOR**: Topological graph invariant eliminates dead-end false alarms. |
| **01:50 – 02:25** | Click on **`📋 Rescue Priority Dispatch List`**. Scroll through the top-ranked clusters. Click the **`📥 Export Rescue Manifest (CSV)`** button. | *"We intersect cut-off road clusters with WorldPop 100-meter population rasters. In Dadu, we isolate 3 trapped communities totaling 8,101 citizens. Our Multi-Factor Rescue Priority Index weights trapped population, medical hub distance, and bridge damage scores. Village Cluster 02 is assigned HIGH priority with a score of 56.2, triggering an automated recommendation: **Boat evacuation**. With one click, incident commanders export an actionable manifest ready for military dispatch."* | Demonstrates complete end-to-end operational workflow from satellite to first responder. |
| **02:25 – 02:50** | Click on **`📊 Copernicus & Sen1Floods11 Ground Truth Benchmark`**. Highlight the pooled 22.4% baseline and the open water vs cropland breakdown. | *"We lead with honest benchmarks: across 11 held-out chips from the historical 2017 Pakistan event, our fast baseline achieves 22.4% pooled IoU, 36.6% F1, and 51.9% recall. The breakdown explains the physics: open water delivers 57% IoU and 80% recall, while flooded crops suffer double-bounce under-detection. On the 2022 disaster, running the exact same pipeline across two zones shows it tracks the real flood footprint: 22 square kilometers behind the protected eastern Indus levees, and 144 square kilometers where the western Johi breach occurred. We know the limits, and our next milestone targets recovering flooded vegetation."* | Demonstrates intellectual honesty, physical grounding, and dual-zone empirical validation. |
| **02:50 – 03:00** | Return to Tab 1 map overview. Display project title and GitHub link. | *"FloodRescue AI turns raw radar pixels into lifesaving evacuation manifests in 10 seconds. Built for first responders when every minute counts. Thank you."* | Clean, authoritative close. |

---

## 💡 Practical Recording Tips

- **Mouse Movement**: Move your cursor deliberately. Don't shake or circle erratically. Hover over key tooltips (e.g., road status, village markers) for 2 seconds so the viewer can read the popups.
- **Audio Clarity**: Use a headset mic or dedicated USB microphone. Keep a calm, authoritative, emergency-commander pace.
- **Backup Plan**: If recording fails, the Streamlit app is 100% self-contained and reproducible. Simply run:
  ```powershell
  .venv\Scripts\python scripts/run_pipeline.py --demo
  ```
  to display the complete terminal assessment in 3.4 seconds.
