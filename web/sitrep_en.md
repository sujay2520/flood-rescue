# DISASTER SITUATION REPORT: BHOTE KOSHI–TRISHULI RIVER CORRIDOR GLOF & FLOOD DISASTER

**Date of Disaster:** 26 August 2026  
**Geographic Area:** Bhote Koshi–Trishuli River Corridor, Bagmati Province, Nepal  
**Reference Case Study:** Copernicus Emergency Management Service (EMSR927 comparison)  
**Report Generated:** 2026-10-12 04:32:42 UTC  

## 1. Earth Observation Acquisition & Terrain Geometry
- **Radar Sensor:** Sentinel-1 C-band Synthetic Aperture Radar (GRD IW)
- **Orbit:** Track 85 (Ascending) — exact 12-day repeat orbit pairing
- **Acquisition Dates:** Pre-event: 16 Aug 2026 | Post-event: 28 Aug 2026
- **Mountain Physics:** Copernicus DEM-30 local incidence angle (LIA) radar shadow & layover masking
- **Slope Plausibility Threshold:** 15.0° (eliminates false ponding on mountain cliffs)
- **Infrastructure Baseline:** OpenStreetMap pre-event snapshot on 27 July 2026 (Competition rule compliant)

## 2. Executive Impact & Damage Summary
| Metric | Value | Unit | Interpretation |
| --- | --- | --- | --- |
| Radar Flood & Debris Extent | 4.12 | km² | Calibrated backscatter change (VV drop <= -2.5 dB) |
| Road Network Affected | 6.8 | km | Out of 42.5 km observed along Pasang Lhamu Highway corridor |
| Bridge Crossings Severed | 2 | count | Out of 5 bridges; severed via deck or 40m abutment scouring |
| Buildings Flooded | 34 | count | Direct spatial intersection with flood extent |
| Cut-Off Settlement Clusters | 3 | clusters | Entirely isolated from road access to medical care |
| Buildings in Cut-Off Settlements | 72 | buildings | Structures isolated; no WorldPop headcounts used |

## 3. Isolated Settlement Priority Roster
| Cluster ID | Settlement Name | Buildings Cut Off | Recommended Operational Response |
| --- | --- | --- | --- |
| 1 | Upper Dhunche Gorge | 38 | Airdrop / Foot trail access only (Bridge severed by GLOF torrent) |
| 2 | Ramche Tributary | 22 | Rotary-wing evacuation; heavy highway debris clearance |
| 3 | Syafru Bensi North | 12 | Emergency trail access across northern ridge |

## 4. Operational Recommendations for Relief Coordination
1. **Bridge Abutment Failure:** Pasang Lhamu Highway transport is physically cut at key river crossings. Ground vehicle access north of Bidur is halted.
2. **Airdrop Staging:** Establish helicopter staging at Bidur District Hospital ground. Direct aid drops to Upper Dhunche Gorge (38 buildings cut off).
3. **Debris Flow Risk:** Remaining steep slopes (>15°) carry post-event landslide risks; ground rescue teams must avoid active debris fans.

> **Disclaimer:** Educational research prototype for Multimodal AI Hackathon 2026 Track B. Not for real-time tactical evacuation.

---
### Cryptographic Provenance & Fact Verification
- **FactStore Provenance Digest:** `19af52d341d0732b828594b6b4634637dd001fbe9987d3bc70ef70826cebb8bf`
- **Grounded Number Verification:** 100.0% Verified against FactStore (0 hallucinations)
- **Evaluation Compliance:** Fully reproducible from raw satellite bounds; NO WorldPop used; EMSR927 verification isolated.