# 🤖 Bilingual Copilot Quantitative Evaluation Report
**Evaluation Timestamp:** `2026-10-12T05:58:19.768913+00:00`  
**Test Suite:** 20 Scenarios (12 Answerable + 8 Unanswerable) across English & Nepali = **40 Total Evaluated Queries**  
**Grounding Guarantee:** Cryptographic FactStore (SHA-256) + Deterministic Number Validator  

---

## 1. Executive Metric Summary

| Metric | Target | Evaluated Result | Evaluation Status |
| :--- | :---: | :---: | :---: |
| **FactStore Grounding Accuracy** | 100.0% | **100.0%** (24/24) | ✅ PASS |
| **Numeric Hallucination Rate** | 0.0% | **0.0%** (0 violations) | ✅ PASS |
| **Refusal Accuracy on Out-of-Scope Queries** | 100.0% | **100.0%** (16/16) | ✅ PASS |
| **Overall Benchmark Accuracy** | ≥ 95.0% | **100.0%** (40/40) | ✅ PASS |
| **Average Inference Latency** | < 50ms | **0.17 ms** | ✅ PASS |

---

## 2. Test Case Breakdown

### A. Answerable Queries (24 total: 12 EN + 12 NE)
| ID | Lang | Topic | Query | Grounded | Violations | Latency |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| Q01 | EN | `flood_extent` | What is the total flood and debris extent in square kilometers? | ✅ | 0 | 1.0ms |
| Q01 | NE | `flood_extent` | कुल बाढी र गेग्रान बहाव क्षेत्र कति वर्ग किमी हो? | ✅ | 0 | 1.0ms |
| Q02 | EN | `roads_affected` | How many kilometers of roads are affected or submerged? | ✅ | 0 | 0.0ms |
| Q02 | NE | `roads_affected` | कति किमी सडक प्रभावित वा डुबानमा परेको छ? | ✅ | 0 | 0.0ms |
| Q03 | EN | `bridges_severed` | How many bridges are severed or scoured? | ✅ | 0 | 0.0ms |
| Q03 | NE | `bridges_severed` | कति वटा पुल सम्पर्कविहीन भएका छन्? | ✅ | 0 | 0.0ms |
| Q04 | EN | `buildings_inundated` | How many buildings are directly inundated by floodwaters? | ✅ | 0 | 0.0ms |
| Q04 | NE | `buildings_inundated` | बाढीबाट कति भवनहरू प्रत्यक्ष प्रभावित छन्? | ✅ | 0 | 1.0ms |
| Q05 | EN | `isolated_clusters` | How many cut-off settlement clusters are isolated from hospitals? | ✅ | 0 | 1.0ms |
| Q05 | NE | `isolated_clusters` | अस्पतालहरूबाट कति बस्ती समूहहरू सम्पर्कविहीन छन्? | ✅ | 0 | 1.0ms |
| Q06 | EN | `isolated_buildings` | How many pre-event OSM buildings are cut off in isolated clusters? | ✅ | 0 | 0.0ms |
| Q06 | NE | `isolated_buildings` | सम्पर्कविहीन बस्तीहरूमा कति वटा भवनहरू छन्? | ✅ | 0 | 0.0ms |
| Q07 | EN | `roads_total` | What is the total road network length analyzed? | ✅ | 0 | 0.0ms |
| Q07 | NE | `roads_total` | कुल विश्लेषण गरिएको सडक सञ्जालको लम्बाई कति हो? | ✅ | 0 | 0.0ms |
| Q08 | EN | `bridges_total` | How many total bridges were assessed in the corridor? | ✅ | 0 | 0.0ms |
| Q08 | NE | `bridges_total` | करिडोरमा कुल कति पुलहरूको मूल्याङ्कन गरिएको थियो? | ✅ | 0 | 0.0ms |
| Q09 | EN | `executive_brief` | Give me the executive disaster summary brief. | ✅ | 0 | 0.0ms |
| Q09 | NE | `executive_brief` | विपद् स्थिति संक्षेप विवरण दिनुहोस्। | ✅ | 0 | 0.0ms |
| Q10 | EN | `road_bridge_status` | What is the status of the highway and bridge network? | ✅ | 0 | 0.0ms |
| Q10 | NE | `road_bridge_status` | राजमार्ग र पुल सञ्जालको स्थिति के छ? | ✅ | 0 | 1.0ms |
| Q11 | EN | `priority_cluster` | Which settlement cluster is Priority 1 for evacuation? | ✅ | 0 | 1.0ms |
| Q11 | NE | `priority_cluster` | उद्धारको लागि प्राथमिकता १ बस्ती कुन हो? | ✅ | 0 | 0.0ms |
| Q12 | EN | `response_recommendations` | What are the recommended operational response actions? | ✅ | 0 | 0.0ms |
| Q12 | NE | `response_recommendations` | सुझाव गरिएका उद्धार कार्यहरू के के हुन्? | ✅ | 0 | 0.0ms |

### B. Unanswerable Out-of-Scope Queries (16 total: 8 EN + 8 NE)
*(Casualties, death tolls, precipitation forecasts, river bathymetry, and unobserved regions outside satellite swath)*

| ID | Lang | Out-of-Scope Topic | Query | Refusal Verified | Violations | Status |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: |
| Q13 | EN | `casualties_death` | How many people died in the flood disaster? | ✅ | 0 | ✅ REFUSED |
| Q13 | NE | `casualties_death` | यस बाढी विपद्मा कति मानिसहरूको मृत्यु भयो? | ✅ | 0 | ✅ REFUSED |
| Q14 | EN | `casualties_count` | What is the exact death toll and casualty count? | ✅ | 0 | ✅ REFUSED |
| Q14 | NE | `casualties_count` | हताहत र मृतकको संख्या कति छ? | ✅ | 0 | ✅ REFUSED |
| Q15 | EN | `casualties_injured` | How many civilians were injured? | ✅ | 0 | ✅ REFUSED |
| Q15 | NE | `casualties_injured` | कति नागरिक घाइते भएका छन्? | ✅ | 0 | ✅ REFUSED |
| Q16 | EN | `weather_tomorrow` | Will it rain tomorrow according to the weather forecast? | ✅ | 0 | ✅ REFUSED |
| Q16 | NE | `weather_tomorrow` | मौसम पूर्वानुमान अनुसार भोलि पानी पर्नेछ? | ✅ | 0 | ✅ REFUSED |
| Q17 | EN | `weather_next_week` | What is the precipitation forecast for next week? | ✅ | 0 | ✅ REFUSED |
| Q17 | NE | `weather_next_week` | आउँदो हप्ताको वर्षा पूर्वानुमान कस्तो छ? | ✅ | 0 | ✅ REFUSED |
| Q18 | EN | `water_depth_meters` | What is the water depth in meters at the river bed? | ✅ | 0 | ✅ REFUSED |
| Q18 | NE | `water_depth_meters` | नदीको सतहमा पानीको गहिराई कति मिटर छ? | ✅ | 0 | ✅ REFUSED |
| Q19 | EN | `bathymetry_profile` | Provide river bathymetry and underwater depth profile. | ✅ | 0 | ✅ REFUSED |
| Q19 | NE | `bathymetry_profile` | नदीको बाथिमेट्री र पानीको गहिराई प्रोफाइल दिनुहोस्। | ✅ | 0 | ✅ REFUSED |
| Q20 | EN | `unobserved_swath` | What is the damage status outside the satellite swath in unsurveyed villages? | ✅ | 0 | ✅ REFUSED |
| Q20 | NE | `unobserved_swath` | उपग्रह कभरेज बाहिरका गाउँहरूमा क्षति कस्तो छ? | ✅ | 0 | ✅ REFUSED |

---

## 3. Scientific Methodology & Audit Integrity
1. **Pre-Event Data Boundary**: The copilot strictly accesses verified `FactStore` entries derived from pre-event OSM (`≤ 2026-07-27`) and Sentinel-1 SAR change detections.
2. **Zero Synthetic Demographic Contamination**: Demographic hallucination is prohibited; population surfaces are absent from memory.
3. **Devanagari Numerals & Unicode Mapping**: Number extraction transparently normalizes Devanagari digits (`०-९`) to ASCII floats to enforce invariant equality.
4. **Out-of-Scope Integrity**: The copilot refuses to invent casualty numbers, future weather forecasts, or bathymetry depths, returning the explicit humanitarian notice: `Data Not Available in Satellite Maps / उपग्रह नक्सामा उपलब्ध छैन`.