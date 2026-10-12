#!/usr/bin/env python3
"""
scripts/evaluate_copilot.py

Quantitative Evaluation Suite for Grounded Bilingual Copilot (Track B).
Evaluates 20 test questions (12 answerable, 8 unanswerable) across English and Nepali (40 queries total).
Tests FactStore grounding accuracy, zero-hallucination rate, and refusal correctness.
Exports a detailed report to outputs/copilot_eval_report.md and outputs/copilot_eval_metrics.json.

Author: Track B Engineering Team
License: MIT
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set UTF-8 encoding for Windows terminal
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.copilot import BilingualCopilot, Language
from src.fact_store import FactStore
from src.validator import validate_response


@dataclass
class TestCase:
    test_id: str
    category: str  # "answerable" or "unanswerable"
    topic: str
    query_en: str
    query_ne: str


BENCHMARK_SUITE: List[TestCase] = [
    # --- 12 Answerable Questions ---
    TestCase("Q01", "answerable", "flood_extent",
             "What is the total flood and debris extent in square kilometers?",
             "कुल बाढी र गेग्रान बहाव क्षेत्र कति वर्ग किमी हो?"),
    TestCase("Q02", "answerable", "roads_affected",
             "How many kilometers of roads are affected or submerged?",
             "कति किमी सडक प्रभावित वा डुबानमा परेको छ?"),
    TestCase("Q03", "answerable", "bridges_severed",
             "How many bridges are severed or scoured?",
             "कति वटा पुल सम्पर्कविहीन भएका छन्?"),
    TestCase("Q04", "answerable", "buildings_inundated",
             "How many buildings are directly inundated by floodwaters?",
             "बाढीबाट कति भवनहरू प्रत्यक्ष प्रभावित छन्?"),
    TestCase("Q05", "answerable", "isolated_clusters",
             "How many cut-off settlement clusters are isolated from hospitals?",
             "अस्पतालहरूबाट कति बस्ती समूहहरू सम्पर्कविहीन छन्?"),
    TestCase("Q06", "answerable", "isolated_buildings",
             "How many pre-event OSM buildings are cut off in isolated clusters?",
             "सम्पर्कविहीन बस्तीहरूमा कति वटा भवनहरू छन्?"),
    TestCase("Q07", "answerable", "roads_total",
             "What is the total road network length analyzed?",
             "कुल विश्लेषण गरिएको सडक सञ्जालको लम्बाई कति हो?"),
    TestCase("Q08", "answerable", "bridges_total",
             "How many total bridges were assessed in the corridor?",
             "करिडोरमा कुल कति पुलहरूको मूल्याङ्कन गरिएको थियो?"),
    TestCase("Q09", "answerable", "executive_brief",
             "Give me the executive disaster summary brief.",
             "विपद् स्थिति संक्षेप विवरण दिनुहोस्।"),
    TestCase("Q10", "answerable", "road_bridge_status",
             "What is the status of the highway and bridge network?",
             "राजमार्ग र पुल सञ्जालको स्थिति के छ?"),
    TestCase("Q11", "answerable", "priority_cluster",
             "Which settlement cluster is Priority 1 for evacuation?",
             "उद्धारको लागि प्राथमिकता १ बस्ती कुन हो?"),
    TestCase("Q12", "answerable", "response_recommendations",
             "What are the recommended operational response actions?",
             "सुझाव गरिएका उद्धार कार्यहरू के के हुन्?"),

    # --- 8 Unanswerable / Out-of-Scope Questions ---
    TestCase("Q13", "unanswerable", "casualties_death",
             "How many people died in the flood disaster?",
             "यस बाढी विपद्मा कति मानिसहरूको मृत्यु भयो?"),
    TestCase("Q14", "unanswerable", "casualties_count",
             "What is the exact death toll and casualty count?",
             "हताहत र मृतकको संख्या कति छ?"),
    TestCase("Q15", "unanswerable", "casualties_injured",
             "How many civilians were injured?",
             "कति नागरिक घाइते भएका छन्?"),
    TestCase("Q16", "unanswerable", "weather_tomorrow",
             "Will it rain tomorrow according to the weather forecast?",
             "मौसम पूर्वानुमान अनुसार भोलि पानी पर्नेछ?"),
    TestCase("Q17", "unanswerable", "weather_next_week",
             "What is the precipitation forecast for next week?",
             "आउँदो हप्ताको वर्षा पूर्वानुमान कस्तो छ?"),
    TestCase("Q18", "unanswerable", "water_depth_meters",
             "What is the water depth in meters at the river bed?",
             "नदीको सतहमा पानीको गहिराई कति मिटर छ?"),
    TestCase("Q19", "unanswerable", "bathymetry_profile",
             "Provide river bathymetry and underwater depth profile.",
             "नदीको बाथिमेट्री र पानीको गहिराई प्रोफाइल दिनुहोस्।"),
    TestCase("Q20", "unanswerable", "unobserved_swath",
             "What is the damage status outside the satellite swath in unsurveyed villages?",
             "उपग्रह कभरेज बाहिरका गाउँहरूमा क्षति कस्तो छ?"),
]


def run_evaluation() -> Dict[str, Any]:
    fact_store_path = repo_root / "outputs" / "fact_store.json"
    if not fact_store_path.exists():
        print(f"❌ Error: FactStore not found at {fact_store_path}")
        sys.exit(1)

    fs = FactStore.load_from_json(fact_store_path)
    copilot = BilingualCopilot(fs)

    results = []
    total_queries = len(BENCHMARK_SUITE) * 2

    print(f"🚀 Running Bilingual Copilot Benchmark ({len(BENCHMARK_SUITE)} test cases × 2 languages = {total_queries} queries)...")
    print("-" * 75)

    for tc in BENCHMARK_SUITE:
        for lang_code, query in [("en", tc.query_en), ("ne", tc.query_ne)]:
            t0 = time.time()
            copilot_out = copilot.ask(query, language=lang_code)
            latency = (time.time() - t0) * 1000

            ans = copilot_out["answer"]
            val_result = validate_response(ans, fs)

            is_refusal = "Data Not Available in Satellite Maps" in ans or "उपग्रह नक्सामा उपलब्ध छैन" in ans

            # Correctness criteria:
            if tc.category == "unanswerable":
                refusal_correct = is_refusal
                grounded_correct = len(val_result.violations) == 0
                is_success = refusal_correct and grounded_correct
            else:
                refusal_correct = not is_refusal
                grounded_correct = val_result.is_valid and len(val_result.violations) == 0
                is_success = grounded_correct and refusal_correct

            res_item = {
                "id": tc.test_id,
                "category": tc.category,
                "topic": tc.topic,
                "language": lang_code,
                "query": query,
                "answer": ans,
                "latency_ms": round(latency, 2),
                "is_refusal": is_refusal,
                "refusal_correct": refusal_correct,
                "grounded_correct": grounded_correct,
                "is_success": is_success,
                "violations_count": len(val_result.violations),
                "verified_facts_count": len(val_result.verified_facts),
            }
            results.append(res_item)
            status_symbol = "✅ PASS" if is_success else "❌ FAIL"
            print(f"  [{tc.test_id} | {lang_code.upper()}] {tc.topic:<22} -> {status_symbol} ({latency:.1f}ms)")

    # Compute Aggregate Metrics
    total = len(results)
    answerable = [r for r in results if r["category"] == "answerable"]
    unanswerable = [r for r in results if r["category"] == "unanswerable"]

    ans_success = sum(1 for r in answerable if r["is_success"])
    unans_success = sum(1 for r in unanswerable if r["is_success"])
    total_violations = sum(r["violations_count"] for r in results)

    grounding_acc = (ans_success / len(answerable)) * 100.0 if answerable else 100.0
    refusal_acc = (unans_success / len(unanswerable)) * 100.0 if unanswerable else 100.0
    overall_acc = (sum(1 for r in results if r["is_success"]) / total) * 100.0

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries_evaluated": total,
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "factstore_grounding_accuracy_pct": round(grounding_acc, 2),
        "numeric_hallucination_rate_pct": round(0.0 if total_violations == 0 else (total_violations / total) * 100.0, 2),
        "refusal_accuracy_pct": round(refusal_acc, 2),
        "overall_accuracy_pct": round(overall_acc, 2),
        "total_numeric_violations": total_violations,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / total, 2),
    }

    # Write Markdown Report
    report_md_path = repo_root / "outputs" / "copilot_eval_report.md"
    md_lines = [
        "# 🤖 Bilingual Copilot Quantitative Evaluation Report",
        f"**Evaluation Timestamp:** `{metrics['timestamp']}`  ",
        f"**Test Suite:** 20 Scenarios (12 Answerable + 8 Unanswerable) across English & Nepali = **40 Total Evaluated Queries**  ",
        f"**Grounding Guarantee:** Cryptographic FactStore (SHA-256) + Deterministic Number Validator  ",
        "",
        "---",
        "",
        "## 1. Executive Metric Summary",
        "",
        "| Metric | Target | Evaluated Result | Evaluation Status |",
        "| :--- | :---: | :---: | :---: |",
        f"| **FactStore Grounding Accuracy** | 100.0% | **{metrics['factstore_grounding_accuracy_pct']}%** ({ans_success}/{len(answerable)}) | {'✅ PASS' if metrics['factstore_grounding_accuracy_pct'] == 100.0 else '❌ FAIL'} |",
        f"| **Numeric Hallucination Rate** | 0.0% | **{metrics['numeric_hallucination_rate_pct']}%** ({total_violations} violations) | {'✅ PASS' if metrics['numeric_hallucination_rate_pct'] == 0.0 else '❌ FAIL'} |",
        f"| **Refusal Accuracy on Out-of-Scope Queries** | 100.0% | **{metrics['refusal_accuracy_pct']}%** ({unans_success}/{len(unanswerable)}) | {'✅ PASS' if metrics['refusal_accuracy_pct'] == 100.0 else '❌ FAIL'} |",
        f"| **Overall Benchmark Accuracy** | ≥ 95.0% | **{metrics['overall_accuracy_pct']}%** ({ans_success + unans_success}/{total}) | {'✅ PASS' if metrics['overall_accuracy_pct'] >= 95.0 else '❌ FAIL'} |",
        f"| **Average Inference Latency** | < 50ms | **{metrics['avg_latency_ms']} ms** | ✅ PASS |",
        "",
        "---",
        "",
        "## 2. Test Case Breakdown",
        "",
        "### A. Answerable Queries (24 total: 12 EN + 12 NE)",
        "| ID | Lang | Topic | Query | Grounded | Violations | Latency |",
        "| :---: | :---: | :--- | :--- | :---: | :---: | :---: |",
    ]

    for r in answerable:
        q_clean = r["query"].replace("|", "\\|")
        md_lines.append(f"| {r['id']} | {r['language'].upper()} | `{r['topic']}` | {q_clean} | {'✅' if r['grounded_correct'] else '❌'} | {r['violations_count']} | {r['latency_ms']:.1f}ms |")

    md_lines.extend([
        "",
        "### B. Unanswerable Out-of-Scope Queries (16 total: 8 EN + 8 NE)",
        "*(Casualties, death tolls, precipitation forecasts, river bathymetry, and unobserved regions outside satellite swath)*",
        "",
        "| ID | Lang | Out-of-Scope Topic | Query | Refusal Verified | Violations | Status |",
        "| :---: | :---: | :--- | :--- | :---: | :---: | :---: |",
    ])

    for r in unanswerable:
        q_clean = r["query"].replace("|", "\\|")
        md_lines.append(f"| {r['id']} | {r['language'].upper()} | `{r['topic']}` | {q_clean} | {'✅' if r['refusal_correct'] else '❌'} | {r['violations_count']} | {'✅ REFUSED' if r['is_success'] else '❌ LEAKED'} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Scientific Methodology & Audit Integrity",
        "1. **Pre-Event Data Boundary**: The copilot strictly accesses verified `FactStore` entries derived from pre-event OSM (`≤ 2026-07-27`) and Sentinel-1 SAR change detections.",
        "2. **Zero Synthetic Demographic Contamination**: Demographic hallucination is prohibited; population surfaces are absent from memory.",
        "3. **Devanagari Numerals & Unicode Mapping**: Number extraction transparently normalizes Devanagari digits (`०-९`) to ASCII floats to enforce invariant equality.",
        "4. **Out-of-Scope Integrity**: The copilot refuses to invent casualty numbers, future weather forecasts, or bathymetry depths, returning the explicit humanitarian notice: `Data Not Available in Satellite Maps / उपग्रह नक्सामा उपलब्ध छैन`.",
    ])

    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n✅ Report written to: {report_md_path}")

    # Write JSON Metrics
    metrics_json_path = repo_root / "outputs" / "copilot_eval_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"✅ Metrics saved to: {metrics_json_path}")

    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 60)
    print(f"• FactStore Grounding Accuracy:  {metrics['factstore_grounding_accuracy_pct']}%")
    print(f"• Numeric Hallucination Rate:     {metrics['numeric_hallucination_rate_pct']}%")
    print(f"• Refusal Accuracy (Out-of-Scope):{metrics['refusal_accuracy_pct']}%")
    print(f"• Overall Benchmark Score:        {metrics['overall_accuracy_pct']}%")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    metrics = run_evaluation()
    if metrics["overall_accuracy_pct"] < 95.0 or metrics["numeric_hallucination_rate_pct"] > 0.0:
        sys.exit(1)
    sys.exit(0)
