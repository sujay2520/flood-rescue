"""
src/validator.py

Deterministic Number Hallucination Validator for Bilingual Reports (English & Nepali).
Extracts all numbers (Arabic digits & Devanagari numerals), checks against the FactStore,
and flags any ungrounded figures with sentence context.

Author: Track B Engineering Team
License: MIT
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

# Mapping of Devanagari (Nepali) digits to ASCII digits
DEVANAGARI_DIGIT_MAP: Dict[str, str] = {
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

# Whitelist of known baseline constants allowed without FactStore entry
DEFAULT_BASELINE_CONSTANTS: Set[float] = {
    0.0,     # Zero / null count
    2026.0,  # Event year
    15.0,    # Slope plausibility threshold (degrees)
    12.0,    # Sentinel-1 orbit repeat cycle (days)
    40.0,    # Bridge approach buffer (meters)
    1.0, 2.0, 3.0, 4.0, 5.0,  # Section numbers
    16.0, 26.0, 27.0, 28.0,  # Pre/Post event & disaster dates
    85.0,    # S1 Relative Orbit Track 85
    927.0,   # EMSR927 case study identifier
    2.5,     # dB backscatter threshold
    10.0,    # 10m spatial resolution
    30.0,    # DEM-30 resolution
    100.0,   # Percentage / meter scaling
}

DEFAULT_REL_TOLERANCE: float = 0.03  # 3% relative tolerance for string rounding

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?।])\s+")

NUMBER_PATTERN = re.compile(
    r"""
    (?P<full_match>
        (?P<number>
            [+-]?
            (?:
                (?:[\d०-९]+(?:[.,][\d०-९]+)?)
                |
                [.,][\d०-९]+
            )
        )
        (?P<unit>\s*[a-zA-Z%°km²m²वटा]*\b)?
    )
    """,
    re.VERBOSE | re.UNICODE,
)


class VerifiedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number_in_text: str
    normalized_value: float
    fact_id: str
    fact_value: float
    fact_description: Optional[str] = None
    unit_in_text: str = ""


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number_in_text: str
    normalized_value: float
    context_sentence: str
    reason: str
    unit_in_text: str = ""


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_valid: bool
    total_numbers_found: int
    verified_facts: List[VerifiedFact] = Field(default_factory=list)
    violations: List[Violation] = Field(default_factory=list)
    grounding_score: float = Field(ge=0.0, le=1.0)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


def translate_nepali_numerals(text: str) -> str:
    """Replaces Devanagari digits with ASCII equivalents."""
    return "".join(DEVANAGARI_DIGIT_MAP.get(ch, ch) for ch in text)


def normalize_number_string(num_str: str) -> float:
    """Clean comma separators and parse float."""
    cleaned = num_str.replace(",", "").replace(" ", "").rstrip(".")
    return float(cleaned)


def split_sentences(text: str) -> List[str]:
    """Splits bilingual text into sentences (handles English '.', Nepali '।', etc.)."""
    cleaned = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in SENTENCE_SPLIT_REGEX.split(cleaned) if s.strip()]


def find_sentence_context(text: str, match_start: int, match_end: int) -> str:
    """Extract context sentence surrounding the match."""
    sentences = split_sentences(text)
    char_idx = 0
    for sent in sentences:
        sent_len = len(sent)
        if char_idx <= match_start < char_idx + sent_len + 10:
            return sent
        char_idx += sent_len + 1
    start = max(0, match_start - 60)
    end = min(len(text), match_end + 60)
    return text[start:end].strip()


class NumberValidator:
    """
    Deterministic validator that verifies all numbers against FactStore.
    """

    def __init__(
        self,
        fact_store: Any,
        baseline_constants: Optional[Set[float]] = None,
        rel_tolerance: float = DEFAULT_REL_TOLERANCE,
    ):
        self.fact_store = fact_store
        self.baseline_constants = baseline_constants or DEFAULT_BASELINE_CONSTANTS
        self.rel_tolerance = rel_tolerance

    def _is_baseline_constant(self, value: float) -> bool:
        return any(abs(abs(value) - c) < 1e-4 for c in self.baseline_constants)

    def _match_fact_store(self, value: float) -> Optional[Dict[str, Any]]:
        """Look up value in FactStore."""
        if hasattr(self.fact_store, "get_all_facts"):
            all_facts = self.fact_store.get_all_facts()
            for fid, rec in all_facts.items():
                if isinstance(rec.value, (int, float)):
                    fval = float(rec.value)
                    if fval == 0.0 and value == 0.0:
                        return {"fact_id": fid, "value": fval, "description": rec.description_en}
                    if fval != 0.0 and abs(value - fval) / abs(fval) <= self.rel_tolerance:
                        return {"fact_id": fid, "value": fval, "description": rec.description_en}
        elif hasattr(self.fact_store, "get_all_numeric_values"):
            nums = self.fact_store.get_all_numeric_values()
            for fid, fval in nums.items():
                if fval != 0.0 and abs(value - fval) / abs(fval) <= self.rel_tolerance:
                    return {"fact_id": fid, "value": fval, "description": fid}
        return None

    def extract_numbers(self, text: str) -> List[Tuple[str, float, str, int, int]]:
        # Strip code spans (e.g. `hash`, `EMSR927`, `code`) so hashes aren't parsed as numbers
        cleaned_text = re.sub(r"`[^`]*`", " ", text)
        # Strip alphanumeric tokens like EMSR927, H21, C-band, DEM-30, Sentinel-1
        cleaned_text = re.sub(r"\b[A-Za-z]+[0-9]+[A-Za-z0-9]*\b", " ", cleaned_text)
        cleaned_text = re.sub(r"\b[A-Za-z]+-[0-9]+\b", " ", cleaned_text)
        cleaned_text = re.sub(r"\b[A-Za-z]+-[A-Za-z]+\b", " ", cleaned_text)

        results = []
        for match in NUMBER_PATTERN.finditer(cleaned_text):
            num_str_raw = match.group("number")
            unit_str = (match.group("unit") or "").strip()
            start, end = match.span("full_match")

            num_str_ascii = translate_nepali_numerals(num_str_raw)
            try:
                norm_val = normalize_number_string(num_str_ascii)
                results.append((num_str_raw, norm_val, unit_str, start, end))
            except ValueError:
                continue
        return results

    def validate(self, text: str) -> ValidationResult:
        if not text or not text.strip():
            return ValidationResult(is_valid=True, total_numbers_found=0, grounding_score=1.0)

        extracted = self.extract_numbers(text)
        verified: List[VerifiedFact] = []
        violations: List[Violation] = []

        for orig_num, norm_val, unit, start, end in extracted:
            context = find_sentence_context(text, start, end)

            # Check baseline constants
            if self._is_baseline_constant(norm_val):
                verified.append(VerifiedFact(
                    number_in_text=orig_num,
                    normalized_value=norm_val,
                    fact_id="BASELINE_CONSTANT",
                    fact_value=norm_val,
                    fact_description="Whitelisted constant (year, resolution, or slope threshold)",
                    unit_in_text=unit
                ))
                continue

            # Match FactStore
            match = self._match_fact_store(norm_val)
            if match:
                verified.append(VerifiedFact(
                    number_in_text=orig_num,
                    normalized_value=norm_val,
                    fact_id=match["fact_id"],
                    fact_value=float(match["value"]),
                    fact_description=match.get("description"),
                    unit_in_text=unit
                ))
            else:
                violations.append(Violation(
                    number_in_text=orig_num,
                    normalized_value=norm_val,
                    context_sentence=context,
                    reason=f"Ungrounded number: {norm_val} not in FactStore (tolerance {self.rel_tolerance*100:.1f}%)",
                    unit_in_text=unit
                ))

        total = len(extracted)
        valid_count = len(verified)
        score = valid_count / total if total > 0 else 1.0
        is_valid = len(violations) == 0

        return ValidationResult(
            is_valid=is_valid,
            total_numbers_found=total,
            verified_facts=verified,
            violations=violations,
            grounding_score=round(score, 4),
        )


def validate_response(text: str, fact_store: Any) -> ValidationResult:
    """Validate any bilingual string against the given FactStore."""
    validator = NumberValidator(fact_store)
    return validator.validate(text)


def main() -> int:
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Deterministic Fact Validator")
    parser.add_argument("--text", type=str, default="Flooding covered 4.12 km2 and cut off 72 buildings across 2 bridges in 2026.")
    args = parser.parse_args()

    # Mock fact store
    from .fact_store import FactStore, FactRecord
    facts = {
        "FLOOD_AREA_KM2": FactRecord(
            fact_id="FLOOD_AREA_KM2", value=4.12, unit="km2", source_layer="test",
            description_en="Flooded area", description_ne="बाढी क्षेत्र", provenance_hash="a"*64, timestamp="2026-08-28T00:00:00Z"
        ),
        "CUTOFF_BUILDINGS_COUNT": FactRecord(
            fact_id="CUTOFF_BUILDINGS_COUNT", value=72, unit="count", source_layer="test",
            description_en="Cutoff buildings", description_ne="भवन संख्या", provenance_hash="a"*64, timestamp="2026-08-28T00:00:00Z"
        ),
        "BRIDGES_SEVERED_COUNT": FactRecord(
            fact_id="BRIDGES_SEVERED_COUNT", value=2, unit="count", source_layer="test",
            description_en="Severed bridges", description_ne="पुल संख्या", provenance_hash="a"*64, timestamp="2026-08-28T00:00:00Z"
        ),
    }
    fs = FactStore(facts)

    print("=" * 70)
    print("Testing English text validation:")
    res_en = validate_response(args.text, fs)
    print(f"Valid: {res_en.is_valid}, Score: {res_en.grounding_score}")
    print(f"Verified: {len(res_en.verified_facts)}, Violations: {len(res_en.violations)}")

    print("\nTesting Nepali text validation:")
    nepali_text = "२०२६ को बाढीले ४.१२ वर्ग किमी डुबान गर्‍यो र ७२ वटा भवनहरू २ वटा पुल काटिँदा सम्पर्कविहीन भए।"
    res_ne = validate_response(nepali_text, fs)
    print(f"Valid: {res_ne.is_valid}, Score: {res_ne.grounding_score}")
    for vf in res_ne.verified_facts:
        print(f"  Verified: {vf.number_in_text} -> {vf.fact_id} ({vf.fact_value})")

    print("\nTesting Hallucination detection:")
    hallucinated_text = "The flood killed 950 people and damaged 8800 schools."
    res_bad = validate_response(hallucinated_text, fs)
    print(f"Valid: {res_bad.is_valid}, Score: {res_bad.grounding_score}")
    for viol in res_bad.violations:
        print(f"  Violation: {viol.number_in_text} in '{viol.context_sentence}'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
