"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — PRODUCT CLASSIFICATION
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Industrial product classification based on operational fingerprints.

CLASSIFICATION APPROACH
═══════════════════════

Products are classified by analyzing:
1. Operation sequence (op_codes)
2. Processing times
3. Machines used
4. Route taken
5. Material complexity

PRODUCT TYPES
═════════════

VIDRO_DUPLO (Double Glazing):
- Standard double pane unit
- Ops: CUT, EDGE, WASH, ASSEMBLY
- Machines: cutting table, edger, washer, IG line
- Typical time: 30-45 min

VIDRO_TRIPLO (Triple Glazing):
- Three pane unit with enhanced insulation
- Ops: CUT, EDGE, WASH, COATING, ASSEMBLY, SEAL
- Additional: coating machine
- Typical time: 45-60 min

VIDRO_LAMINADO (Laminated Glass):
- Safety glass with interlayer
- Ops: CUT, CLEAN, LAMINATE, AUTOCLAVE, TRIM
- Requires: autoclave process
- Typical time: 60-90 min

VIDRO_TEMPERADO (Tempered Glass):
- Heat-strengthened glass
- Ops: CUT, EDGE, WASH, FURNACE, COOL
- Requires: tempering furnace
- Typical time: 40-60 min

FINGERPRINTING
══════════════

A fingerprint uniquely identifies a product type:

    F = hash(ops_sequence, machine_set, time_class, route_id)

Similarity is computed using Jaccard index for ops and machines:

    J(A, B) = |A ∩ B| / |A ∪ B|

R&D / SIFIDE: WP7 - Product Intelligence
────────────────────────────────────────
- Hypothesis H7.1: Fingerprinting achieves >95% classification accuracy
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# ENUMS AND CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class ProductType(str, Enum):
    """Industrial glass product types."""
    VIDRO_DUPLO = "vidro_duplo"           # Double glazing
    VIDRO_TRIPLO = "vidro_triplo"         # Triple glazing
    VIDRO_LAMINADO = "vidro_laminado"     # Laminated glass
    VIDRO_TEMPERADO = "vidro_temperado"   # Tempered glass
    VIDRO_SIMPLES = "vidro_simples"       # Single pane
    OUTRO = "outro"                        # Other/unclassified


class TimeClass(str, Enum):
    """Processing time classification."""
    FAST = "fast"           # < 30 min
    STANDARD = "standard"   # 30-60 min
    SLOW = "slow"          # 60-120 min
    VERY_SLOW = "very_slow" # > 120 min


# Typical operation signatures for each product type
PRODUCT_SIGNATURES = {
    ProductType.VIDRO_DUPLO: {
        'required_ops': {'CUT', 'EDGE', 'WASH'},
        'optional_ops': {'ASSEMBLE', 'SEAL', 'INSPECT'},
        'forbidden_ops': {'AUTOCLAVE', 'LAMINATE', 'FURNACE'},
        'typical_time_range': (30, 60),  # minutes
        'min_ops': 3,
        'max_ops': 6,
    },
    ProductType.VIDRO_TRIPLO: {
        'required_ops': {'CUT', 'EDGE', 'WASH', 'COAT'},
        'optional_ops': {'ASSEMBLE', 'SEAL', 'INSPECT', 'GAS'},
        'forbidden_ops': {'AUTOCLAVE', 'LAMINATE'},
        'typical_time_range': (45, 90),
        'min_ops': 4,
        'max_ops': 8,
    },
    ProductType.VIDRO_LAMINADO: {
        'required_ops': {'CUT', 'CLEAN', 'LAMINATE'},
        'optional_ops': {'AUTOCLAVE', 'TRIM', 'EDGE', 'INSPECT'},
        'forbidden_ops': {'FURNACE'},
        'typical_time_range': (60, 120),
        'min_ops': 3,
        'max_ops': 7,
    },
    ProductType.VIDRO_TEMPERADO: {
        'required_ops': {'CUT', 'EDGE', 'FURNACE'},
        'optional_ops': {'WASH', 'COOL', 'INSPECT', 'DRILL'},
        'forbidden_ops': {'LAMINATE', 'AUTOCLAVE'},
        'typical_time_range': (40, 80),
        'min_ops': 3,
        'max_ops': 6,
    },
    ProductType.VIDRO_SIMPLES: {
        'required_ops': {'CUT'},
        'optional_ops': {'EDGE', 'WASH', 'INSPECT'},
        'forbidden_ops': {'LAMINATE', 'AUTOCLAVE', 'FURNACE'},
        'typical_time_range': (10, 30),
        'min_ops': 1,
        'max_ops': 4,
    },
}

# Machine indicators for classification
MACHINE_INDICATORS = {
    'autoclave': ProductType.VIDRO_LAMINADO,
    'laminator': ProductType.VIDRO_LAMINADO,
    'furnace': ProductType.VIDRO_TEMPERADO,
    'tempering': ProductType.VIDRO_TEMPERADO,
    'coating': ProductType.VIDRO_TRIPLO,
    'ig_line': ProductType.VIDRO_DUPLO,
}


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ProductFingerprint:
    """
    Unique fingerprint for a product based on its manufacturing characteristics.
    
    Used for classification and similarity analysis.
    """
    article_id: str
    
    # Operation characteristics
    operations: List[str]
    operation_set: Set[str] = field(default_factory=set)
    num_operations: int = 0
    
    # Machine characteristics
    machines: List[str] = field(default_factory=list)
    machine_set: Set[str] = field(default_factory=set)
    
    # Time characteristics
    total_time_min: float = 0.0
    avg_op_time_min: float = 0.0
    time_class: TimeClass = TimeClass.STANDARD
    
    # Route
    route_id: Optional[str] = None
    route_label: Optional[str] = None
    
    # Classification
    product_type: ProductType = ProductType.OUTRO
    confidence: float = 0.0
    
    # Hash
    fingerprint_hash: str = ""
    
    def __post_init__(self):
        if isinstance(self.operations, list):
            self.operation_set = set(self.operations)
            self.num_operations = len(self.operations)
        if isinstance(self.machines, list):
            self.machine_set = set(self.machines)
        self._compute_hash()
    
    def _compute_hash(self):
        """Compute unique hash for this fingerprint."""
        data = f"{sorted(self.operation_set)}|{sorted(self.machine_set)}|{self.time_class.value}|{self.route_id}"
        self.fingerprint_hash = hashlib.md5(data.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'article_id': self.article_id,
            'operations': self.operations,
            'num_operations': self.num_operations,
            'machines': self.machines,
            'total_time_min': round(self.total_time_min, 1),
            'avg_op_time_min': round(self.avg_op_time_min, 1),
            'time_class': self.time_class.value,
            'route_id': self.route_id,
            'route_label': self.route_label,
            'product_type': self.product_type.value,
            'confidence': round(self.confidence, 3),
            'fingerprint_hash': self.fingerprint_hash,
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def _normalize_op_code(op_code: str) -> str:
    """Normalize operation code for matching."""
    if pd.isna(op_code):
        return ''
    op = str(op_code).upper().strip()
    # Map common variations
    mappings = {
        'CORTE': 'CUT',
        'CORTAR': 'CUT',
        'POLIR': 'EDGE',
        'POLIMENTO': 'EDGE',
        'LAVAR': 'WASH',
        'LAVAGEM': 'WASH',
        'MONTAR': 'ASSEMBLE',
        'MONTAGEM': 'ASSEMBLE',
        'SELAR': 'SEAL',
        'REVESTIR': 'COAT',
        'REVESTIMENTO': 'COAT',
        'LAMINAR': 'LAMINATE',
        'LAMINAGEM': 'LAMINATE',
        'TEMPERAR': 'FURNACE',
        'TÊMPERA': 'FURNACE',
    }
    return mappings.get(op, op)


def _classify_time(total_min: float) -> TimeClass:
    """Classify total processing time."""
    if total_min < 30:
        return TimeClass.FAST
    elif total_min < 60:
        return TimeClass.STANDARD
    elif total_min < 120:
        return TimeClass.SLOW
    else:
        return TimeClass.VERY_SLOW


def _compute_signature_score(
    ops: Set[str],
    machines: Set[str],
    total_time: float,
    signature: Dict[str, Any]
) -> float:
    """
    Compute match score between product and signature.
    
    Score = w1 * required_match + w2 * forbidden_penalty + w3 * time_match
    """
    score = 0.0
    
    # Required operations (40% weight)
    required = signature.get('required_ops', set())
    if required:
        match_pct = len(ops & required) / len(required)
        score += 0.4 * match_pct
    else:
        score += 0.4
    
    # Forbidden operations (30% weight - penalty)
    forbidden = signature.get('forbidden_ops', set())
    if forbidden:
        violations = len(ops & forbidden)
        if violations > 0:
            score -= 0.3 * (violations / len(forbidden))
        else:
            score += 0.3
    else:
        score += 0.3
    
    # Time range (20% weight)
    time_range = signature.get('typical_time_range', (0, 1000))
    if time_range[0] <= total_time <= time_range[1]:
        score += 0.2
    elif total_time < time_range[0]:
        score += 0.1 * (total_time / time_range[0])
    else:
        score += 0.1 * (time_range[1] / total_time)
    
    # Operation count (10% weight)
    min_ops = signature.get('min_ops', 1)
    max_ops = signature.get('max_ops', 10)
    num_ops = len(ops)
    if min_ops <= num_ops <= max_ops:
        score += 0.1
    else:
        score += 0.05
    
    return max(0, min(1, score))


def classify_product(
    article_id: str,
    operations: List[str],
    machines: Optional[List[str]] = None,
    total_time_min: float = 0.0,
    route_id: Optional[str] = None
) -> Tuple[ProductType, float]:
    """
    Classify a product based on its operational characteristics.
    
    Uses signature matching with weighted scoring.
    
    Args:
        article_id: Article identifier
        operations: List of operation codes
        machines: Optional list of machines used
        total_time_min: Total processing time
        route_id: Route identifier
    
    Returns:
        (ProductType, confidence_score)
    """
    # Normalize operations
    ops = {_normalize_op_code(op) for op in operations if op}
    ops.discard('')
    
    if not ops:
        return ProductType.OUTRO, 0.0
    
    # Check machine indicators first
    machines_lower = {m.lower() for m in (machines or [])}
    for indicator, ptype in MACHINE_INDICATORS.items():
        if any(indicator in m for m in machines_lower):
            # Machine strongly suggests type
            sig = PRODUCT_SIGNATURES.get(ptype, {})
            score = _compute_signature_score(ops, set(machines or []), total_time_min, sig)
            return ptype, max(0.7, score)
    
    # Score against all signatures
    scores = {}
    for ptype, signature in PRODUCT_SIGNATURES.items():
        score = _compute_signature_score(ops, set(machines or []), total_time_min, signature)
        scores[ptype] = score
    
    # Get best match
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    
    # Require minimum confidence
    if best_score < 0.3:
        return ProductType.OUTRO, best_score
    
    return best_type, best_score


def build_product_fingerprint(
    article_id: str,
    routing_df: pd.DataFrame,
    plan_df: Optional[pd.DataFrame] = None,
    article_col: str = 'article_id',
    op_col: str = 'op_code',
    machine_col: str = 'primary_machine_id',
    time_col: str = 'base_time_per_unit_min',
    route_col: str = 'route_id',
    route_label_col: str = 'route_label'
) -> ProductFingerprint:
    """
    Build a fingerprint for an article from routing data.
    
    Args:
        article_id: Article to fingerprint
        routing_df: Routing DataFrame
        plan_df: Optional production plan for actual times
        article_col: Column for article ID
        op_col: Column for operation code
        machine_col: Column for machine ID
        time_col: Column for processing time
        route_col: Column for route ID
        route_label_col: Column for route label
    
    Returns:
        ProductFingerprint with classification
    """
    # Filter routing for this article
    if article_col not in routing_df.columns:
        return ProductFingerprint(article_id=article_id, operations=[], product_type=ProductType.OUTRO)
    
    article_routing = routing_df[routing_df[article_col] == article_id]
    
    if article_routing.empty:
        return ProductFingerprint(article_id=article_id, operations=[], product_type=ProductType.OUTRO)
    
    # Extract operations
    operations = []
    if op_col in article_routing.columns:
        operations = article_routing[op_col].dropna().tolist()
    
    # Extract machines
    machines = []
    if machine_col in article_routing.columns:
        machines = article_routing[machine_col].dropna().tolist()
    
    # Calculate time
    total_time = 0.0
    if time_col in article_routing.columns:
        total_time = float(article_routing[time_col].sum())
    
    # Get route info
    route_id = None
    route_label = None
    if route_col in article_routing.columns:
        routes = article_routing[route_col].dropna()
        if not routes.empty:
            route_id = str(routes.iloc[0])
    if route_label_col in article_routing.columns:
        labels = article_routing[route_label_col].dropna()
        if not labels.empty:
            route_label = str(labels.iloc[0])
    
    # Classify
    product_type, confidence = classify_product(
        article_id=article_id,
        operations=operations,
        machines=machines,
        total_time_min=total_time,
        route_id=route_id
    )
    
    # Build fingerprint
    fp = ProductFingerprint(
        article_id=article_id,
        operations=operations,
        machines=machines,
        total_time_min=total_time,
        avg_op_time_min=total_time / len(operations) if operations else 0,
        time_class=_classify_time(total_time),
        route_id=route_id,
        route_label=route_label,
        product_type=product_type,
        confidence=confidence,
    )
    
    return fp


def classify_all_products(
    routing_df: pd.DataFrame,
    plan_df: Optional[pd.DataFrame] = None,
    article_col: str = 'article_id',
    **kwargs
) -> Dict[str, ProductFingerprint]:
    """
    Classify all products in routing data.
    
    Returns:
        Dict mapping article_id -> ProductFingerprint
    """
    fingerprints = {}
    
    if article_col not in routing_df.columns:
        return fingerprints
    
    articles = routing_df[article_col].unique()
    
    for article_id in articles:
        if pd.isna(article_id):
            continue
        
        fp = build_product_fingerprint(
            article_id=str(article_id),
            routing_df=routing_df,
            plan_df=plan_df,
            article_col=article_col,
            **kwargs
        )
        fingerprints[str(article_id)] = fp
    
    logger.info(f"Classified {len(fingerprints)} products")
    
    # Log distribution
    type_counts = {}
    for fp in fingerprints.values():
        ptype = fp.product_type.value
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    logger.info(f"Type distribution: {type_counts}")
    
    return fingerprints


def get_product_family(article_id: str) -> str:
    """
    Extract product family from article ID.
    
    E.g., ART-100 -> ART-1xx, VIDRO-2345 -> VIDRO-23xx
    """
    if not article_id:
        return "UNKNOWN"
    
    article_id = str(article_id)
    
    # Pattern: PREFIX-NUMBERS
    if '-' in article_id:
        parts = article_id.split('-')
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ''
        
        if suffix and suffix[0].isdigit():
            # Take first 2 digits + 'xx'
            family_num = suffix[:2] if len(suffix) >= 2 else suffix[0]
            return f"{prefix}-{family_num}xx"
        return f"{prefix}-xxx"
    
    # Just take first 6 chars
    return article_id[:6] + "..."


def compute_fingerprint_similarity(fp1: ProductFingerprint, fp2: ProductFingerprint) -> float:
    """
    Compute similarity between two fingerprints using Jaccard index.
    
    Similarity = 0.5 * J(ops) + 0.3 * J(machines) + 0.2 * time_similarity
    """
    # Operations Jaccard
    union_ops = fp1.operation_set | fp2.operation_set
    inter_ops = fp1.operation_set & fp2.operation_set
    j_ops = len(inter_ops) / len(union_ops) if union_ops else 0
    
    # Machines Jaccard
    union_mach = fp1.machine_set | fp2.machine_set
    inter_mach = fp1.machine_set & fp2.machine_set
    j_mach = len(inter_mach) / len(union_mach) if union_mach else 0
    
    # Time similarity
    max_time = max(fp1.total_time_min, fp2.total_time_min)
    min_time = min(fp1.total_time_min, fp2.total_time_min)
    time_sim = min_time / max_time if max_time > 0 else 1.0
    
    return 0.5 * j_ops + 0.3 * j_mach + 0.2 * time_sim


def group_products_by_type(
    fingerprints: Dict[str, ProductFingerprint]
) -> Dict[ProductType, List[str]]:
    """
    Group products by their classified type.
    
    Returns:
        Dict mapping ProductType -> list of article_ids
    """
    groups = {ptype: [] for ptype in ProductType}
    
    for article_id, fp in fingerprints.items():
        groups[fp.product_type].append(article_id)
    
    return groups



