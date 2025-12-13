"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — PRODUCT KPI ENGINE
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Industrial KPIs computed per product type and family.

KPI DEFINITIONS
═══════════════

1. PROCESSING TIME
   ────────────────
   
   μ_proc = (Σ processing_time) / n
   σ_proc = √(Σ(t - μ)² / (n-1))
   
   Six Sigma levels:
   - USL, LSL = μ ± 3σ
   - Cp = (USL - LSL) / 6σ

2. SETUP TIME
   ────────────
   
   μ_setup = (Σ setup_time) / n_setups
   
   Setup ratio = setup_time / (setup_time + processing_time)

3. LEAD TIME
   ────────────
   
   LT = T_complete - T_start
   
   Includes: processing + setup + queue + transport

4. WASTE / SCRAP
   ────────────────
   
   Scrap_rate = units_scrapped / units_produced
   
   Yield = 1 - scrap_rate

5. SIGNAL-TO-NOISE RATIO (SNR)
   ────────────────────────────
   
   SNR = Var(μ_by_product) / Var(residual)
   
   High SNR: Process is stable, variability is between products
   Low SNR: High within-product variability (process unstable)

6. PROCESS CAPABILITY
   ────────────────────
   
   Cp = (USL - LSL) / 6σ
   Cpk = min((USL - μ)/3σ, (μ - LSL)/3σ)
   
   Cp ≥ 1.33: Process capable
   Cp < 1.0: Process not capable

R&D / SIFIDE: WP7 - Product Intelligence
────────────────────────────────────────
- Hypothesis H7.3: Product type explains >60% of lead time variance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .product_classification import (
    ProductType,
    ProductFingerprint,
    classify_all_products,
    group_products_by_type,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class ProductKPIs:
    """
    KPIs for a single product (article).
    """
    article_id: str
    product_type: str
    
    # Processing time
    avg_processing_time_min: float = 0.0
    std_processing_time_min: float = 0.0
    min_processing_time_min: float = 0.0
    max_processing_time_min: float = 0.0
    
    # Setup time
    avg_setup_time_min: float = 0.0
    total_setup_time_min: float = 0.0
    setup_ratio: float = 0.0  # setup / (setup + processing)
    
    # Lead time
    avg_lead_time_min: float = 0.0
    std_lead_time_min: float = 0.0
    lead_time_hours: float = 0.0
    
    # Waste
    scrap_rate: float = 0.0
    yield_rate: float = 1.0
    
    # Volume
    total_units: float = 0.0
    total_orders: int = 0
    total_operations: int = 0
    
    # SNR
    snr_process: float = 1.0
    snr_level: str = "FAIR"
    
    # Process capability (if spec limits available)
    cp: Optional[float] = None
    cpk: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'article_id': self.article_id,
            'product_type': self.product_type,
            'avg_processing_time_min': round(self.avg_processing_time_min, 1),
            'std_processing_time_min': round(self.std_processing_time_min, 1),
            'min_processing_time_min': round(self.min_processing_time_min, 1),
            'max_processing_time_min': round(self.max_processing_time_min, 1),
            'avg_setup_time_min': round(self.avg_setup_time_min, 1),
            'setup_ratio_pct': round(self.setup_ratio * 100, 1),
            'avg_lead_time_min': round(self.avg_lead_time_min, 1),
            'lead_time_hours': round(self.lead_time_hours, 2),
            'scrap_rate_pct': round(self.scrap_rate * 100, 2),
            'yield_rate_pct': round(self.yield_rate * 100, 2),
            'total_units': round(self.total_units, 0),
            'total_orders': self.total_orders,
            'total_operations': self.total_operations,
            'snr_process': round(self.snr_process, 2),
            'snr_level': self.snr_level,
            'cp': round(self.cp, 2) if self.cp else None,
            'cpk': round(self.cpk, 2) if self.cpk else None,
        }


@dataclass
class ProductTypeKPIs:
    """
    Aggregated KPIs for a product type (e.g., vidro_duplo).
    """
    product_type: str
    
    # Counts
    num_products: int = 0
    num_orders: int = 0
    num_operations: int = 0
    total_units: float = 0.0
    
    # Processing time
    avg_processing_time_min: float = 0.0
    std_processing_time_min: float = 0.0
    median_processing_time_min: float = 0.0
    
    # Setup time
    avg_setup_time_min: float = 0.0
    total_setup_time_min: float = 0.0
    
    # Lead time
    avg_lead_time_hours: float = 0.0
    std_lead_time_hours: float = 0.0
    percentile_90_lead_time_hours: float = 0.0
    
    # Quality
    avg_scrap_rate: float = 0.0
    avg_yield: float = 1.0
    
    # Variation
    sigma_levels: Dict[str, float] = field(default_factory=dict)  # 1σ, 2σ, 3σ
    
    # SNR
    snr_between_products: float = 1.0
    snr_within_products: float = 1.0
    snr_level: str = "FAIR"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'product_type': self.product_type,
            'num_products': self.num_products,
            'num_orders': self.num_orders,
            'num_operations': self.num_operations,
            'total_units': round(self.total_units, 0),
            'avg_processing_time_min': round(self.avg_processing_time_min, 1),
            'std_processing_time_min': round(self.std_processing_time_min, 1),
            'median_processing_time_min': round(self.median_processing_time_min, 1),
            'avg_setup_time_min': round(self.avg_setup_time_min, 1),
            'avg_lead_time_hours': round(self.avg_lead_time_hours, 2),
            'std_lead_time_hours': round(self.std_lead_time_hours, 2),
            'percentile_90_lead_time_hours': round(self.percentile_90_lead_time_hours, 2),
            'avg_scrap_rate_pct': round(self.avg_scrap_rate * 100, 2),
            'avg_yield_pct': round(self.avg_yield * 100, 2),
            'sigma_levels': {k: round(v, 1) for k, v in self.sigma_levels.items()},
            'snr_between_products': round(self.snr_between_products, 2),
            'snr_within_products': round(self.snr_within_products, 2),
            'snr_level': self.snr_level,
        }


@dataclass
class GlobalProductKPIs:
    """
    Global KPIs across all product types.
    """
    timestamp: str
    
    # Counts
    total_product_types: int = 0
    total_products: int = 0
    total_orders: int = 0
    total_units: float = 0.0
    
    # Best/worst by type
    fastest_type: str = ""
    slowest_type: str = ""
    most_variable_type: str = ""
    most_stable_type: str = ""
    
    # Averages
    global_avg_lead_time_hours: float = 0.0
    global_avg_processing_time_min: float = 0.0
    
    # Distribution
    type_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'total_product_types': self.total_product_types,
            'total_products': self.total_products,
            'total_orders': self.total_orders,
            'total_units': round(self.total_units, 0),
            'fastest_type': self.fastest_type,
            'slowest_type': self.slowest_type,
            'most_variable_type': self.most_variable_type,
            'most_stable_type': self.most_stable_type,
            'global_avg_lead_time_hours': round(self.global_avg_lead_time_hours, 2),
            'global_avg_processing_time_min': round(self.global_avg_processing_time_min, 1),
            'type_distribution': self.type_distribution,
        }


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# SNR COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_snr(values: np.ndarray, groups: Optional[np.ndarray] = None) -> Tuple[float, str]:
    """
    Compute Signal-to-Noise Ratio.
    
    If groups provided: SNR = Var(group_means) / Var(residuals)
    Otherwise: simple SNR based on moving average
    
    Returns:
        (snr_value, snr_level)
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    
    if len(values) < 3:
        return 1.0, "FAIR"
    
    if groups is not None:
        # Between-group vs within-group variance
        groups = np.asarray(groups)
        unique_groups = np.unique(groups[~pd.isna(groups)])
        
        if len(unique_groups) < 2:
            return 1.0, "FAIR"
        
        group_means = []
        residuals = []
        
        for g in unique_groups:
            mask = groups == g
            group_vals = values[mask]
            if len(group_vals) > 0:
                gmean = np.mean(group_vals)
                group_means.append(gmean)
                residuals.extend(group_vals - gmean)
        
        var_between = np.var(group_means) if len(group_means) > 1 else 0
        var_within = np.var(residuals) if len(residuals) > 1 else 1
        
        snr = var_between / var_within if var_within > 1e-10 else 100
    else:
        # Simple SNR using moving average as signal
        window = min(5, len(values) // 2)
        if window < 2:
            return 1.0, "FAIR"
        
        kernel = np.ones(window) / window
        signal = np.convolve(values, kernel, mode='valid')
        aligned = values[window // 2:window // 2 + len(signal)]
        
        if len(aligned) != len(signal):
            aligned = values[:len(signal)]
        
        residual = aligned - signal
        var_signal = np.var(signal)
        var_residual = np.var(residual)
        
        snr = var_signal / var_residual if var_residual > 1e-10 else 100
    
    snr = min(max(snr, 0.1), 100)
    
    # Classify
    if snr >= 10:
        level = "EXCELLENT"
    elif snr >= 5:
        level = "GOOD"
    elif snr >= 2:
        level = "FAIR"
    else:
        level = "POOR"
    
    return snr, level


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# KPI COMPUTATION
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def compute_product_kpis(
    article_id: str,
    plan_df: pd.DataFrame,
    fingerprint: Optional[ProductFingerprint] = None,
    article_col: str = 'article_id',
    duration_col: str = 'duration_min',
    start_col: str = 'start_time',
    end_col: str = 'end_time',
    qty_col: str = 'qty',
    order_col: str = 'order_id',
) -> ProductKPIs:
    """
    Compute KPIs for a single product (article).
    """
    product_type = fingerprint.product_type.value if fingerprint else ProductType.OUTRO.value
    
    kpis = ProductKPIs(
        article_id=article_id,
        product_type=product_type,
    )
    
    # Filter plan for this article
    if article_col not in plan_df.columns:
        return kpis
    
    article_ops = plan_df[plan_df[article_col] == article_id]
    
    if article_ops.empty:
        return kpis
    
    # Processing time
    if duration_col in article_ops.columns:
        durations = article_ops[duration_col].dropna()
        if len(durations) > 0:
            kpis.avg_processing_time_min = float(durations.mean())
            kpis.std_processing_time_min = float(durations.std()) if len(durations) > 1 else 0
            kpis.min_processing_time_min = float(durations.min())
            kpis.max_processing_time_min = float(durations.max())
    
    # Lead time (from first op start to last op end)
    if start_col in article_ops.columns and end_col in article_ops.columns:
        starts = pd.to_datetime(article_ops[start_col], errors='coerce')
        ends = pd.to_datetime(article_ops[end_col], errors='coerce')
        
        # Group by order and compute lead time
        if order_col in article_ops.columns:
            lead_times = []
            for order_id, group in article_ops.groupby(order_col):
                order_starts = pd.to_datetime(group[start_col], errors='coerce')
                order_ends = pd.to_datetime(group[end_col], errors='coerce')
                
                valid_starts = order_starts.dropna()
                valid_ends = order_ends.dropna()
                
                if not valid_starts.empty and not valid_ends.empty:
                    lt = (valid_ends.max() - valid_starts.min()).total_seconds() / 60
                    lead_times.append(lt)
            
            if lead_times:
                kpis.avg_lead_time_min = np.mean(lead_times)
                kpis.std_lead_time_min = np.std(lead_times) if len(lead_times) > 1 else 0
                kpis.lead_time_hours = kpis.avg_lead_time_min / 60
    
    # Volume
    if qty_col in article_ops.columns:
        kpis.total_units = float(article_ops[qty_col].sum())
    
    if order_col in article_ops.columns:
        kpis.total_orders = article_ops[order_col].nunique()
    
    kpis.total_operations = len(article_ops)
    
    # SNR
    if duration_col in article_ops.columns:
        durations = article_ops[duration_col].dropna().values
        if len(durations) >= 3:
            snr, level = compute_snr(durations)
            kpis.snr_process = snr
            kpis.snr_level = level
    
    return kpis


def compute_product_type_kpis(
    product_type: ProductType,
    article_ids: List[str],
    plan_df: pd.DataFrame,
    product_kpis: Dict[str, ProductKPIs],
    article_col: str = 'article_id',
    duration_col: str = 'duration_min',
) -> ProductTypeKPIs:
    """
    Compute aggregated KPIs for a product type.
    """
    type_kpis = ProductTypeKPIs(
        product_type=product_type.value,
        num_products=len(article_ids),
    )
    
    if not article_ids:
        return type_kpis
    
    # Collect all operations for this type
    type_ops = plan_df[plan_df[article_col].isin(article_ids)]
    
    if type_ops.empty:
        return type_kpis
    
    # Aggregate from product KPIs
    processing_times = []
    lead_times = []
    scrap_rates = []
    
    for aid in article_ids:
        kpi = product_kpis.get(aid)
        if kpi:
            if kpi.avg_processing_time_min > 0:
                processing_times.append(kpi.avg_processing_time_min)
            if kpi.lead_time_hours > 0:
                lead_times.append(kpi.lead_time_hours)
            if kpi.scrap_rate >= 0:
                scrap_rates.append(kpi.scrap_rate)
            type_kpis.num_orders += kpi.total_orders
            type_kpis.num_operations += kpi.total_operations
            type_kpis.total_units += kpi.total_units
    
    # Processing time stats
    if processing_times:
        type_kpis.avg_processing_time_min = np.mean(processing_times)
        type_kpis.std_processing_time_min = np.std(processing_times) if len(processing_times) > 1 else 0
        type_kpis.median_processing_time_min = np.median(processing_times)
        
        # Sigma levels
        mu = type_kpis.avg_processing_time_min
        sigma = type_kpis.std_processing_time_min
        if sigma > 0:
            type_kpis.sigma_levels = {
                '1σ': mu + sigma,
                '2σ': mu + 2 * sigma,
                '3σ': mu + 3 * sigma,
                '-1σ': mu - sigma,
                '-2σ': mu - 2 * sigma,
                '-3σ': max(0, mu - 3 * sigma),
            }
    
    # Lead time stats
    if lead_times:
        type_kpis.avg_lead_time_hours = np.mean(lead_times)
        type_kpis.std_lead_time_hours = np.std(lead_times) if len(lead_times) > 1 else 0
        type_kpis.percentile_90_lead_time_hours = np.percentile(lead_times, 90)
    
    # Quality
    if scrap_rates:
        type_kpis.avg_scrap_rate = np.mean(scrap_rates)
        type_kpis.avg_yield = 1 - type_kpis.avg_scrap_rate
    
    # SNR - between products vs within products
    if duration_col in type_ops.columns and article_col in type_ops.columns:
        durations = type_ops[duration_col].dropna().values
        groups = type_ops.loc[type_ops[duration_col].notna(), article_col].values
        
        if len(durations) >= 5:
            snr, level = compute_snr(durations, groups)
            type_kpis.snr_between_products = snr
            type_kpis.snr_level = level
    
    return type_kpis


def compute_all_product_kpis(
    routing_df: pd.DataFrame,
    plan_df: pd.DataFrame,
    orders_df: Optional[pd.DataFrame] = None,
    article_col: str = 'article_id',
    **kwargs
) -> Tuple[Dict[str, ProductKPIs], Dict[str, ProductTypeKPIs], GlobalProductKPIs]:
    """
    Compute all product KPIs.
    
    Returns:
        (product_kpis, type_kpis, global_kpis)
    """
    # Classify all products
    fingerprints = classify_all_products(routing_df, plan_df, article_col=article_col)
    
    # Group by type
    type_groups = group_products_by_type(fingerprints)
    
    # Compute per-product KPIs
    product_kpis = {}
    for article_id, fp in fingerprints.items():
        kpi = compute_product_kpis(
            article_id=article_id,
            plan_df=plan_df,
            fingerprint=fp,
            article_col=article_col,
            **kwargs
        )
        product_kpis[article_id] = kpi
    
    # Compute per-type KPIs
    type_kpis = {}
    for ptype, article_ids in type_groups.items():
        if article_ids:
            tkpi = compute_product_type_kpis(
                product_type=ptype,
                article_ids=article_ids,
                plan_df=plan_df,
                product_kpis=product_kpis,
                article_col=article_col,
            )
            type_kpis[ptype.value] = tkpi
    
    # Global KPIs
    global_kpis = GlobalProductKPIs(
        timestamp=datetime.now().isoformat(),
        total_product_types=len([t for t, ids in type_groups.items() if ids]),
        total_products=len(fingerprints),
        total_orders=sum(k.total_orders for k in product_kpis.values()),
        total_units=sum(k.total_units for k in product_kpis.values()),
    )
    
    # Best/worst types
    valid_types = {k: v for k, v in type_kpis.items() if v.num_products > 0 and v.avg_processing_time_min > 0}
    
    if valid_types:
        global_kpis.fastest_type = min(valid_types, key=lambda k: valid_types[k].avg_processing_time_min)
        global_kpis.slowest_type = max(valid_types, key=lambda k: valid_types[k].avg_processing_time_min)
        global_kpis.most_stable_type = max(valid_types, key=lambda k: valid_types[k].snr_between_products)
        global_kpis.most_variable_type = min(valid_types, key=lambda k: valid_types[k].snr_between_products)
    
    # Averages
    if product_kpis:
        valid_lead = [k.lead_time_hours for k in product_kpis.values() if k.lead_time_hours > 0]
        valid_proc = [k.avg_processing_time_min for k in product_kpis.values() if k.avg_processing_time_min > 0]
        
        global_kpis.global_avg_lead_time_hours = np.mean(valid_lead) if valid_lead else 0
        global_kpis.global_avg_processing_time_min = np.mean(valid_proc) if valid_proc else 0
    
    # Distribution
    global_kpis.type_distribution = {
        ptype.value: len(ids)
        for ptype, ids in type_groups.items()
        if ids
    }
    
    return product_kpis, type_kpis, global_kpis



