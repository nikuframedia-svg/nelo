"""
Intelligent Suggestions Engine for Nikufra Production OS

Analyzes production plan data and generates actionable recommendations:
- Machine overload reduction suggestions
- Idle gap identification
- Product risk / bottleneck detection
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd


@dataclass
class OverloadSuggestion:
    """Suggestion to reduce machine overload by moving an operation."""
    type: str = "overload_reduction"
    machine: str = ""
    candidate_op: str = ""
    order_id: str = ""
    article_id: str = ""
    duration_min: float = 0
    alternative_machine: Optional[str] = None
    expected_gain_h: float = 0
    reason: str = ""


@dataclass
class IdleGapSuggestion:
    """Identifies significant idle gaps in machine schedules."""
    type: str = "idle_gap"
    machine: str = ""
    gap_start: str = ""
    gap_end: str = ""
    gap_min: float = 0
    reason: str = ""


@dataclass
class ProductRiskSuggestion:
    """Identifies articles with long waiting times between operations."""
    type: str = "product_risk"
    article: str = ""
    bottleneck_op: str = ""
    wait_min: float = 0
    from_op: str = ""
    to_op: str = ""
    reason: str = ""


def compute_machine_loads(plan_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Compute load statistics per machine.
    
    Returns dict: machine_id -> {total_min, num_ops, idle_min, utilization_pct}
    """
    if plan_df.empty:
        return {}
    
    # Ensure datetime columns
    if not pd.api.types.is_datetime64_any_dtype(plan_df["start_time"]):
        plan_df = plan_df.copy()
        plan_df["start_time"] = pd.to_datetime(plan_df["start_time"])
        plan_df["end_time"] = pd.to_datetime(plan_df["end_time"])
    
    machine_stats = {}
    
    for machine_id, ops in plan_df.groupby("machine_id"):
        ops_sorted = ops.sort_values("start_time")
        
        total_duration = ops_sorted["duration_min"].sum()
        num_ops = len(ops_sorted)
        
        # Calculate idle time (gaps between operations)
        idle_min = 0
        prev_end = None
        for _, op in ops_sorted.iterrows():
            if prev_end is not None:
                gap = (op["start_time"] - prev_end).total_seconds() / 60
                if gap > 0:
                    idle_min += gap
            prev_end = op["end_time"]
        
        # Calculate span and utilization
        if num_ops > 0:
            span_start = ops_sorted["start_time"].min()
            span_end = ops_sorted["end_time"].max()
            span_min = (span_end - span_start).total_seconds() / 60
            utilization_pct = (total_duration / span_min * 100) if span_min > 0 else 0
        else:
            span_min = 0
            utilization_pct = 0
        
        machine_stats[str(machine_id)] = {
            "total_min": total_duration,
            "num_ops": num_ops,
            "idle_min": idle_min,
            "span_min": span_min,
            "utilization_pct": utilization_pct,
        }
    
    return machine_stats


def find_idle_gaps(plan_df: pd.DataFrame, min_gap_minutes: float = 60) -> List[IdleGapSuggestion]:
    """
    Identify idle gaps larger than threshold in machine schedules.
    """
    suggestions = []
    
    if plan_df.empty:
        return suggestions
    
    # Ensure datetime columns
    if not pd.api.types.is_datetime64_any_dtype(plan_df["start_time"]):
        plan_df = plan_df.copy()
        plan_df["start_time"] = pd.to_datetime(plan_df["start_time"])
        plan_df["end_time"] = pd.to_datetime(plan_df["end_time"])
    
    for machine_id, ops in plan_df.groupby("machine_id"):
        ops_sorted = ops.sort_values("start_time")
        
        prev_end = None
        for _, op in ops_sorted.iterrows():
            if prev_end is not None:
                gap_minutes = (op["start_time"] - prev_end).total_seconds() / 60
                
                if gap_minutes >= min_gap_minutes:
                    suggestions.append(IdleGapSuggestion(
                        machine=str(machine_id),
                        gap_start=prev_end.strftime("%Y-%m-%d %H:%M"),
                        gap_end=op["start_time"].strftime("%Y-%m-%d %H:%M"),
                        gap_min=round(gap_minutes, 1),
                        reason=f"Gap de {gap_minutes/60:.1f}h ocioso identificado",
                    ))
            
            prev_end = op["end_time"]
    
    # Sort by gap size descending
    suggestions.sort(key=lambda x: x.gap_min, reverse=True)
    return suggestions[:10]  # Limit to top 10


def find_overload_candidates(
    plan_df: pd.DataFrame,
    machine_stats: Dict[str, Dict[str, Any]],
    routing_df: Optional[pd.DataFrame] = None
) -> List[OverloadSuggestion]:
    """
    Find operations that could be moved from overloaded machines to less loaded ones.
    """
    suggestions = []
    
    if plan_df.empty or not machine_stats:
        return suggestions
    
    # Find the most overloaded machine (highest total_min)
    sorted_machines = sorted(
        machine_stats.items(),
        key=lambda x: x[1]["total_min"],
        reverse=True
    )
    
    if not sorted_machines:
        return suggestions
    
    # Top overloaded machine
    overloaded_machine = sorted_machines[0][0]
    overloaded_stats = sorted_machines[0][1]
    
    # Find the largest operation on this machine
    machine_ops = plan_df[plan_df["machine_id"] == overloaded_machine].copy()
    if machine_ops.empty:
        return suggestions
    
    # Sort by duration descending
    machine_ops = machine_ops.sort_values("duration_min", ascending=False)
    largest_op = machine_ops.iloc[0]
    
    # Find potential alternative machines (those with lower utilization)
    alternative_machine = None
    expected_gain_h = 0
    
    for machine_id, stats in sorted_machines[1:]:
        # Check if this machine has significant idle time
        if stats["idle_min"] > largest_op["duration_min"]:
            alternative_machine = machine_id
            expected_gain_h = largest_op["duration_min"] / 60
            break
        # Or if utilization is much lower
        elif stats["utilization_pct"] < 60 and overloaded_stats["utilization_pct"] > 80:
            alternative_machine = machine_id
            expected_gain_h = (overloaded_stats["total_min"] - stats["total_min"]) / 60 * 0.3
            break
    
    if alternative_machine:
        suggestions.append(OverloadSuggestion(
            machine=overloaded_machine,
            candidate_op=str(largest_op.get("op_code", "OP-????")),
            order_id=str(largest_op.get("order_id", "")),
            article_id=str(largest_op.get("article_id", "")),
            duration_min=float(largest_op["duration_min"]),
            alternative_machine=alternative_machine,
            expected_gain_h=round(expected_gain_h, 1),
            reason=f"Mover de {overloaded_machine} ({overloaded_stats['utilization_pct']:.0f}% util.) para {alternative_machine}",
        ))
    else:
        # Still report the overload even without alternative
        suggestions.append(OverloadSuggestion(
            machine=overloaded_machine,
            candidate_op=str(largest_op.get("op_code", "OP-????")),
            order_id=str(largest_op.get("order_id", "")),
            article_id=str(largest_op.get("article_id", "")),
            duration_min=float(largest_op["duration_min"]),
            alternative_machine=None,
            expected_gain_h=0,
            reason=f"Máquina sobrecarregada ({overloaded_stats['utilization_pct']:.0f}% util.) - verificar capacidade adicional",
        ))
    
    return suggestions


def find_product_risks(plan_df: pd.DataFrame, min_wait_minutes: float = 60) -> List[ProductRiskSuggestion]:
    """
    Identify articles with long waiting times between sequential operations.
    """
    suggestions = []
    
    if plan_df.empty:
        return suggestions
    
    # Ensure datetime columns
    if not pd.api.types.is_datetime64_any_dtype(plan_df["start_time"]):
        plan_df = plan_df.copy()
        plan_df["start_time"] = pd.to_datetime(plan_df["start_time"])
        plan_df["end_time"] = pd.to_datetime(plan_df["end_time"])
    
    # Analyze each article's operations
    for article_id, article_ops in plan_df.groupby("article_id"):
        # Sort by operation sequence or start time
        if "op_seq" in article_ops.columns:
            ops_sorted = article_ops.sort_values(["op_seq", "start_time"])
        else:
            ops_sorted = article_ops.sort_values("start_time")
        
        if len(ops_sorted) < 2:
            continue
        
        # Find gaps between consecutive operations
        prev_op = None
        max_wait = 0
        max_wait_ops = (None, None)
        
        for _, op in ops_sorted.iterrows():
            if prev_op is not None:
                wait_min = (op["start_time"] - prev_op["end_time"]).total_seconds() / 60
                
                if wait_min > max_wait and wait_min >= min_wait_minutes:
                    max_wait = wait_min
                    max_wait_ops = (prev_op, op)
            
            prev_op = op
        
        if max_wait >= min_wait_minutes and max_wait_ops[0] is not None:
            from_op, to_op = max_wait_ops
            suggestions.append(ProductRiskSuggestion(
                article=str(article_id),
                bottleneck_op=str(to_op.get("op_code", "OP-????")),
                wait_min=round(max_wait, 1),
                from_op=str(from_op.get("op_code", "OP-????")),
                to_op=str(to_op.get("op_code", "OP-????")),
                reason=f"Espera de {max_wait/60:.1f}h entre {from_op.get('op_code', '?')} e {to_op.get('op_code', '?')}",
            ))
    
    # Sort by wait time descending
    suggestions.sort(key=lambda x: x.wait_min, reverse=True)
    return suggestions[:10]  # Limit to top 10


def compute_suggestions(
    plan_df: pd.DataFrame,
    orders_df: pd.DataFrame = None,
    machines_df: pd.DataFrame = None,
    routing_df: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Main entry point: compute all intelligent suggestions for the production plan.
    
    Returns:
        Dict with:
        - overload_suggestions: List of overload reduction suggestions
        - idle_gaps: List of idle gap suggestions  
        - product_risks: List of product risk suggestions
        - summary: High-level summary
    """
    if plan_df.empty:
        return {
            "overload_suggestions": [],
            "idle_gaps": [],
            "product_risks": [],
            "summary": {
                "total_suggestions": 0,
                "high_priority": 0,
            }
        }
    
    # Compute machine load statistics
    machine_stats = compute_machine_loads(plan_df)
    
    # Find different types of suggestions
    overload_suggestions = find_overload_candidates(plan_df, machine_stats, routing_df)
    idle_gaps = find_idle_gaps(plan_df, min_gap_minutes=60)
    product_risks = find_product_risks(plan_df, min_wait_minutes=60)
    
    # Count high priority items
    high_priority = 0
    for s in overload_suggestions:
        if s.expected_gain_h > 2:
            high_priority += 1
    for s in idle_gaps:
        if s.gap_min > 120:
            high_priority += 1
    for s in product_risks:
        if s.wait_min > 120:
            high_priority += 1
    
    total_suggestions = len(overload_suggestions) + len(idle_gaps) + len(product_risks)
    
    return {
        "overload_suggestions": [asdict(s) for s in overload_suggestions],
        "idle_gaps": [asdict(s) for s in idle_gaps],
        "product_risks": [asdict(s) for s in product_risks],
        "machine_loads": machine_stats,
        "summary": {
            "total_suggestions": total_suggestions,
            "high_priority": high_priority,
            "overload_count": len(overload_suggestions),
            "idle_gap_count": len(idle_gaps),
            "product_risk_count": len(product_risks),
        }
    }


def format_suggestion_pt(suggestion: Dict[str, Any]) -> str:
    """
    Format a suggestion as a human-readable PT-PT string.
    """
    stype = suggestion.get("type", "")
    
    if stype == "overload_reduction":
        if suggestion.get("alternative_machine"):
            return (
                f"💡 Mover {suggestion['candidate_op']} de {suggestion['machine']} "
                f"para {suggestion['alternative_machine']} liberta {suggestion['expected_gain_h']}h."
            )
        else:
            return (
                f"⚠️ Máquina {suggestion['machine']} está sobrecarregada. "
                f"Operação maior: {suggestion['candidate_op']} ({suggestion['duration_min']:.0f} min)."
            )
    
    elif stype == "idle_gap":
        hours = suggestion['gap_min'] / 60
        return (
            f"⏳ Máquina {suggestion['machine']} tem um gap de {hours:.1f}h ocioso "
            f"entre {suggestion['gap_start']} e {suggestion['gap_end']}."
        )
    
    elif stype == "product_risk":
        hours = suggestion['wait_min'] / 60
        return (
            f"🚨 {suggestion['article']} tem espera de {hours:.1f}h entre operações "
            f"({suggestion['from_op']} → {suggestion['to_op']})."
        )
    
    return str(suggestion)



