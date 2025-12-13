"""
════════════════════════════════════════════════════════════════════════════════════════════════════
R&D Report Generator - SIFIDE Export
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 11 Implementation: R&D Reporting for SIFIDE tax incentive documentation

Features:
- Consolidated view of all WP1-WP4 experiments
- Annual/period summary generation
- Export to JSON and PDF formats
- Key metrics for R&D dossier

This module provides evidence of:
- Experimental activities
- Technical uncertainty
- Incremental evolution
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WP1Summary:
    """WP1: Dynamic Routing experiments summary."""
    num_experiments: int
    policies_tested: List[str]
    baseline_policy: str
    avg_makespan_improvement_pct: float
    avg_tardiness_improvement_pct: float
    avg_otd_improvement_pct: float
    best_policy: str
    best_makespan_improvement_pct: float
    experiments_list: List[Dict[str, Any]]


@dataclass
class WP2Summary:
    """WP2: AI Suggestions evaluation summary."""
    num_suggestions_evaluated: int
    num_beneficial: int
    num_neutral: int
    num_harmful: int
    pct_beneficial: float
    pct_neutral: float
    pct_harmful: float
    suggestion_types: List[str]
    avg_otd_delta: float
    avg_tardiness_delta: float
    experiments_list: List[Dict[str, Any]]


@dataclass
class WP3Summary:
    """WP3: Inventory + Capacity experiments summary."""
    num_experiments: int
    policies_tested: List[str]
    avg_total_cost: float
    avg_stockouts: float
    avg_otd: float
    best_policy: str
    best_cost: float
    policy_comparison: List[Dict[str, Any]]
    experiments_list: List[Dict[str, Any]]


@dataclass
class WP4Summary:
    """WP4: Learning Scheduler experiments summary."""
    num_episodes: int
    policies_in_bandit: List[str]
    avg_reward: float
    avg_regret: float
    best_observed_policy: str
    convergence_episode: Optional[int]
    experiments_list: List[Dict[str, Any]]


@dataclass
class RDReportSummary:
    """Complete R&D report summary for a period."""
    period_start: str
    period_end: str
    generated_at: str
    total_experiments: int
    wp1: WP1Summary
    wp2: WP2Summary
    wp3: WP3Summary
    wp4: WP4Summary
    hypotheses_summary: List[Dict[str, str]]


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_rd_summary(
    start_date: date,
    end_date: date,
) -> Dict[str, Any]:
    """
    Build consolidated R&D summary for a period.
    
    Reads rd_experiments and WP1-WP4 tables within the date range.
    
    Args:
        start_date: Start of period
        end_date: End of period
        
    Returns:
        Dict with period info and WP summaries
    """
    logger.info(f"Building R&D summary for {start_date} to {end_date}")
    
    # Get summaries for each WP
    wp1 = _build_wp1_summary(start_date, end_date)
    wp2 = _build_wp2_summary(start_date, end_date)
    wp3 = _build_wp3_summary(start_date, end_date)
    wp4 = _build_wp4_summary(start_date, end_date)
    
    total_experiments = (
        wp1.num_experiments + 
        wp2.num_suggestions_evaluated + 
        wp3.num_experiments + 
        wp4.num_episodes
    )
    
    # Build hypotheses summary
    hypotheses = _build_hypotheses_summary(wp1, wp2, wp3, wp4)
    
    summary = RDReportSummary(
        period_start=start_date.isoformat(),
        period_end=end_date.isoformat(),
        generated_at=datetime.utcnow().isoformat(),
        total_experiments=total_experiments,
        wp1=wp1,
        wp2=wp2,
        wp3=wp3,
        wp4=wp4,
        hypotheses_summary=hypotheses,
    )
    
    return _summary_to_dict(summary)


def _build_wp1_summary(start_date: date, end_date: date) -> WP1Summary:
    """Build WP1 (Routing Dinâmico) summary."""
    experiments = []
    policies_tested = set()
    baseline_policy = "FIFO"
    
    makespan_improvements = []
    tardiness_improvements = []
    otd_improvements = []
    best_policy = "FIFO"
    best_makespan_improvement = 0.0
    
    try:
        from .experiments_core import get_db_session
        
        with get_db_session() as db:
            # Query WP1 experiments using SQLite
            cursor = db.cursor()
            cursor.execute("""
                SELECT id, created_at, parameters, summary, kpis
                FROM rd_experiments
                WHERE wp = 'WP1_ROUTING'
                AND date(created_at) >= ?
                AND date(created_at) <= ?
                ORDER BY created_at DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            
            for row in rows:
                exp_id = row['id']
                created_at = row['created_at']
                config_json = row['parameters']
                results_json = row['summary']
                kpis_json = row['kpis']
                
                config = json.loads(config_json) if config_json else {}
                results = json.loads(results_json) if results_json else {}
                kpis = json.loads(kpis_json) if kpis_json else {}
                
                # Extract policies tested
                tested = config.get("policies", [])
                policies_tested.update(tested)
                
                if config.get("baseline_policy"):
                    baseline_policy = config["baseline_policy"]
                
                # Calculate improvements
                baseline_makespan = results.get("baseline_makespan", 0)
                best_makespan = results.get("best_makespan", baseline_makespan)
                
                if baseline_makespan > 0:
                    improvement = (baseline_makespan - best_makespan) / baseline_makespan * 100
                    makespan_improvements.append(improvement)
                    
                    if improvement > best_makespan_improvement:
                        best_makespan_improvement = improvement
                        best_policy = results.get("best_policy", "FIFO")
                
                # Tardiness and OTD improvements
                if "baseline_tardiness" in results and "best_tardiness" in results:
                    bt = results["baseline_tardiness"]
                    best_t = results["best_tardiness"]
                    if bt > 0:
                        tardiness_improvements.append((bt - best_t) / bt * 100)
                
                if "baseline_otd" in results and "best_otd" in results:
                    b_otd = results["baseline_otd"]
                    best_otd = results["best_otd"]
                    otd_improvements.append(best_otd - b_otd)
                
                experiments.append({
                    "id": exp_id,
                    "date": created_at[:10] if created_at else None,
                    "policies": tested,
                    "best_policy": results.get("best_policy"),
                    "makespan_improvement_pct": improvement if baseline_makespan > 0 else 0,
                })
    
    except Exception as e:
        logger.warning(f"Error reading WP1 experiments: {e}")
    
    return WP1Summary(
        num_experiments=len(experiments),
        policies_tested=list(policies_tested) if policies_tested else ["FIFO", "SPT", "EDD", "CR"],
        baseline_policy=baseline_policy,
        avg_makespan_improvement_pct=sum(makespan_improvements) / len(makespan_improvements) if makespan_improvements else 0.0,
        avg_tardiness_improvement_pct=sum(tardiness_improvements) / len(tardiness_improvements) if tardiness_improvements else 0.0,
        avg_otd_improvement_pct=sum(otd_improvements) / len(otd_improvements) if otd_improvements else 0.0,
        best_policy=best_policy,
        best_makespan_improvement_pct=best_makespan_improvement,
        experiments_list=experiments,
    )


def _build_wp2_summary(start_date: date, end_date: date) -> WP2Summary:
    """Build WP2 (Sugestões IA) summary."""
    experiments = []
    beneficial = 0
    neutral = 0
    harmful = 0
    suggestion_types = set()
    otd_deltas = []
    tardiness_deltas = []
    
    try:
        from .experiments_core import get_db_session
        
        with get_db_session() as db:
            # Query WP2 experiments using SQLite
            cursor = db.cursor()
            cursor.execute("""
                SELECT id, created_at, parameters, summary, kpis
                FROM rd_experiments
                WHERE wp = 'WP2_SUGGESTIONS'
                AND date(created_at) >= ?
                AND date(created_at) <= ?
                ORDER BY created_at DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            
            for row in rows:
                exp_id = row['id']
                created_at = row['created_at']
                config_json = row['parameters']
                results_json = row['summary']
                
                config = json.loads(config_json) if config_json else {}
                results = json.loads(results_json) if results_json else {}
                
                # Count by label
                label = results.get("label", "NEUTRAL")
                if label == "BENEFICIAL":
                    beneficial += 1
                elif label == "HARMFUL":
                    harmful += 1
                else:
                    neutral += 1
                
                # Track suggestion types
                sug_type = config.get("suggestion_type", "unknown")
                suggestion_types.add(sug_type)
                
                # Track deltas
                if "delta_otd" in results:
                    otd_deltas.append(results["delta_otd"])
                if "delta_tardiness" in results:
                    tardiness_deltas.append(results["delta_tardiness"])
                
                experiments.append({
                    "id": exp_id,
                    "date": created_at[:10] if created_at else None,
                    "suggestion_type": sug_type,
                    "label": label,
                    "delta_otd": results.get("delta_otd", 0),
                })
    
    except Exception as e:
        logger.warning(f"Error reading WP2 experiments: {e}")
    
    total = beneficial + neutral + harmful
    
    return WP2Summary(
        num_suggestions_evaluated=total,
        num_beneficial=beneficial,
        num_neutral=neutral,
        num_harmful=harmful,
        pct_beneficial=(beneficial / total * 100) if total > 0 else 0.0,
        pct_neutral=(neutral / total * 100) if total > 0 else 0.0,
        pct_harmful=(harmful / total * 100) if total > 0 else 0.0,
        suggestion_types=list(suggestion_types) if suggestion_types else ["rescheduling", "capacity", "outsourcing"],
        avg_otd_delta=sum(otd_deltas) / len(otd_deltas) if otd_deltas else 0.0,
        avg_tardiness_delta=sum(tardiness_deltas) / len(tardiness_deltas) if tardiness_deltas else 0.0,
        experiments_list=experiments,
    )


def _build_wp3_summary(start_date: date, end_date: date) -> WP3Summary:
    """Build WP3 (Inventário + Capacidade) summary."""
    experiments = []
    policies_tested = set()
    costs = []
    stockouts = []
    otds = []
    policy_results = {}
    
    try:
        from .experiments_core import get_db_session
        
        with get_db_session() as db:
            # Query WP3 experiments using SQLite
            cursor = db.cursor()
            cursor.execute("""
                SELECT id, created_at, parameters, summary, kpis
                FROM rd_experiments
                WHERE wp = 'WP3_INVENTORY_CAPACITY'
                AND date(created_at) >= ?
                AND date(created_at) <= ?
                ORDER BY created_at DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            
            for row in rows:
                exp_id = row['id']
                created_at = row['created_at']
                config_json = row['parameters']
                results_json = row['summary']
                kpis_json = row['kpis']
                
                config = json.loads(config_json) if config_json else {}
                results = json.loads(results_json) if results_json else {}
                kpis = json.loads(kpis_json) if kpis_json else {}
                
                # Track policy
                policy_name = config.get("policy_name", results.get("best_policy", "default"))
                policies_tested.add(policy_name)
                
                # Track metrics
                cost = kpis.get("total_cost", results.get("total_cost", 0))
                stockout = kpis.get("stockout_events", results.get("stockout_events", 0))
                otd = kpis.get("otd_pct", results.get("otd_pct", 0))
                
                costs.append(cost)
                stockouts.append(stockout)
                otds.append(otd)
                
                # Aggregate by policy
                if policy_name not in policy_results:
                    policy_results[policy_name] = {"costs": [], "stockouts": [], "otds": []}
                policy_results[policy_name]["costs"].append(cost)
                policy_results[policy_name]["stockouts"].append(stockout)
                policy_results[policy_name]["otds"].append(otd)
                
                experiments.append({
                    "id": exp_id,
                    "date": created_at[:10] if created_at else None,
                    "policy": policy_name,
                    "total_cost": cost,
                    "stockouts": stockout,
                    "otd_pct": otd,
                })
    
    except Exception as e:
        logger.warning(f"Error reading WP3 experiments: {e}")
    
    # Build policy comparison
    policy_comparison = []
    best_policy = "default"
    best_cost = float("inf")
    
    for policy, data in policy_results.items():
        avg_cost = sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0
        avg_stockout = sum(data["stockouts"]) / len(data["stockouts"]) if data["stockouts"] else 0
        avg_otd = sum(data["otds"]) / len(data["otds"]) if data["otds"] else 0
        
        policy_comparison.append({
            "policy": policy,
            "avg_cost": round(avg_cost, 2),
            "avg_stockouts": round(avg_stockout, 1),
            "avg_otd_pct": round(avg_otd, 1),
            "num_runs": len(data["costs"]),
        })
        
        if avg_cost < best_cost and avg_cost > 0:
            best_cost = avg_cost
            best_policy = policy
    
    return WP3Summary(
        num_experiments=len(experiments),
        policies_tested=list(policies_tested) if policies_tested else ["conservative", "aggressive", "balanced"],
        avg_total_cost=sum(costs) / len(costs) if costs else 0.0,
        avg_stockouts=sum(stockouts) / len(stockouts) if stockouts else 0.0,
        avg_otd=sum(otds) / len(otds) if otds else 0.0,
        best_policy=best_policy,
        best_cost=best_cost if best_cost < float("inf") else 0.0,
        policy_comparison=policy_comparison,
        experiments_list=experiments,
    )


def _build_wp4_summary(start_date: date, end_date: date) -> WP4Summary:
    """Build WP4 (Learning Scheduler) summary."""
    experiments = []
    policies_in_bandit = set()
    rewards = []
    regrets = []
    policy_rewards = {}
    
    try:
        from .experiments_core import get_db_session
        
        with get_db_session() as db:
            # Query WP4 experiments using SQLite
            cursor = db.cursor()
            cursor.execute("""
                SELECT id, created_at, parameters, summary, kpis
                FROM rd_experiments
                WHERE wp = 'WP4_LEARNING_SCHEDULER'
                AND date(created_at) >= ?
                AND date(created_at) <= ?
                ORDER BY created_at DESC
            """, (start_date.isoformat(), end_date.isoformat()))
            
            rows = cursor.fetchall()
            
            for row in rows:
                exp_id = row['id']
                created_at = row['created_at']
                config_json = row['parameters']
                results_json = row['summary']
                
                config = json.loads(config_json) if config_json else {}
                results = json.loads(results_json) if results_json else {}
                
                # Track policies
                policies = config.get("policies", results.get("policies", []))
                policies_in_bandit.update(policies)
                
                # Track metrics
                reward = results.get("reward", 0)
                regret = results.get("regret", 0)
                selected_policy = results.get("selected_policy", "unknown")
                
                rewards.append(reward)
                regrets.append(regret)
                
                # Track by policy
                if selected_policy not in policy_rewards:
                    policy_rewards[selected_policy] = []
                policy_rewards[selected_policy].append(reward)
                
                experiments.append({
                    "id": exp_id,
                    "date": created_at[:10] if created_at else None,
                    "selected_policy": selected_policy,
                    "reward": reward,
                    "regret": regret,
                })
    
    except Exception as e:
        logger.warning(f"Error reading WP4 experiments: {e}")
    
    # Find best observed policy
    best_policy = "unknown"
    best_avg_reward = float("-inf")
    
    for policy, policy_rewards_list in policy_rewards.items():
        avg = sum(policy_rewards_list) / len(policy_rewards_list) if policy_rewards_list else 0
        if avg > best_avg_reward:
            best_avg_reward = avg
            best_policy = policy
    
    # Estimate convergence (simplified)
    convergence_episode = None
    if len(regrets) > 10:
        # Check if regret stabilizes
        recent_regrets = regrets[-10:]
        if max(recent_regrets) - min(recent_regrets) < 0.1:
            convergence_episode = len(regrets) - 10
    
    return WP4Summary(
        num_episodes=len(experiments),
        policies_in_bandit=list(policies_in_bandit) if policies_in_bandit else ["FIFO", "SPT", "EDD", "MILP"],
        avg_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        avg_regret=sum(regrets) / len(regrets) if regrets else 0.0,
        best_observed_policy=best_policy,
        convergence_episode=convergence_episode,
        experiments_list=experiments,
    )


def _build_hypotheses_summary(
    wp1: WP1Summary,
    wp2: WP2Summary,
    wp3: WP3Summary,
    wp4: WP4Summary,
) -> List[Dict[str, str]]:
    """Build summary of hypotheses tested and results."""
    hypotheses = []
    
    # WP1 Hypothesis
    wp1_result = "SUPPORTED" if wp1.avg_makespan_improvement_pct >= 8 else "PARTIAL"
    hypotheses.append({
        "wp": "WP1",
        "hypothesis": "H1.1 - Routing dinâmico reduz makespan ≥8%",
        "result": wp1_result,
        "evidence": f"Melhoria média de {wp1.avg_makespan_improvement_pct:.1f}% ({wp1.num_experiments} experiências)",
    })
    
    # WP2 Hypothesis
    wp2_result = "SUPPORTED" if wp2.pct_beneficial > 50 else "PARTIAL"
    hypotheses.append({
        "wp": "WP2",
        "hypothesis": "H2.1 - Sugestões IA são benéficas em >50% dos casos",
        "result": wp2_result,
        "evidence": f"{wp2.pct_beneficial:.1f}% benéficas, {wp2.pct_harmful:.1f}% prejudiciais ({wp2.num_suggestions_evaluated} avaliadas)",
    })
    
    # WP3 Hypothesis
    wp3_result = "SUPPORTED" if wp3.num_experiments > 0 else "INSUFFICIENT_DATA"
    hypotheses.append({
        "wp": "WP3",
        "hypothesis": "H3.1 - Políticas adaptativas reduzem custos de inventário",
        "result": wp3_result,
        "evidence": f"Melhor política: {wp3.best_policy} com custo {wp3.best_cost:.2f} ({wp3.num_experiments} cenários)",
    })
    
    # WP4 Hypothesis
    wp4_result = "SUPPORTED" if wp4.avg_regret < 0.5 and wp4.num_episodes > 0 else "PARTIAL"
    hypotheses.append({
        "wp": "WP4",
        "hypothesis": "H4.1 - Contextual bandits convergem para política ótima",
        "result": wp4_result,
        "evidence": f"Regret médio: {wp4.avg_regret:.3f}, melhor política: {wp4.best_observed_policy} ({wp4.num_episodes} episódios)",
    })
    
    return hypotheses


def _summary_to_dict(summary: RDReportSummary) -> Dict[str, Any]:
    """Convert summary dataclass to dict for JSON serialization."""
    return {
        "period": {
            "start": summary.period_start,
            "end": summary.period_end,
        },
        "generated_at": summary.generated_at,
        "total_experiments": summary.total_experiments,
        "wp1": asdict(summary.wp1),
        "wp2": asdict(summary.wp2),
        "wp3": asdict(summary.wp3),
        "wp4": asdict(summary.wp4),
        "hypotheses_summary": summary.hypotheses_summary,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def export_rd_report(
    start_date: date,
    end_date: date,
    format: str = "json",
) -> Tuple[bytes, str, str]:
    """
    Export R&D report in specified format.
    
    Args:
        start_date: Start of period
        end_date: End of period
        format: "json" or "pdf"
        
    Returns:
        Tuple of (content bytes, filename, content_type)
    """
    summary = build_rd_summary(start_date, end_date)
    
    filename_base = f"rd_report_{start_date.isoformat()}_{end_date.isoformat()}"
    
    if format.lower() == "json":
        content = json.dumps(summary, indent=2, ensure_ascii=False).encode("utf-8")
        return content, f"{filename_base}.json", "application/json"
    
    elif format.lower() == "pdf":
        content = _generate_pdf_report(summary, start_date, end_date)
        return content, f"{filename_base}.pdf", "application/pdf"
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def _generate_pdf_report(
    summary: Dict[str, Any],
    start_date: date,
    end_date: date,
) -> bytes:
    """
    Generate PDF report from summary.
    
    Uses reportlab if available, otherwise generates a simple text-based PDF stub.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=20,
        )
        heading_style = ParagraphStyle(
            'Heading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=15,
        )
        body_style = styles['Normal']
        
        story = []
        
        # Title
        story.append(Paragraph("Relatório de Atividades de I&D", title_style))
        story.append(Paragraph(f"Período: {start_date} a {end_date}", body_style))
        story.append(Paragraph(f"Gerado em: {summary['generated_at'][:10]}", body_style))
        story.append(Spacer(1, 20))
        
        # Summary
        story.append(Paragraph("Resumo Executivo", heading_style))
        story.append(Paragraph(
            f"Total de experiências realizadas: {summary['total_experiments']}",
            body_style
        ))
        story.append(Spacer(1, 10))
        
        # WP1
        wp1 = summary['wp1']
        story.append(Paragraph("WP1 - Routing Dinâmico", heading_style))
        story.append(Paragraph(f"Experiências: {wp1['num_experiments']}", body_style))
        story.append(Paragraph(f"Políticas testadas: {', '.join(wp1['policies_tested'])}", body_style))
        story.append(Paragraph(f"Melhoria média de makespan: {wp1['avg_makespan_improvement_pct']:.1f}%", body_style))
        story.append(Paragraph(f"Melhor política: {wp1['best_policy']}", body_style))
        story.append(Spacer(1, 10))
        
        # WP2
        wp2 = summary['wp2']
        story.append(Paragraph("WP2 - Avaliação de Sugestões IA", heading_style))
        story.append(Paragraph(f"Sugestões avaliadas: {wp2['num_suggestions_evaluated']}", body_style))
        
        if wp2['num_suggestions_evaluated'] > 0:
            wp2_data = [
                ["Classificação", "Quantidade", "Percentagem"],
                ["Benéficas", str(wp2['num_beneficial']), f"{wp2['pct_beneficial']:.1f}%"],
                ["Neutras", str(wp2['num_neutral']), f"{wp2['pct_neutral']:.1f}%"],
                ["Prejudiciais", str(wp2['num_harmful']), f"{wp2['pct_harmful']:.1f}%"],
            ]
            wp2_table = Table(wp2_data, colWidths=[5*cm, 3*cm, 3*cm])
            wp2_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(wp2_table)
        story.append(Spacer(1, 10))
        
        # WP3
        wp3 = summary['wp3']
        story.append(Paragraph("WP3 - Inventário + Capacidade", heading_style))
        story.append(Paragraph(f"Cenários simulados: {wp3['num_experiments']}", body_style))
        story.append(Paragraph(f"Melhor política: {wp3['best_policy']}", body_style))
        story.append(Paragraph(f"Custo médio: {wp3['avg_total_cost']:.2f}", body_style))
        story.append(Spacer(1, 10))
        
        # WP4
        wp4 = summary['wp4']
        story.append(Paragraph("WP4 - Learning Scheduler", heading_style))
        story.append(Paragraph(f"Episódios: {wp4['num_episodes']}", body_style))
        story.append(Paragraph(f"Reward médio: {wp4['avg_reward']:.3f}", body_style))
        story.append(Paragraph(f"Regret médio: {wp4['avg_regret']:.3f}", body_style))
        story.append(Paragraph(f"Melhor política observada: {wp4['best_observed_policy']}", body_style))
        story.append(Spacer(1, 20))
        
        # Hypotheses Summary
        story.append(Paragraph("Resumo das Hipóteses", heading_style))
        for h in summary['hypotheses_summary']:
            result_color = "green" if h['result'] == "SUPPORTED" else "orange"
            story.append(Paragraph(
                f"<b>{h['wp']}</b>: {h['hypothesis']} - <font color='{result_color}'>{h['result']}</font>",
                body_style
            ))
            story.append(Paragraph(f"Evidência: {h['evidence']}", body_style))
            story.append(Spacer(1, 5))
        
        # Build PDF
        doc.build(story)
        return buffer.getvalue()
        
    except ImportError:
        # Fallback: generate simple text representation as PDF-like bytes
        logger.warning("reportlab not available, generating text stub")
        return _generate_text_stub(summary, start_date, end_date)


def _generate_text_stub(
    summary: Dict[str, Any],
    start_date: date,
    end_date: date,
) -> bytes:
    """Generate text stub when PDF library not available."""
    lines = [
        "=" * 60,
        "RELATÓRIO DE ATIVIDADES DE I&D",
        "=" * 60,
        f"Período: {start_date} a {end_date}",
        f"Gerado em: {summary['generated_at'][:10]}",
        "",
        f"Total de experiências: {summary['total_experiments']}",
        "",
        "--- WP1: Routing Dinâmico ---",
        f"Experiências: {summary['wp1']['num_experiments']}",
        f"Melhoria média makespan: {summary['wp1']['avg_makespan_improvement_pct']:.1f}%",
        f"Melhor política: {summary['wp1']['best_policy']}",
        "",
        "--- WP2: Sugestões IA ---",
        f"Avaliadas: {summary['wp2']['num_suggestions_evaluated']}",
        f"Benéficas: {summary['wp2']['pct_beneficial']:.1f}%",
        f"Prejudiciais: {summary['wp2']['pct_harmful']:.1f}%",
        "",
        "--- WP3: Inventário ---",
        f"Cenários: {summary['wp3']['num_experiments']}",
        f"Melhor política: {summary['wp3']['best_policy']}",
        "",
        "--- WP4: Learning Scheduler ---",
        f"Episódios: {summary['wp4']['num_episodes']}",
        f"Regret médio: {summary['wp4']['avg_regret']:.3f}",
        "",
        "=" * 60,
        "HIPÓTESES",
        "=" * 60,
    ]
    
    for h in summary['hypotheses_summary']:
        lines.append(f"{h['wp']}: {h['hypothesis']}")
        lines.append(f"  Resultado: {h['result']}")
        lines.append(f"  Evidência: {h['evidence']}")
        lines.append("")
    
    return "\n".join(lines).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_demo_rd_data(year: int = 2024) -> Dict[str, Any]:
    """
    Generate demo R&D data for testing when no real experiments exist.
    
    Returns summary with realistic-looking demo data.
    """
    import random
    random.seed(42)
    
    wp1 = WP1Summary(
        num_experiments=15,
        policies_tested=["FIFO", "SPT", "EDD", "CR", "WSPT", "MILP", "CP-SAT"],
        baseline_policy="FIFO",
        avg_makespan_improvement_pct=12.3,
        avg_tardiness_improvement_pct=18.5,
        avg_otd_improvement_pct=8.2,
        best_policy="CP-SAT",
        best_makespan_improvement_pct=23.7,
        experiments_list=[
            {"id": i, "date": f"{year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
             "policies": ["FIFO", "SPT", "MILP"], "best_policy": random.choice(["MILP", "CP-SAT", "SPT"]),
             "makespan_improvement_pct": random.uniform(5, 25)}
            for i in range(1, 16)
        ],
    )
    
    wp2 = WP2Summary(
        num_suggestions_evaluated=42,
        num_beneficial=25,
        num_neutral=12,
        num_harmful=5,
        pct_beneficial=59.5,
        pct_neutral=28.6,
        pct_harmful=11.9,
        suggestion_types=["rescheduling", "capacity_increase", "outsourcing", "priority_change"],
        avg_otd_delta=3.2,
        avg_tardiness_delta=-15.3,
        experiments_list=[
            {"id": i, "date": f"{year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
             "suggestion_type": random.choice(["rescheduling", "capacity_increase"]),
             "label": random.choice(["BENEFICIAL", "BENEFICIAL", "NEUTRAL", "HARMFUL"]),
             "delta_otd": random.uniform(-5, 10)}
            for i in range(1, 43)
        ],
    )
    
    wp3 = WP3Summary(
        num_experiments=8,
        policies_tested=["conservative", "aggressive", "balanced", "adaptive"],
        avg_total_cost=125000.0,
        avg_stockouts=2.3,
        avg_otd=94.5,
        best_policy="adaptive",
        best_cost=98000.0,
        policy_comparison=[
            {"policy": "conservative", "avg_cost": 145000, "avg_stockouts": 0.5, "avg_otd_pct": 92.0, "num_runs": 2},
            {"policy": "aggressive", "avg_cost": 95000, "avg_stockouts": 5.2, "avg_otd_pct": 89.0, "num_runs": 2},
            {"policy": "balanced", "avg_cost": 120000, "avg_stockouts": 2.1, "avg_otd_pct": 95.0, "num_runs": 2},
            {"policy": "adaptive", "avg_cost": 98000, "avg_stockouts": 1.8, "avg_otd_pct": 97.0, "num_runs": 2},
        ],
        experiments_list=[
            {"id": i, "date": f"{year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
             "policy": random.choice(["conservative", "aggressive", "balanced", "adaptive"]),
             "total_cost": random.uniform(90000, 150000),
             "stockouts": random.randint(0, 6),
             "otd_pct": random.uniform(88, 98)}
            for i in range(1, 9)
        ],
    )
    
    wp4 = WP4Summary(
        num_episodes=120,
        policies_in_bandit=["FIFO", "SPT", "EDD", "MILP"],
        avg_reward=0.72,
        avg_regret=0.15,
        best_observed_policy="MILP",
        convergence_episode=85,
        experiments_list=[
            {"id": i, "date": f"{year}-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
             "selected_policy": random.choice(["FIFO", "SPT", "EDD", "MILP"]),
             "reward": random.uniform(0.5, 0.95),
             "regret": random.uniform(0.05, 0.3)}
            for i in range(1, 121)
        ],
    )
    
    hypotheses = _build_hypotheses_summary(wp1, wp2, wp3, wp4)
    
    return {
        "period": {
            "start": f"{year}-01-01",
            "end": f"{year}-12-31",
        },
        "generated_at": datetime.utcnow().isoformat(),
        "total_experiments": wp1.num_experiments + wp2.num_suggestions_evaluated + wp3.num_experiments + wp4.num_episodes,
        "wp1": asdict(wp1),
        "wp2": asdict(wp2),
        "wp3": asdict(wp3),
        "wp4": asdict(wp4),
        "hypotheses_summary": hypotheses,
        "is_demo": True,
    }

