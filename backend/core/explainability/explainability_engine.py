"""
═══════════════════════════════════════════════════════════════════════════════
                 PRODPLAN 4.0 - EXPLAINABILITY ENGINE
═══════════════════════════════════════════════════════════════════════════════

This module provides human-readable explanations for APS and ML decisions,
integrating Signal-to-Noise Ratio (SNR) to communicate confidence levels.

Explainable AI (XAI) Principles
═══════════════════════════════

1. TRANSPARENCY: Make decision process understandable
2. CAUSALITY: Explain why this decision, not just what
3. CONFIDENCE: Quantify uncertainty using SNR
4. ACTIONABILITY: Suggest what could change the decision

SNR-Based Confidence Communication
──────────────────────────────────

SNR provides a principled way to communicate prediction reliability:

    SNR > 10  : "Alta confiança" (High confidence)
                "Esta previsão baseia-se em dados altamente consistentes."
    
    3 < SNR ≤ 10 : "Confiança moderada" (Moderate confidence)
                   "Esta previsão é fiável, mas deve ser monitorizada."
    
    1 < SNR ≤ 3  : "Confiança limitada" (Limited confidence)
                   "Esta previsão tem incerteza significativa."
    
    SNR ≤ 1   : "Baixa confiança" (Low confidence)
                "Os dados apresentam alta variabilidade."

Language Standards
──────────────────
All explanations are in Portuguese (Portugal) with industrial terminology.

R&D / SIFIDE: WP4 - Evaluation & Explainability
Research Questions:
- Q4.4: Do SNR-based explanations improve user trust calibration?
- Q4.5: Which explanation factors are most useful for operators?

References:
[1] Miller, T. (2019). Explanation in Artificial Intelligence: Insights from
    the Social Sciences. Artificial Intelligence.
[2] Ribeiro et al. (2016). "Why Should I Trust You?": Explaining the
    Predictions of Any Classifier. KDD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SNR INTERPRETATION
# ════════════════════════════════════════════════════════════════════════════

def interpret_snr_level(snr: float) -> Tuple[str, str, str]:
    """
    Interpret SNR value for human communication.
    
    Returns:
        (level_code, level_name_pt, confidence_phrase_pt)
    
    Based on statistical thresholds:
    - SNR > 10: Equivalent to R² > 0.91 (excellent model fit)
    - SNR > 3: Equivalent to R² > 0.75 (good model fit)
    - SNR > 1: Equivalent to R² > 0.50 (acceptable)
    - SNR ≤ 1: Noise-dominated
    """
    if snr >= 10:
        return (
            "EXCELLENT",
            "Alta confiança",
            "Esta decisão baseia-se em dados altamente consistentes (SNR={:.1f})."
        )
    elif snr >= 3:
        return (
            "GOOD",
            "Confiança moderada",
            "Esta decisão é fiável, mas recomenda-se monitorização (SNR={:.1f})."
        )
    elif snr >= 1:
        return (
            "FAIR",
            "Confiança limitada",
            "Esta decisão tem incerteza significativa (SNR={:.1f}). Use com cautela."
        )
    else:
        return (
            "POOR",
            "Baixa confiança",
            "Os dados apresentam alta variabilidade (SNR={:.1f}). Decisão pouco fiável."
        )


def format_confidence_bar(snr: float, width: int = 10) -> str:
    """
    Create ASCII confidence bar from SNR.
    
    Example: SNR=5 → [████████░░]
    """
    # Map SNR to 0-1 scale using R² = SNR/(1+SNR)
    r_squared = snr / (1 + snr) if snr > 0 else 0
    filled = int(r_squared * width)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


# ════════════════════════════════════════════════════════════════════════════
# EXPLANATION DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ScheduleExplanation:
    """
    Explanation for a scheduling decision.
    """
    operation_id: str
    decision_type: str  # 'machine_assignment', 'timing', 'sequence'
    
    # The decision made
    decision_summary: str
    
    # Factors that influenced the decision
    factors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Alternatives considered
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    
    # SNR-based confidence
    snr: float = 1.0
    confidence_level: str = "FAIR"
    confidence_phrase: str = ""
    
    # What would change the decision
    what_if: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_id': self.operation_id,
            'decision_type': self.decision_type,
            'decision_summary': self.decision_summary,
            'factors': self.factors,
            'alternatives': self.alternatives,
            'snr': round(self.snr, 2),
            'confidence_level': self.confidence_level,
            'confidence_phrase': self.confidence_phrase,
            'what_if': self.what_if,
        }
    
    def to_text(self) -> str:
        """Generate human-readable explanation text."""
        lines = [
            f"═══ Explicação: {self.operation_id} ═══",
            "",
            f"📋 Decisão: {self.decision_summary}",
            "",
            f"🎯 Confiança: {self.confidence_level} {format_confidence_bar(self.snr)}",
            f"   {self.confidence_phrase}",
            "",
        ]
        
        if self.factors:
            lines.append("📊 Fatores considerados:")
            for f in self.factors:
                lines.append(f"   • {f.get('description', f)}")
        
        if self.alternatives:
            lines.append("")
            lines.append("🔄 Alternativas consideradas:")
            for a in self.alternatives:
                lines.append(f"   • {a.get('option', a)}: {a.get('reason_rejected', '')}")
        
        if self.what_if:
            lines.append("")
            lines.append("❓ O que mudaria esta decisão:")
            for w in self.what_if:
                lines.append(f"   • {w}")
        
        return "\n".join(lines)


@dataclass
class ForecastExplanation:
    """
    Explanation for a forecast/prediction.
    """
    target_id: str  # Article, machine, or entity being forecast
    forecast_type: str  # 'demand', 'setup_time', 'lead_time', 'rul'
    
    # The prediction
    predicted_value: float
    unit: str
    prediction_interval: Optional[Tuple[float, float]] = None
    
    # Model information
    model_name: str = ""
    features_used: List[str] = field(default_factory=list)
    
    # SNR-based confidence
    snr: float = 1.0
    confidence_level: str = "FAIR"
    confidence_phrase: str = ""
    
    # Historical context
    historical_mean: Optional[float] = None
    historical_std: Optional[float] = None
    n_historical_points: int = 0
    
    # Key drivers
    key_drivers: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'target_id': self.target_id,
            'forecast_type': self.forecast_type,
            'predicted_value': round(self.predicted_value, 2),
            'unit': self.unit,
            'prediction_interval': self.prediction_interval,
            'model_name': self.model_name,
            'features_used': self.features_used,
            'snr': round(self.snr, 2),
            'confidence_level': self.confidence_level,
            'confidence_phrase': self.confidence_phrase,
            'historical_mean': round(self.historical_mean, 2) if self.historical_mean else None,
            'historical_std': round(self.historical_std, 2) if self.historical_std else None,
            'n_historical_points': self.n_historical_points,
            'key_drivers': self.key_drivers,
        }
    
    def to_text(self) -> str:
        """Generate human-readable explanation text."""
        lines = [
            f"═══ Previsão: {self.target_id} ({self.forecast_type}) ═══",
            "",
            f"📈 Valor previsto: {self.predicted_value:.2f} {self.unit}",
        ]
        
        if self.prediction_interval:
            lines.append(f"   Intervalo 95%: [{self.prediction_interval[0]:.2f}, {self.prediction_interval[1]:.2f}]")
        
        lines.extend([
            "",
            f"🎯 Confiança: {self.confidence_level} {format_confidence_bar(self.snr)}",
            f"   {self.confidence_phrase}",
            "",
        ])
        
        if self.historical_mean is not None:
            lines.append(f"📊 Contexto histórico:")
            lines.append(f"   Média: {self.historical_mean:.2f} {self.unit}")
            if self.historical_std:
                lines.append(f"   Desvio padrão: {self.historical_std:.2f} {self.unit}")
            lines.append(f"   Pontos de dados: {self.n_historical_points}")
        
        if self.model_name:
            lines.append(f"")
            lines.append(f"🤖 Modelo: {self.model_name}")
        
        if self.features_used:
            lines.append(f"   Variáveis: {', '.join(self.features_used)}")
        
        if self.key_drivers:
            lines.append("")
            lines.append("🔑 Fatores principais:")
            for d in self.key_drivers:
                lines.append(f"   • {d.get('feature', d)}: {d.get('impact', '')}")
        
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# EXPLAINABILITY ENGINE
# ════════════════════════════════════════════════════════════════════════════

class ExplainabilityEngine:
    """
    Engine for generating explanations for APS and ML decisions.
    
    Usage:
        engine = ExplainabilityEngine()
        
        # Explain scheduling decision
        explanation = engine.explain_machine_assignment(
            operation_id='OP-001',
            assigned_machine='M-301',
            eligible_machines=['M-301', 'M-302', 'M-303'],
            processing_times={'M-301': 45, 'M-302': 50, 'M-303': 55},
            machine_loads={'M-301': 0.7, 'M-302': 0.8, 'M-303': 0.5},
            snr=5.3
        )
        print(explanation.to_text())
    """
    
    def __init__(self, language: str = 'pt-PT'):
        self.language = language
    
    def explain_machine_assignment(
        self,
        operation_id: str,
        assigned_machine: str,
        eligible_machines: List[str],
        processing_times: Dict[str, float],
        machine_loads: Optional[Dict[str, float]] = None,
        setup_times: Optional[Dict[str, float]] = None,
        snr: float = 1.0
    ) -> ScheduleExplanation:
        """
        Explain why an operation was assigned to a specific machine.
        
        Args:
            operation_id: ID of the operation
            assigned_machine: Machine that was assigned
            eligible_machines: List of machines that could process this operation
            processing_times: Dict machine_id -> processing time
            machine_loads: Dict machine_id -> current load (0-1)
            setup_times: Dict machine_id -> setup time from previous op
            snr: SNR of the assignment decision
        """
        level_code, level_name, conf_template = interpret_snr_level(snr)
        conf_phrase = conf_template.format(snr)
        
        # Build factors list
        factors = []
        
        # Processing time factor
        if processing_times:
            assigned_time = processing_times.get(assigned_machine, 0)
            min_time = min(processing_times.values())
            if assigned_time == min_time:
                factors.append({
                    'type': 'processing_time',
                    'description': f"Tempo de processamento mais rápido ({assigned_time:.0f} min)",
                    'weight': 'high'
                })
            else:
                factors.append({
                    'type': 'processing_time',
                    'description': f"Tempo de processamento: {assigned_time:.0f} min (mínimo: {min_time:.0f} min)",
                    'weight': 'medium'
                })
        
        # Machine load factor
        if machine_loads:
            assigned_load = machine_loads.get(assigned_machine, 0)
            min_load = min(machine_loads.values())
            if assigned_load == min_load:
                factors.append({
                    'type': 'machine_load',
                    'description': f"Máquina com menor carga ({assigned_load*100:.0f}%)",
                    'weight': 'high'
                })
            else:
                factors.append({
                    'type': 'machine_load',
                    'description': f"Carga da máquina: {assigned_load*100:.0f}%",
                    'weight': 'low'
                })
        
        # Setup time factor
        if setup_times:
            assigned_setup = setup_times.get(assigned_machine, 0)
            min_setup = min(setup_times.values())
            if assigned_setup == min_setup:
                factors.append({
                    'type': 'setup_time',
                    'description': f"Menor tempo de setup ({assigned_setup:.0f} min)",
                    'weight': 'high'
                })
        
        # Build alternatives
        alternatives = []
        for m in eligible_machines:
            if m != assigned_machine:
                reasons = []
                if processing_times and processing_times.get(m, 0) > processing_times.get(assigned_machine, 0):
                    diff = processing_times[m] - processing_times[assigned_machine]
                    reasons.append(f"+{diff:.0f} min processamento")
                if machine_loads and machine_loads.get(m, 0) > machine_loads.get(assigned_machine, 0):
                    reasons.append("maior carga")
                
                alternatives.append({
                    'option': m,
                    'reason_rejected': ", ".join(reasons) if reasons else "não selecionada"
                })
        
        # What-if scenarios
        what_if = []
        if machine_loads:
            min_load_machine = min(machine_loads, key=machine_loads.get)
            if min_load_machine != assigned_machine:
                what_if.append(
                    f"Se a máquina {min_load_machine} tivesse velocidade equivalente, seria preferida."
                )
        
        # Decision summary
        summary = f"Operação {operation_id} atribuída à máquina {assigned_machine}"
        if processing_times:
            summary += f" (tempo: {processing_times.get(assigned_machine, 0):.0f} min)"
        
        return ScheduleExplanation(
            operation_id=operation_id,
            decision_type='machine_assignment',
            decision_summary=summary,
            factors=factors,
            alternatives=alternatives[:3],  # Limit to top 3
            snr=snr,
            confidence_level=level_code,
            confidence_phrase=conf_phrase,
            what_if=what_if,
        )
    
    def explain_schedule_timing(
        self,
        operation_id: str,
        start_time: datetime,
        end_time: datetime,
        machine_id: str,
        predecessor_end: Optional[datetime] = None,
        machine_available: Optional[datetime] = None,
        due_date: Optional[datetime] = None,
        snr: float = 1.0
    ) -> ScheduleExplanation:
        """
        Explain why an operation was scheduled at a specific time.
        """
        level_code, level_name, conf_template = interpret_snr_level(snr)
        conf_phrase = conf_template.format(snr)
        
        factors = []
        
        # Predecessor constraint
        if predecessor_end:
            if start_time >= predecessor_end:
                factors.append({
                    'type': 'precedence',
                    'description': f"Operação anterior termina às {predecessor_end.strftime('%H:%M')}",
                    'weight': 'high'
                })
        
        # Machine availability
        if machine_available:
            if start_time >= machine_available:
                factors.append({
                    'type': 'machine_availability',
                    'description': f"Máquina {machine_id} disponível às {machine_available.strftime('%H:%M')}",
                    'weight': 'high'
                })
        
        # Due date impact
        what_if = []
        if due_date:
            if end_time <= due_date:
                factors.append({
                    'type': 'due_date',
                    'description': f"Conclusão antes da data devida ({due_date.strftime('%d/%m %H:%M')})",
                    'weight': 'medium'
                })
            else:
                lateness = (end_time - due_date).total_seconds() / 3600
                factors.append({
                    'type': 'due_date',
                    'description': f"ATRASO: {lateness:.1f}h após data devida",
                    'weight': 'critical'
                })
                what_if.append("Para cumprir o prazo, seria necessário antecipar operações anteriores.")
        
        summary = (
            f"Operação {operation_id} agendada para {start_time.strftime('%d/%m %H:%M')} - "
            f"{end_time.strftime('%H:%M')} na máquina {machine_id}"
        )
        
        return ScheduleExplanation(
            operation_id=operation_id,
            decision_type='timing',
            decision_summary=summary,
            factors=factors,
            alternatives=[],
            snr=snr,
            confidence_level=level_code,
            confidence_phrase=conf_phrase,
            what_if=what_if,
        )
    
    def explain_forecast(
        self,
        target_id: str,
        forecast_type: str,
        predicted_value: float,
        unit: str,
        model_name: str,
        features_used: List[str],
        snr: float,
        historical_data: Optional[Dict[str, Any]] = None,
        prediction_interval: Optional[Tuple[float, float]] = None
    ) -> ForecastExplanation:
        """
        Generate explanation for a forecast/prediction.
        
        Args:
            target_id: Entity being forecast (article_id, machine_id, etc.)
            forecast_type: Type of forecast (demand, lead_time, setup_time, rul)
            predicted_value: The predicted value
            unit: Unit of measurement
            model_name: Name of the model used
            features_used: Features that went into the prediction
            snr: Signal-to-Noise Ratio of the prediction
            historical_data: Optional dict with mean, std, n_points
            prediction_interval: Optional (lower, upper) bounds
        """
        level_code, level_name, conf_template = interpret_snr_level(snr)
        conf_phrase = conf_template.format(snr)
        
        # Key drivers based on forecast type
        key_drivers = []
        
        if forecast_type == 'demand':
            key_drivers.append({
                'feature': 'historical_pattern',
                'impact': 'Baseado em padrão histórico de consumo'
            })
            if 'seasonality' in features_used:
                key_drivers.append({
                    'feature': 'seasonality',
                    'impact': 'Ajustado para sazonalidade'
                })
        
        elif forecast_type == 'setup_time':
            key_drivers.append({
                'feature': 'family_transition',
                'impact': 'Baseado na transição entre famílias de setup'
            })
        
        elif forecast_type == 'lead_time':
            key_drivers.append({
                'feature': 'routing_complexity',
                'impact': 'Considera número de operações no percurso'
            })
            if 'machine_load' in features_used:
                key_drivers.append({
                    'feature': 'machine_load',
                    'impact': 'Ajustado pela carga atual das máquinas'
                })
        
        elif forecast_type == 'rul':
            key_drivers.append({
                'feature': 'degradation_signal',
                'impact': 'Baseado no padrão de degradação observado'
            })
        
        return ForecastExplanation(
            target_id=target_id,
            forecast_type=forecast_type,
            predicted_value=predicted_value,
            unit=unit,
            prediction_interval=prediction_interval,
            model_name=model_name,
            features_used=features_used,
            snr=snr,
            confidence_level=level_code,
            confidence_phrase=conf_phrase,
            historical_mean=historical_data.get('mean') if historical_data else None,
            historical_std=historical_data.get('std') if historical_data else None,
            n_historical_points=historical_data.get('n_points', 0) if historical_data else 0,
            key_drivers=key_drivers,
        )


# ════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def explain_schedule_decision(
    operation_id: str,
    machine_id: str,
    context: Dict[str, Any],
    snr: float = 1.0
) -> str:
    """
    Generate a simple text explanation for a scheduling decision.
    
    Args:
        operation_id: Operation identifier
        machine_id: Assigned machine
        context: Dict with processing_times, machine_loads, etc.
        snr: Signal-to-Noise Ratio
    
    Returns:
        Human-readable explanation text
    """
    engine = ExplainabilityEngine()
    
    explanation = engine.explain_machine_assignment(
        operation_id=operation_id,
        assigned_machine=machine_id,
        eligible_machines=context.get('eligible_machines', [machine_id]),
        processing_times=context.get('processing_times', {}),
        machine_loads=context.get('machine_loads', {}),
        setup_times=context.get('setup_times', {}),
        snr=snr,
    )
    
    return explanation.to_text()


def explain_forecast(
    target_id: str,
    model_type: str,
    predicted_value: float,
    unit: str,
    snr: float,
    metrics: Optional[Dict[str, float]] = None
) -> str:
    """
    Generate a simple text explanation for a forecast.
    
    Args:
        target_id: Entity being forecast
        model_type: Type of model (arima, xgboost, etc.)
        predicted_value: The prediction
        unit: Unit of measurement
        snr: Signal-to-Noise Ratio
        metrics: Optional performance metrics (mape, rmse)
    
    Returns:
        Human-readable explanation text
    """
    engine = ExplainabilityEngine()
    
    explanation = engine.explain_forecast(
        target_id=target_id,
        forecast_type=model_type,
        predicted_value=predicted_value,
        unit=unit,
        model_name=model_type,
        features_used=[],
        snr=snr,
        historical_data=metrics,
    )
    
    return explanation.to_text()



