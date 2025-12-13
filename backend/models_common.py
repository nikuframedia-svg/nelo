"""
ProdPlan 4.0 - Common Models
============================

Modelos Pydantic reutilizáveis para KPIs e contextos.
Usados em múltiplos módulos para garantir consistência.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING KPIS
# ═══════════════════════════════════════════════════════════════════════════════

class SchedulingKPIs(BaseModel):
    """
    KPIs de scheduling.
    
    Métricas industriais standard para avaliação de planos de produção.
    """
    makespan_hours: float = Field(
        default=0.0,
        description="Tempo total do plano (primeira operação até última)"
    )
    total_tardiness_hours: float = Field(
        default=0.0,
        description="Soma de atrasos (max(0, end - due_date) para cada ordem)"
    )
    num_late_orders: int = Field(
        default=0,
        description="Número de ordens atrasadas"
    )
    total_setup_time_hours: float = Field(
        default=0.0,
        description="Tempo total de setup/changeover"
    )
    avg_machine_utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Utilização média das máquinas (0-1)"
    )
    otd_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="On-Time Delivery rate (0-1)"
    )
    total_operations: int = Field(
        default=0,
        description="Número total de operações agendadas"
    )
    total_orders: int = Field(
        default=0,
        description="Número total de ordens"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "makespan_hours": 48.5,
                "total_tardiness_hours": 2.3,
                "num_late_orders": 3,
                "total_setup_time_hours": 4.2,
                "avg_machine_utilization": 0.78,
                "otd_rate": 0.95,
                "total_operations": 150,
                "total_orders": 25,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# INVENTORY KPIS
# ═══════════════════════════════════════════════════════════════════════════════

class InventoryKPIs(BaseModel):
    """
    KPIs de inventário.
    
    Métricas para avaliação de gestão de stock.
    """
    avg_stock_units: float = Field(
        default=0.0,
        description="Stock médio (unidades)"
    )
    stock_value_eur: float = Field(
        default=0.0,
        description="Valor do stock (€)"
    )
    stockout_days: int = Field(
        default=0,
        description="Dias com ruptura de stock"
    )
    backorders_count: int = Field(
        default=0,
        description="Número de backorders"
    )
    service_level: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Nível de serviço (0-1)"
    )
    inventory_turnover: float = Field(
        default=0.0,
        description="Rotatividade de inventário (vendas/stock médio)"
    )
    coverage_days: float = Field(
        default=0.0,
        description="Dias de cobertura (stock/consumo diário)"
    )
    rop_alerts: int = Field(
        default=0,
        description="Número de SKUs abaixo do ROP"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "avg_stock_units": 1500.0,
                "stock_value_eur": 75000.0,
                "stockout_days": 2,
                "backorders_count": 5,
                "service_level": 0.97,
                "inventory_turnover": 12.0,
                "coverage_days": 30.0,
                "rop_alerts": 3,
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RESILIENCE KPIS (ZDM)
# ═══════════════════════════════════════════════════════════════════════════════

class ResilienceKPIs(BaseModel):
    """
    KPIs de resiliência (Zero Disruption Manufacturing).
    
    Métricas de robustez do plano face a falhas.
    """
    resilience_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Score de resiliência (0-100)"
    )
    avg_recovery_time_hours: float = Field(
        default=0.0,
        description="Tempo médio de recuperação após falha"
    )
    avg_throughput_loss_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Perda média de throughput (%)"
    )
    avg_otd_impact_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Impacto médio no OTD (%)"
    )
    scenarios_simulated: int = Field(
        default=0,
        description="Número de cenários simulados"
    )
    full_recovery_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Taxa de recuperação total"
    )
    critical_machines: List[str] = Field(
        default_factory=list,
        description="Máquinas mais críticas identificadas"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "resilience_score": 72.5,
                "avg_recovery_time_hours": 3.2,
                "avg_throughput_loss_pct": 8.5,
                "avg_otd_impact_pct": 5.2,
                "scenarios_simulated": 100,
                "full_recovery_rate": 0.85,
                "critical_machines": ["M-305B", "M-412"],
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DIGITAL TWIN KPIS
# ═══════════════════════════════════════════════════════════════════════════════

class DigitalTwinKPIs(BaseModel):
    """
    KPIs de Digital Twin (Health & RUL).
    """
    overall_health_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Health score geral da fábrica (0-1)"
    )
    machines_healthy: int = Field(
        default=0,
        description="Número de máquinas saudáveis"
    )
    machines_degraded: int = Field(
        default=0,
        description="Número de máquinas degradadas"
    )
    machines_warning: int = Field(
        default=0,
        description="Número de máquinas em warning"
    )
    machines_critical: int = Field(
        default=0,
        description="Número de máquinas críticas"
    )
    avg_rul_hours: float = Field(
        default=0.0,
        description="RUL médio (horas)"
    )
    min_rul_hours: float = Field(
        default=0.0,
        description="RUL mínimo (horas)"
    )
    maintenance_recommendations: int = Field(
        default=0,
        description="Número de recomendações de manutenção"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CAUSAL KPIS
# ═══════════════════════════════════════════════════════════════════════════════

class CausalKPIs(BaseModel):
    """
    KPIs de análise causal.
    """
    complexity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Score de complexidade do sistema (0-100)"
    )
    n_variables: int = Field(
        default=0,
        description="Número de variáveis no grafo causal"
    )
    n_relations: int = Field(
        default=0,
        description="Número de relações causais"
    )
    n_tradeoffs: int = Field(
        default=0,
        description="Número de trade-offs identificados"
    )
    n_leverage_points: int = Field(
        default=0,
        description="Número de pontos de alavancagem"
    )
    n_risks: int = Field(
        default=0,
        description="Número de riscos identificados"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentContext(BaseModel):
    """
    Contexto de uma experiência R&D.
    
    Define o cenário onde a experiência foi executada.
    """
    factory_id: str = Field(
        default="default",
        description="ID da fábrica/instalação"
    )
    time_window_start: datetime = Field(
        default_factory=datetime.now,
        description="Início da janela temporal"
    )
    time_window_end: Optional[datetime] = Field(
        default=None,
        description="Fim da janela temporal"
    )
    scenario_name: Optional[str] = Field(
        default=None,
        description="Nome do cenário (ex: 'peak_demand', 'maintenance_window')"
    )
    dataset_version: Optional[str] = Field(
        default=None,
        description="Versão do dataset usado"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notas adicionais"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "factory_id": "FACTORY_001",
                "time_window_start": "2025-01-01T00:00:00",
                "time_window_end": "2025-01-31T23:59:59",
                "scenario_name": "high_demand_january",
                "dataset_version": "v2.3",
                "notes": "Test with new product mix",
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# R&D EXPERIMENT MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentStatus(str, Enum):
    """Status de uma experiência."""
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentSummary(BaseModel):
    """
    Resumo de uma experiência R&D.
    """
    experiment_id: int = Field(description="ID da experiência")
    name: str = Field(description="Nome da experiência")
    wp: str = Field(description="Work Package (WP1-WP4)")
    status: ExperimentStatus = Field(description="Status atual")
    created_at: datetime = Field(description="Data de criação")
    duration_sec: Optional[float] = Field(
        default=None,
        description="Duração em segundos"
    )
    conclusion: Optional[str] = Field(
        default=None,
        description="Conclusão principal"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "experiment_id": 42,
                "name": "routing_comparison_spt_vs_edd",
                "wp": "WP1",
                "status": "finished",
                "created_at": "2025-01-15T10:30:00",
                "duration_sec": 45.2,
                "conclusion": "EDD reduces tardiness by 15% vs SPT",
            }
        }


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATED KPIS
# ═══════════════════════════════════════════════════════════════════════════════

class AggregatedKPIs(BaseModel):
    """
    KPIs agregados de todos os módulos.
    
    Usado para dashboards executivos.
    """
    scheduling: Optional[SchedulingKPIs] = None
    inventory: Optional[InventoryKPIs] = None
    resilience: Optional[ResilienceKPIs] = None
    digital_twin: Optional[DigitalTwinKPIs] = None
    causal: Optional[CausalKPIs] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    def get_health_score(self) -> float:
        """Calcula score de saúde geral (0-100)."""
        scores = []
        
        if self.scheduling:
            scores.append(self.scheduling.otd_rate * 100)
            scores.append(self.scheduling.avg_machine_utilization * 100)
        
        if self.inventory:
            scores.append(self.inventory.service_level * 100)
        
        if self.resilience:
            scores.append(self.resilience.resilience_score)
        
        if self.digital_twin:
            scores.append(self.digital_twin.overall_health_score * 100)
        
        return sum(scores) / len(scores) if scores else 0.0

