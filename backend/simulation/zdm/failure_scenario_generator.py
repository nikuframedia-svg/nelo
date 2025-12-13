"""
════════════════════════════════════════════════════════════════════════════════════════════════════
FAILURE SCENARIO GENERATOR - Geração de Cenários de Falha para Simulação ZDM
════════════════════════════════════════════════════════════════════════════════════════════════════

Gera cenários de falha realistas baseados em:
- Distribuição de RUL do módulo Digital Twin
- Histórico de falhas (MTBF, MTTR)
- Padrões de degradação

Tipos de Falha:
- SUDDEN: Falha súbita de máquina (downtime imediato)
- GRADUAL: Degradação gradual (aumento de tempo de ciclo)
- QUALITY: Defeito de qualidade (retrabalho/rejeição)
- MATERIAL: Falta de material
- OPERATOR: Ausência de operador
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Optional, Tuple
import random
import numpy as np
import pandas as pd


class FailureType(Enum):
    """Tipos de falha suportados."""
    SUDDEN = "sudden"           # Falha súbita (máquina para)
    GRADUAL = "gradual"         # Degradação gradual (tempo de ciclo aumenta)
    QUALITY = "quality"         # Defeito de qualidade (retrabalho)
    MATERIAL = "material"       # Falta de material
    OPERATOR = "operator"       # Ausência de operador
    

@dataclass
class FailureScenario:
    """
    Representa um cenário de falha para simulação.
    """
    scenario_id: str
    failure_type: FailureType
    machine_id: str
    start_time: datetime
    duration_hours: float
    severity: float  # 0.0 (leve) a 1.0 (grave)
    
    # Parâmetros específicos por tipo
    cycle_time_increase_pct: float = 0.0  # Para GRADUAL
    quality_reject_rate: float = 0.0      # Para QUALITY
    
    # Metadados
    probability: float = 1.0              # Probabilidade do cenário
    triggered_by_rul: bool = False        # Se foi gerado com base em RUL baixo
    description: str = ""
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "scenario_id": self.scenario_id,
            "failure_type": self.failure_type.value,
            "machine_id": self.machine_id,
            "start_time": self.start_time.isoformat(),
            "duration_hours": self.duration_hours,
            "severity": round(self.severity, 2),
            "cycle_time_increase_pct": round(self.cycle_time_increase_pct, 1),
            "quality_reject_rate": round(self.quality_reject_rate, 2),
            "probability": round(self.probability, 3),
            "triggered_by_rul": self.triggered_by_rul,
            "description": self.description,
        }


@dataclass
class FailureConfig:
    """
    Configuração para geração de cenários de falha.
    """
    # Probabilidades por tipo de falha
    type_weights: Dict[FailureType, float] = field(default_factory=lambda: {
        FailureType.SUDDEN: 0.3,
        FailureType.GRADUAL: 0.25,
        FailureType.QUALITY: 0.2,
        FailureType.MATERIAL: 0.15,
        FailureType.OPERATOR: 0.1,
    })
    
    # Parâmetros de duração (horas)
    sudden_duration_range: Tuple[float, float] = (2.0, 24.0)
    gradual_duration_range: Tuple[float, float] = (8.0, 72.0)
    quality_duration_range: Tuple[float, float] = (1.0, 8.0)
    material_duration_range: Tuple[float, float] = (4.0, 48.0)
    operator_duration_range: Tuple[float, float] = (4.0, 12.0)
    
    # Parâmetros de severidade
    avg_severity: float = 0.5
    severity_std: float = 0.2
    
    # Parâmetros de degradação gradual
    cycle_time_increase_range: Tuple[float, float] = (10.0, 50.0)  # %
    
    # Parâmetros de qualidade
    reject_rate_range: Tuple[float, float] = (5.0, 30.0)  # %
    
    # Uso de RUL
    use_rul_data: bool = True
    rul_critical_threshold_hours: float = 100.0  # Máquinas com RUL < X são prioritárias
    rul_weight_multiplier: float = 3.0  # Peso extra para máquinas com RUL baixo


def generate_single_failure(
    machine_id: str,
    failure_type: FailureType,
    plan_start: datetime,
    plan_end: datetime,
    config: FailureConfig,
    scenario_idx: int = 0,
    rul_hours: Optional[float] = None,
) -> FailureScenario:
    """
    Gera um único cenário de falha.
    
    Args:
        machine_id: ID da máquina
        failure_type: Tipo de falha
        plan_start: Início do horizonte de planeamento
        plan_end: Fim do horizonte de planeamento
        config: Configuração de geração
        scenario_idx: Índice do cenário (para ID único)
        rul_hours: RUL da máquina (opcional)
    
    Returns:
        FailureScenario configurado
    """
    # Gerar tempo de início aleatório dentro do horizonte
    horizon_hours = (plan_end - plan_start).total_seconds() / 3600
    
    # Se temos RUL, a falha tende a ocorrer perto do fim do RUL
    if rul_hours is not None and rul_hours < config.rul_critical_threshold_hours:
        # Falha mais provável perto do fim do RUL
        mean_offset = min(rul_hours * 0.8, horizon_hours * 0.5)
        start_offset_hours = max(0, np.random.normal(mean_offset, rul_hours * 0.2))
        start_offset_hours = min(start_offset_hours, horizon_hours - 1)
        triggered_by_rul = True
    else:
        # Distribuição uniforme
        start_offset_hours = random.uniform(0, max(1, horizon_hours - 4))
        triggered_by_rul = False
    
    start_time = plan_start + timedelta(hours=start_offset_hours)
    
    # Gerar duração baseada no tipo
    duration_range = {
        FailureType.SUDDEN: config.sudden_duration_range,
        FailureType.GRADUAL: config.gradual_duration_range,
        FailureType.QUALITY: config.quality_duration_range,
        FailureType.MATERIAL: config.material_duration_range,
        FailureType.OPERATOR: config.operator_duration_range,
    }[failure_type]
    
    duration_hours = random.uniform(*duration_range)
    
    # Gerar severidade
    severity = np.clip(np.random.normal(config.avg_severity, config.severity_std), 0.1, 1.0)
    
    # Parâmetros específicos
    cycle_time_increase_pct = 0.0
    quality_reject_rate = 0.0
    
    if failure_type == FailureType.GRADUAL:
        cycle_time_increase_pct = random.uniform(*config.cycle_time_increase_range) * severity
    elif failure_type == FailureType.QUALITY:
        quality_reject_rate = random.uniform(*config.reject_rate_range) * severity / 100.0
    
    # Calcular probabilidade (baseada em RUL se disponível)
    probability = 0.5
    if rul_hours is not None:
        if rul_hours < 50:
            probability = 0.9
        elif rul_hours < 100:
            probability = 0.7
        elif rul_hours < 200:
            probability = 0.4
        else:
            probability = 0.2
    
    # Gerar descrição
    descriptions = {
        FailureType.SUDDEN: f"Falha súbita na máquina {machine_id} - downtime de {duration_hours:.1f}h",
        FailureType.GRADUAL: f"Degradação gradual em {machine_id} - tempo de ciclo +{cycle_time_increase_pct:.0f}%",
        FailureType.QUALITY: f"Problema de qualidade em {machine_id} - taxa de rejeição {quality_reject_rate*100:.0f}%",
        FailureType.MATERIAL: f"Falta de material para {machine_id} - espera de {duration_hours:.1f}h",
        FailureType.OPERATOR: f"Operador indisponível para {machine_id} - ausência de {duration_hours:.1f}h",
    }
    
    return FailureScenario(
        scenario_id=f"ZDM-{scenario_idx:04d}-{failure_type.value[:3].upper()}-{machine_id}",
        failure_type=failure_type,
        machine_id=machine_id,
        start_time=start_time,
        duration_hours=duration_hours,
        severity=severity,
        cycle_time_increase_pct=cycle_time_increase_pct,
        quality_reject_rate=quality_reject_rate,
        probability=probability,
        triggered_by_rul=triggered_by_rul,
        description=descriptions[failure_type],
    )


def generate_failure_scenarios(
    plan_df: pd.DataFrame,
    n_scenarios: int = 10,
    rul_info: Optional[Dict[str, float]] = None,
    config: Optional[FailureConfig] = None,
) -> List[FailureScenario]:
    """
    Gera múltiplos cenários de falha para simulação.
    
    Args:
        plan_df: DataFrame do plano de produção com colunas:
                 - machine_id: ID da máquina
                 - start_time: Tempo de início
                 - end_time: Tempo de fim
        n_scenarios: Número de cenários a gerar
        rul_info: Dicionário {machine_id: rul_hours} do módulo Digital Twin
        config: Configuração de geração (usa default se None)
    
    Returns:
        Lista de FailureScenario
    """
    if config is None:
        config = FailureConfig()
    
    if plan_df.empty:
        return []
    
    # Extrair horizonte de planeamento
    try:
        if 'start_time' in plan_df.columns:
            start_col = 'start_time'
            end_col = 'end_time'
        else:
            start_col = plan_df.columns[plan_df.columns.str.contains('start', case=False)][0]
            end_col = plan_df.columns[plan_df.columns.str.contains('end', case=False)][0]
        
        plan_df[start_col] = pd.to_datetime(plan_df[start_col])
        plan_df[end_col] = pd.to_datetime(plan_df[end_col])
        
        plan_start = plan_df[start_col].min()
        plan_end = plan_df[end_col].max()
    except Exception:
        # Fallback: usar horizonte de 7 dias
        plan_start = datetime.now()
        plan_end = plan_start + timedelta(days=7)
    
    # Extrair máquinas do plano
    machines = plan_df['machine_id'].unique().tolist() if 'machine_id' in plan_df.columns else []
    
    if not machines:
        # Máquinas de exemplo
        machines = [f"M-{i:03d}" for i in range(101, 106)]
    
    # Calcular pesos por máquina baseado em RUL
    machine_weights = {}
    for machine_id in machines:
        weight = 1.0
        if rul_info and machine_id in rul_info:
            rul = rul_info[machine_id]
            if rul < config.rul_critical_threshold_hours:
                weight *= config.rul_weight_multiplier
        machine_weights[machine_id] = weight
    
    # Normalizar pesos
    total_weight = sum(machine_weights.values())
    machine_probs = {m: w / total_weight for m, w in machine_weights.items()}
    
    # Preparar tipos de falha com pesos
    failure_types = list(config.type_weights.keys())
    failure_probs = list(config.type_weights.values())
    total_prob = sum(failure_probs)
    failure_probs = [p / total_prob for p in failure_probs]
    
    # Gerar cenários
    scenarios = []
    for i in range(n_scenarios):
        # Selecionar máquina (com bias para máquinas com RUL baixo)
        machine_id = np.random.choice(
            list(machine_probs.keys()),
            p=list(machine_probs.values())
        )
        
        # Selecionar tipo de falha
        failure_type = np.random.choice(failure_types, p=failure_probs)
        
        # Obter RUL se disponível
        rul_hours = rul_info.get(machine_id) if rul_info else None
        
        # Gerar cenário
        scenario = generate_single_failure(
            machine_id=machine_id,
            failure_type=failure_type,
            plan_start=plan_start,
            plan_end=plan_end,
            config=config,
            scenario_idx=i,
            rul_hours=rul_hours,
        )
        scenarios.append(scenario)
    
    # Ordenar por probabilidade (mais prováveis primeiro)
    scenarios.sort(key=lambda s: (-s.probability, s.start_time))
    
    return scenarios


def generate_cascading_failure_scenarios(
    plan_df: pd.DataFrame,
    initial_failure: FailureScenario,
    cascade_probability: float = 0.3,
    max_cascade_depth: int = 3,
    config: Optional[FailureConfig] = None,
) -> List[FailureScenario]:
    """
    Gera cenários de falha em cascata (uma falha causa outras).
    
    TODO[R&D]: Implementar modelo de propagação de falhas baseado em:
    - Dependências entre máquinas
    - Fluxo de produção
    - Histórico de falhas correlacionadas
    
    Args:
        plan_df: DataFrame do plano
        initial_failure: Falha inicial que dispara a cascata
        cascade_probability: Probabilidade de cada falha causar outra
        max_cascade_depth: Profundidade máxima da cascata
        config: Configuração
    
    Returns:
        Lista de cenários incluindo a falha inicial e cascatas
    """
    if config is None:
        config = FailureConfig()
    
    scenarios = [initial_failure]
    current_failures = [initial_failure]
    
    machines = plan_df['machine_id'].unique().tolist() if 'machine_id' in plan_df.columns else []
    
    for depth in range(max_cascade_depth):
        next_failures = []
        for failure in current_failures:
            if random.random() < cascade_probability:
                # Selecionar máquina diferente
                other_machines = [m for m in machines if m != failure.machine_id]
                if other_machines:
                    cascade_machine = random.choice(other_machines)
                    
                    # A cascata ocorre depois da falha original
                    cascade_start = failure.start_time + timedelta(hours=failure.duration_hours * 0.5)
                    
                    cascade_failure = FailureScenario(
                        scenario_id=f"{failure.scenario_id}-CASCADE-{depth+1}",
                        failure_type=FailureType.SUDDEN,  # Cascatas são geralmente súbitas
                        machine_id=cascade_machine,
                        start_time=cascade_start,
                        duration_hours=failure.duration_hours * 0.5,  # Menor duração
                        severity=failure.severity * 0.8,  # Menor severidade
                        probability=failure.probability * cascade_probability,
                        triggered_by_rul=False,
                        description=f"Falha em cascata em {cascade_machine} causada por {failure.machine_id}",
                    )
                    next_failures.append(cascade_failure)
                    scenarios.append(cascade_failure)
        
        current_failures = next_failures
        if not current_failures:
            break
    
    return scenarios



