"""
Comandos de Planeamento - Chat de Planeamento ProdPlan 4.0

Estrutura de comandos que o LLM gera a partir de linguagem natural.
O LLM NUNCA gera código, patches ou modifica ficheiros.
O LLM apenas traduz intenção → comando estruturado.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional
import logging

from app.aps.date_normalizer import normalize_time_range, validate_iso_date

logger = logging.getLogger(__name__)


class CommandType(str, Enum):
    """Tipos de comandos de planeamento."""
    MACHINE_UNAVAILABLE = "machine_unavailable"
    MACHINE_AVAILABLE = "machine_available"  # Remover indisponibilidade
    ADD_MANUAL_ORDER = "add_manual_order"
    CHANGE_PRIORITY = "change_priority"
    CHANGE_HORIZON = "change_horizon"
    RECALCULATE_PLAN = "recalculate_plan"  # Recalcular plano com configuração atual
    UNKNOWN = "unknown"  # Quando não consegue interpretar


@dataclass
class MachineUnavailability:
    """Comando: Marcar máquina indisponível."""
    maquina_id: str
    start_time: datetime
    end_time: datetime
    reason: Optional[str] = None  # "Avaria", "Manutenção", "Falta operador", etc.


@dataclass
class MachineAvailable:
    """Comando: Remover indisponibilidade de máquina (máquina volta a estar disponível)."""
    maquina_id: str


@dataclass
class ManualOrder:
    """Comando: Adicionar ordem manual."""
    artigo: str  # "GO Artigo 6", "GO3", etc.
    quantidade: int
    prioridade: str  # "VIP", "ALTA", "NORMAL", "BAIXA"
    due_date: Optional[datetime] = None
    description: Optional[str] = None  # Descrição opcional


@dataclass
class PriorityChange:
    """Comando: Alterar prioridade de ordem existente."""
    order_id: str  # "ORD-GO Artigo 6" ou artigo "GO Artigo 6"
    new_priority: str  # "VIP", "ALTA", "NORMAL", "BAIXA"


@dataclass
class HorizonChange:
    """Comando: Alterar horizonte de planeamento."""
    horizon_hours: int  # Novo horizonte em horas


@dataclass
class PlanningCommand:
    """
    Comando de planeamento estruturado.
    
    O LLM gera isto a partir de linguagem natural.
    NUNCA código, NUNCA patches, NUNCA ficheiros.
    """
    command_type: CommandType
    machine_unavailable: Optional[MachineUnavailability] = None
    machine_available: Optional[MachineAvailable] = None
    manual_order: Optional[ManualOrder] = None
    priority_change: Optional[PriorityChange] = None
    horizon_change: Optional[HorizonChange] = None
    confidence: float = 1.0  # 0.0-1.0: confiança na interpretação
    requires_clarification: bool = False  # True se precisa de clarificação
    clarification_message: Optional[str] = None  # Mensagem para pedir clarificação
    
    def to_dict(self) -> Dict:
        """Serializa comando para dict."""
        result = {
            "command_type": self.command_type.value,
            "confidence": self.confidence,
            "requires_clarification": self.requires_clarification,
            # Sempre incluir todos os campos, mesmo que None, para o frontend poder validar
            "machine_unavailable": None,
            "machine_available": None,
            "manual_order": None,
            "priority_change": None,
            "horizon_change": None,
        }
        
        if self.machine_unavailable:
            result["machine_unavailable"] = {
                "maquina_id": self.machine_unavailable.maquina_id,
                "start_time": self.machine_unavailable.start_time.isoformat(),
                "end_time": self.machine_unavailable.end_time.isoformat(),
                "reason": self.machine_unavailable.reason,
            }
        
        if self.machine_available:
            result["machine_available"] = {
                "maquina_id": self.machine_available.maquina_id,
            }
        
        if self.manual_order:
            result["manual_order"] = {
                "artigo": self.manual_order.artigo,
                "quantidade": self.manual_order.quantidade,
                "prioridade": self.manual_order.prioridade,
                "due_date": self.manual_order.due_date.isoformat() if self.manual_order.due_date else None,
                "description": self.manual_order.description,
            }
        
        if self.priority_change:
            result["priority_change"] = {
                "order_id": self.priority_change.order_id,
                "new_priority": self.priority_change.new_priority,
            }
        
        if self.horizon_change:
            result["horizon_change"] = {
                "horizon_hours": self.horizon_change.horizon_hours,
            }
        
        if self.clarification_message:
            result["clarification_message"] = self.clarification_message
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PlanningCommand":
        """Deserializa comando de dict com validação robusta e defaults."""
        # Normalizar command_type (aceitar variações)
        command_type_str = data.get("command_type", "unknown")
        
        # Mapear variações comuns para valores válidos
        command_type_map = {
            "machine_unavailability": "machine_unavailable",
            "machine_unavailable": "machine_unavailable",
            "add_manual_order": "add_manual_order",
            "manual_order": "add_manual_order",
            "change_priority": "change_priority",
            "priority_change": "change_priority",
            "change_horizon": "change_horizon",
            "horizon_change": "change_horizon",
            "machine_available": "machine_available",
            "remove_unavailability": "machine_available",
            "recalculate_plan": "recalculate_plan",
            "recalculate": "recalculate_plan",
            "refresh_plan": "recalculate_plan",
            "refresh": "recalculate_plan",
            "replan": "recalculate_plan",
            "replanning": "recalculate_plan",
            "unknown": "unknown",
        }
        
        normalized_type = command_type_map.get(command_type_str.lower(), command_type_str)
        
        try:
            command_type = CommandType(normalized_type)
        except ValueError:
            logger.warning(f"Tipo de comando inválido: '{command_type_str}'. Usando 'unknown'.")
            command_type = CommandType.UNKNOWN
        
        machine_unavailable = None
        # Aceitar tanto "machine_unavailable" quanto "machine_unavailability" no dict
        mu_data = data.get("machine_unavailable") or data.get("machine_unavailability")
        if mu_data:
            
            # Validação robusta com normalização automática
            start_time_str = mu_data.get("start_time")
            end_time_str = mu_data.get("end_time")
            horizon_hours = data.get("horizon_hours", 8)  # Tentar obter do contexto
            
            # Normalizar datas (converte humanas para ISO)
            start_iso, end_iso = normalize_time_range(
                start_time_str,
                end_time_str,
                reference_time=datetime.utcnow(),
                default_duration_hours=horizon_hours,
            )
            
            # Se normalização falhou, usar defaults
            if not start_iso:
                start_iso = datetime.utcnow().isoformat()
                logger.warning(f"start_time não pôde ser normalizado, usando default: {start_iso}")
            
            if not end_iso:
                end_iso = (datetime.utcnow() + timedelta(hours=horizon_hours)).isoformat()
                logger.warning(f"end_time não pôde ser normalizado, usando default: {end_iso}")
            
            # Validar que são ISO válidas
            if not validate_iso_date(start_iso):
                logger.error(f"start_time não é ISO válida após normalização: '{start_iso}'. Usando default.")
                start_iso = datetime.utcnow().isoformat()
            
            if not validate_iso_date(end_iso):
                logger.error(f"end_time não é ISO válida após normalização: '{end_iso}'. Usando default.")
                end_iso = (datetime.utcnow() + timedelta(hours=horizon_hours)).isoformat()
            
            # Parsear para datetime
            try:
                start_time = datetime.fromisoformat(start_iso.replace('Z', '+00:00'))
            except (ValueError, AttributeError) as e:
                logger.error(f"Erro ao parsear start_time '{start_iso}': {e}. Usando default.")
                start_time = datetime.utcnow()
            
            try:
                end_time = datetime.fromisoformat(end_iso.replace('Z', '+00:00'))
            except (ValueError, AttributeError) as e:
                logger.error(f"Erro ao parsear end_time '{end_iso}': {e}. Usando default.")
                end_time = datetime.utcnow() + timedelta(hours=horizon_hours)
            
            # Validar que end_time > start_time
            if end_time <= start_time:
                logger.warning(f"end_time ({end_time}) <= start_time ({start_time}). Ajustando end_time.")
                end_time = start_time + timedelta(hours=horizon_hours)
            
            machine_unavailable = MachineUnavailability(
                maquina_id=str(mu_data.get("maquina_id", "")),
                start_time=start_time,
                end_time=end_time,
                reason=mu_data.get("reason"),
            )
        
        manual_order = None
        if data.get("manual_order"):
            mo_data = data["manual_order"]
            
            # Validação robusta para due_date
            due_date = None
            if mo_data.get("due_date"):
                due_date_str = mo_data["due_date"]
                if isinstance(due_date_str, str) and due_date_str.strip():
                    try:
                        due_date = datetime.fromisoformat(due_date_str.replace('Z', '+00:00'))
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Erro ao parsear due_date '{due_date_str}': {e}. Ignorando.")
                        due_date = None
            
            manual_order = ManualOrder(
                artigo=str(mo_data.get("artigo", "")),
                quantidade=int(mo_data.get("quantidade", 0)),
                prioridade=str(mo_data.get("prioridade", "NORMAL")),
                due_date=due_date,
                description=mo_data.get("description"),
            )
        
        priority_change = None
        if data.get("priority_change"):
            pc_data = data["priority_change"]
            priority_change = PriorityChange(
                order_id=pc_data["order_id"],
                new_priority=pc_data["new_priority"],
            )
        
        horizon_change = None
        if data.get("horizon_change"):
            hc_data = data["horizon_change"]
            horizon_change = HorizonChange(
                horizon_hours=hc_data["horizon_hours"],
            )
        
        # Deserializar machine_available
        machine_available = None
        if data.get("machine_available"):
            ma_data = data["machine_available"]
            machine_available = MachineAvailable(
                maquina_id=str(ma_data.get("maquina_id", "")),
            )
        
        return cls(
            command_type=command_type,
            machine_unavailable=machine_unavailable,
            machine_available=machine_available,
            manual_order=manual_order,
            priority_change=priority_change,
            horizon_change=horizon_change,
            confidence=data.get("confidence", 1.0),
            requires_clarification=data.get("requires_clarification", False),
            clarification_message=data.get("clarification_message"),
        )

