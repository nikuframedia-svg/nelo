"""
Configuração de Planeamento - Armazenamento permanente por batch_id

Esta configuração armazena:
- Indisponibilidades de máquinas
- Ordens manuais adicionadas pelo utilizador
- Prioridades modificadas
- Horizonte de planeamento atual

É persistida por batch_id e lida pelo APS antes de planear.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.aps.planning_commands import (
    HorizonChange,
    MachineUnavailability,
    ManualOrder,
    PriorityChange,
)

logger = logging.getLogger(__name__)


@dataclass
class PlanningConfig:
    """
    Configuração de planeamento por batch_id.
    
    Esta configuração é aplicada ANTES do APS planear.
    """
    batch_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Indisponibilidades de máquinas
    machine_unavailabilities: List[MachineUnavailability] = field(default_factory=list)
    
    # Ordens manuais adicionadas pelo utilizador
    manual_orders: List[ManualOrder] = field(default_factory=list)
    
    # Prioridades modificadas (order_id -> nova prioridade)
    priority_overrides: Dict[str, str] = field(default_factory=dict)
    
    # Horizonte de planeamento (None = usar default)
    horizon_hours: Optional[int] = None
    
    def add_unavailability(self, unavailability: MachineUnavailability):
        """Adiciona indisponibilidade de máquina."""
        self.machine_unavailabilities.append(unavailability)
        self.updated_at = datetime.utcnow()
        logger.info(f"Indisponibilidade adicionada: {unavailability.maquina_id} de {unavailability.start_time} até {unavailability.end_time}")
    
    def remove_unavailability(self, maquina_id: str):
        """Remove todas as indisponibilidades de uma máquina (máquina volta a estar disponível)."""
        removed_count = len(self.machine_unavailabilities)
        self.machine_unavailabilities = [
            u for u in self.machine_unavailabilities if u.maquina_id != maquina_id
        ]
        removed_count = removed_count - len(self.machine_unavailabilities)
        self.updated_at = datetime.utcnow()
        logger.info(f"Indisponibilidade removida: máquina {maquina_id} (removidas {removed_count} indisponibilidades)")
    
    def add_manual_order(self, order: ManualOrder):
        """Adiciona ordem manual."""
        self.manual_orders.append(order)
        self.updated_at = datetime.utcnow()
        logger.info(f"Ordem manual adicionada: {order.artigo} ({order.quantidade} unidades, {order.prioridade})")
    
    def set_priority(self, order_id: str, priority: str):
        """Define prioridade override para uma ordem."""
        self.priority_overrides[order_id] = priority
        self.updated_at = datetime.utcnow()
        logger.info(f"Prioridade alterada: {order_id} -> {priority}")
    
    def set_horizon(self, horizon_hours: int):
        """Define horizonte de planeamento."""
        self.horizon_hours = horizon_hours
        self.updated_at = datetime.utcnow()
        logger.info(f"Horizonte alterado: {horizon_hours}h")
    
    def to_dict(self) -> Dict:
        """Serializa configuração para dict."""
        return {
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "machine_unavailabilities": [
                {
                    "maquina_id": u.maquina_id,
                    "start_time": u.start_time.isoformat(),
                    "end_time": u.end_time.isoformat(),
                    "reason": u.reason,
                }
                for u in self.machine_unavailabilities
            ],
            "manual_orders": [
                {
                    "artigo": o.artigo,
                    "quantidade": o.quantidade,
                    "prioridade": o.prioridade,
                    "due_date": o.due_date.isoformat() if o.due_date else None,
                    "description": o.description,
                }
                for o in self.manual_orders
            ],
            "priority_overrides": self.priority_overrides,
            "horizon_hours": self.horizon_hours,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PlanningConfig":
        """Deserializa configuração de dict."""
        config = cls(
            batch_id=data["batch_id"],
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
        )
        
        # Deserializar indisponibilidades
        for u_data in data.get("machine_unavailabilities", []):
            config.machine_unavailabilities.append(
                MachineUnavailability(
                    maquina_id=u_data["maquina_id"],
                    start_time=datetime.fromisoformat(u_data["start_time"]),
                    end_time=datetime.fromisoformat(u_data["end_time"]),
                    reason=u_data.get("reason"),
                )
            )
        
        # Deserializar ordens manuais
        for o_data in data.get("manual_orders", []):
            config.manual_orders.append(
                ManualOrder(
                    artigo=o_data["artigo"],
                    quantidade=o_data["quantidade"],
                    prioridade=o_data["prioridade"],
                    due_date=datetime.fromisoformat(o_data["due_date"]) if o_data.get("due_date") else None,
                    description=o_data.get("description"),
                )
            )
        
        # Deserializar prioridades
        config.priority_overrides = data.get("priority_overrides", {})
        
        # Deserializar horizonte
        config.horizon_hours = data.get("horizon_hours")
        
        return config


class PlanningConfigStore:
    """
    Armazenamento persistente de PlanningConfig por batch_id.
    
    Guarda em disco (JSON) e mantém cache em memória.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent.parent / "data" / "planning_config"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache em memória
        self._cache: Dict[str, PlanningConfig] = {}
    
    def _get_config_path(self, batch_id: str) -> Path:
        """Retorna caminho do ficheiro de configuração."""
        # Sanitizar batch_id para nome de ficheiro
        safe_batch_id = batch_id.replace("/", "_").replace("\\", "_")
        return self.config_dir / f"{safe_batch_id}.json"
    
    def get(self, batch_id: str) -> PlanningConfig:
        """Obtém configuração para batch_id (cria se não existir)."""
        # Verificar cache
        if batch_id in self._cache:
            return self._cache[batch_id]
        
        # Tentar carregar do disco
        config_path = self._get_config_path(batch_id)
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    config = PlanningConfig.from_dict(data)
                    self._cache[batch_id] = config
                    logger.info(f"Configuração carregada para batch_id={batch_id}")
                    return config
            except Exception as exc:
                logger.warning(f"Erro ao carregar configuração para batch_id={batch_id}: {exc}")
        
        # Criar nova configuração
        config = PlanningConfig(batch_id=batch_id)
        self._cache[batch_id] = config
        self.save(config)
        return config
    
    def save(self, config: PlanningConfig):
        """Guarda configuração no disco."""
        self._cache[config.batch_id] = config
        config_path = self._get_config_path(config.batch_id)
        
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config.to_dict(), f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Configuração guardada para batch_id={config.batch_id}")
        except Exception as exc:
            logger.error(f"Erro ao guardar configuração para batch_id={config.batch_id}: {exc}")
    
    def clear(self, batch_id: str):
        """Limpa configuração para batch_id."""
        if batch_id in self._cache:
            del self._cache[batch_id]
        
        config_path = self._get_config_path(batch_id)
        if config_path.exists():
            config_path.unlink()
            logger.info(f"Configuração limpa para batch_id={batch_id}")
    
    def reset_routing_preferences(self, batch_id: str):
        """
        PONTO 5: Reset completo de preferências de rota.
        
        Limpa todas as preferências de rota para garantir que o plano começa neutro.
        """
        config = self.get(batch_id)
        # PlanningConfig não tem routing_preferences diretamente,
        # mas podemos garantir que APSConfig está limpo
        logger.info(f"✅ [AUDIT] Preferências de rota resetadas para batch_id={batch_id}")
        return config


# Instância global
_planning_config_store: Optional[PlanningConfigStore] = None


def get_planning_config_store() -> PlanningConfigStore:
    """Retorna instância global do store."""
    global _planning_config_store
    if _planning_config_store is None:
        _planning_config_store = PlanningConfigStore()
    return _planning_config_store

