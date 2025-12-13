"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    STOCK STATE (Digital Twin)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Este módulo implementa o "Digital Twin" de inventário, mantendo um estado
sincronizado de stock em tempo real para múltiplos armazéns.

Estrutura do Digital Twin:
──────────────────────────
    StockState:
        - Por SKU:
            - Quantidade atual por warehouse_id
            - Quantidade comprometida (reservada para ordens)
            - Quantidade em trânsito (a chegar)
            - Histórico de movimentos (para forecasting)
            - Última atualização

Mathematical Model:
──────────────────
    stock_available[sku, warehouse] = stock_on_hand[sku, warehouse] 
                                    - stock_committed[sku, warehouse]
    
    stock_global[sku] = Σ(stock_available[sku, w] for w in warehouses)
    
    stock_effective[sku, warehouse] = stock_available[sku, warehouse]
                                    + stock_in_transit[sku, warehouse]

TODO[R&D]: Future enhancements:
    - Distributed state synchronization (multi-node)
    - Event sourcing for full audit trail
    - Time-travel queries (stock state at any point in time)
    - Conflict resolution for concurrent updates
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from smart_inventory.iot_ingestion import StockEvent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WarehouseStock:
    """
    Estado de stock de um SKU num armazém específico.
    
    Attributes:
        sku: Stock Keeping Unit
        warehouse_id: Identificador do armazém
        quantity_on_hand: Quantidade física disponível
        quantity_committed: Quantidade reservada (ordens pendentes)
        quantity_in_transit: Quantidade em trânsito (a chegar)
        last_updated: Última atualização
        location_ids: Set de localizações onde o SKU está armazenado
    """
    sku: str
    warehouse_id: str
    quantity_on_hand: float = 0.0
    quantity_committed: float = 0.0
    quantity_in_transit: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    location_ids: Set[str] = field(default_factory=set)
    
    @property
    def quantity_available(self) -> float:
        """Quantidade disponível (on_hand - committed)."""
        return max(0.0, self.quantity_on_hand - self.quantity_committed)
    
    @property
    def quantity_effective(self) -> float:
        """Quantidade efetiva (available + in_transit)."""
        return self.quantity_available + self.quantity_in_transit
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "sku": self.sku,
            "warehouse_id": self.warehouse_id,
            "quantity_on_hand": self.quantity_on_hand,
            "quantity_committed": self.quantity_committed,
            "quantity_in_transit": self.quantity_in_transit,
            "quantity_available": self.quantity_available,
            "quantity_effective": self.quantity_effective,
            "last_updated": self.last_updated.isoformat(),
            "location_ids": list(self.location_ids),
        }


@dataclass
class StockState:
    """
    Digital Twin de inventário multi-armazém.
    
    Mantém estado sincronizado de stock para todos os SKUs e armazéns.
    Imutável: cada operação retorna um novo StockState.
    
    Attributes:
        warehouses: Dict[warehouse_id, Dict[sku, WarehouseStock]]
        event_history: Lista de eventos aplicados (para auditoria)
        last_snapshot: Timestamp da última snapshot
    """
    warehouses: Dict[str, Dict[str, WarehouseStock]] = field(default_factory=lambda: defaultdict(dict))
    event_history: List[StockEvent] = field(default_factory=list)
    last_snapshot: Optional[datetime] = None
    
    def get_warehouse_stock(self, sku: str, warehouse_id: str) -> Optional[WarehouseStock]:
        """Obtém stock de um SKU num armazém."""
        return self.warehouses.get(warehouse_id, {}).get(sku)
    
    def get_all_skus(self) -> Set[str]:
        """Obtém conjunto de todos os SKUs."""
        skus = set()
        for warehouse_stocks in self.warehouses.values():
            skus.update(warehouse_stocks.keys())
        return skus
    
    def get_all_warehouses(self) -> Set[str]:
        """Obtém conjunto de todos os armazéns."""
        return set(self.warehouses.keys())
    
    def apply_event(self, event: StockEvent) -> StockState:
        """
        Aplica um evento de stock, retornando novo estado (imutável).
        
        Args:
            event: Evento de stock a aplicar
        
        Returns:
            Novo StockState com evento aplicado
        """
        # Criar cópia do estado
        new_warehouses = {}
        for wh_id, stocks in self.warehouses.items():
            new_warehouses[wh_id] = {sku: WarehouseStock(
                sku=ws.sku,
                warehouse_id=ws.warehouse_id,
                quantity_on_hand=ws.quantity_on_hand,
                quantity_committed=ws.quantity_committed,
                quantity_in_transit=ws.quantity_in_transit,
                last_updated=ws.last_updated,
                location_ids=ws.location_ids.copy(),
            ) for sku, ws in stocks.items()}
        
        # Obter ou criar WarehouseStock para este SKU/armazém
        if event.warehouse_id not in new_warehouses:
            new_warehouses[event.warehouse_id] = {}
        
        if event.sku not in new_warehouses[event.warehouse_id]:
            new_warehouses[event.warehouse_id][event.sku] = WarehouseStock(
                sku=event.sku,
                warehouse_id=event.warehouse_id,
            )
        
        warehouse_stock = new_warehouses[event.warehouse_id][event.sku]
        
        # Aplicar mudança de quantidade
        new_quantity = warehouse_stock.quantity_on_hand + event.quantity_change
        
        # Validar (não permitir stock negativo)
        if new_quantity < 0:
            logger.warning(
                f"Evento resultaria em stock negativo para {event.sku} em {event.warehouse_id}. "
                f"Atual: {warehouse_stock.quantity_on_hand}, Mudança: {event.quantity_change}"
            )
            new_quantity = 0.0
        
        # Atualizar WarehouseStock
        new_warehouses[event.warehouse_id][event.sku] = WarehouseStock(
            sku=event.sku,
            warehouse_id=event.warehouse_id,
            quantity_on_hand=new_quantity,
            quantity_committed=warehouse_stock.quantity_committed,
            quantity_in_transit=warehouse_stock.quantity_in_transit,
            last_updated=event.timestamp,
            location_ids=warehouse_stock.location_ids | ({event.location_id} if event.location_id else set()),
        )
        
        # Criar novo estado com evento adicionado ao histórico
        new_history = self.event_history + [event]
        
        return StockState(
            warehouses=new_warehouses,
            event_history=new_history,
            last_snapshot=self.last_snapshot,
        )
    
    def commit_stock(self, sku: str, warehouse_id: str, quantity: float) -> StockState:
        """
        Reserva stock (compromete para uma encomenda).
        
        Args:
            sku: SKU a reservar
            warehouse_id: Armazém
            quantity: Quantidade a reservar
        
        Returns:
            Novo StockState com stock comprometido
        """
        new_warehouses = {}
        for wh_id, stocks in self.warehouses.items():
            new_warehouses[wh_id] = {sku: WarehouseStock(
                sku=ws.sku,
                warehouse_id=ws.warehouse_id,
                quantity_on_hand=ws.quantity_on_hand,
                quantity_committed=ws.quantity_committed,
                quantity_in_transit=ws.quantity_in_transit,
                last_updated=ws.last_updated,
                location_ids=ws.location_ids.copy(),
            ) for sku, ws in stocks.items()}
        
        if warehouse_id not in new_warehouses:
            new_warehouses[warehouse_id] = {}
        
        if sku not in new_warehouses[warehouse_id]:
            new_warehouses[warehouse_id][sku] = WarehouseStock(
                sku=sku,
                warehouse_id=warehouse_id,
            )
        
        ws = new_warehouses[warehouse_id][sku]
        new_committed = ws.quantity_committed + quantity
        
        # Validar que não comprometemos mais do que disponível
        available = ws.quantity_available
        if new_committed > ws.quantity_on_hand:
            logger.warning(
                f"Tentativa de comprometer {quantity} de {sku} em {warehouse_id}, "
                f"mas apenas {available} disponível"
            )
            new_committed = ws.quantity_on_hand
        
        new_warehouses[warehouse_id][sku] = WarehouseStock(
            sku=sku,
            warehouse_id=warehouse_id,
            quantity_on_hand=ws.quantity_on_hand,
            quantity_committed=new_committed,
            quantity_in_transit=ws.quantity_in_transit,
            last_updated=datetime.now(timezone.utc),
            location_ids=ws.location_ids.copy(),
        )
        
        return StockState(
            warehouses=new_warehouses,
            event_history=self.event_history,
            last_snapshot=self.last_snapshot,
        )
    
    def release_stock(self, sku: str, warehouse_id: str, quantity: float) -> StockState:
        """
        Liberta stock comprometido.
        
        Args:
            sku: SKU a libertar
            warehouse_id: Armazém
            quantity: Quantidade a libertar
        
        Returns:
            Novo StockState com stock libertado
        """
        new_warehouses = {}
        for wh_id, stocks in self.warehouses.items():
            new_warehouses[wh_id] = {sku: WarehouseStock(
                sku=ws.sku,
                warehouse_id=ws.warehouse_id,
                quantity_on_hand=ws.quantity_on_hand,
                quantity_committed=ws.quantity_committed,
                quantity_in_transit=ws.quantity_in_transit,
                last_updated=ws.last_updated,
                location_ids=ws.location_ids.copy(),
            ) for sku, ws in stocks.items()}
        
        if warehouse_id in new_warehouses and sku in new_warehouses[warehouse_id]:
            ws = new_warehouses[warehouse_id][sku]
            new_committed = max(0.0, ws.quantity_committed - quantity)
            
            new_warehouses[warehouse_id][sku] = WarehouseStock(
                sku=sku,
                warehouse_id=warehouse_id,
                quantity_on_hand=ws.quantity_on_hand,
                quantity_committed=new_committed,
                quantity_in_transit=ws.quantity_in_transit,
                last_updated=datetime.now(timezone.utc),
                location_ids=ws.location_ids.copy(),
            )
        
        return StockState(
            warehouses=new_warehouses,
            event_history=self.event_history,
            last_snapshot=self.last_snapshot,
        )
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converte estado para DataFrame para análise.
        
        Returns:
            DataFrame com colunas: sku, warehouse_id, quantity_on_hand, 
            quantity_committed, quantity_available, quantity_effective
        """
        records = []
        for warehouse_id, stocks in self.warehouses.items():
            for sku, ws in stocks.items():
                records.append({
                    "sku": sku,
                    "warehouse_id": warehouse_id,
                    "quantity_on_hand": ws.quantity_on_hand,
                    "quantity_committed": ws.quantity_committed,
                    "quantity_in_transit": ws.quantity_in_transit,
                    "quantity_available": ws.quantity_available,
                    "quantity_effective": ws.quantity_effective,
                    "last_updated": ws.last_updated,
                })
        
        return pd.DataFrame(records)
    
    def get_movement_history(self, sku: str, warehouse_id: Optional[str] = None, days: int = 30) -> pd.DataFrame:
        """
        Obtém histórico de movimentos para um SKU.
        
        Args:
            sku: SKU a analisar
            warehouse_id: Armazém (opcional, se None considera todos)
            days: Número de dias de histórico
        
        Returns:
            DataFrame com eventos de movimento
        """
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=days)
        
        events = [
            e for e in self.event_history
            if e.sku == sku
            and e.timestamp >= cutoff
            and (warehouse_id is None or e.warehouse_id == warehouse_id)
        ]
        
        if not events:
            return pd.DataFrame(columns=["timestamp", "warehouse_id", "quantity_change", "source"])
        
        return pd.DataFrame([
            {
                "timestamp": e.timestamp,
                "warehouse_id": e.warehouse_id,
                "quantity_change": e.quantity_change,
                "source": e.source.value,
            }
            for e in events
        ])


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_stock_state(initial_data: Optional[pd.DataFrame] = None) -> StockState:
    """
    Cria um StockState inicial.
    
    Args:
        initial_data: DataFrame com colunas: sku, warehouse_id, quantity_on_hand
    
    Returns:
        StockState inicializado
    """
    state = StockState()
    
    if initial_data is not None:
        for _, row in initial_data.iterrows():
            sku = str(row["sku"])
            warehouse_id = str(row["warehouse_id"])
            quantity = float(row.get("quantity_on_hand", 0))
            
            if warehouse_id not in state.warehouses:
                state.warehouses[warehouse_id] = {}
            
            state.warehouses[warehouse_id][sku] = WarehouseStock(
                sku=sku,
                warehouse_id=warehouse_id,
                quantity_on_hand=quantity,
            )
    
    return state


def get_realtime_stock(stock_state: StockState, sku: str, warehouse_id: str) -> float:
    """
    Obtém stock em tempo real de um SKU num armazém.
    
    Args:
        stock_state: Estado de stock
        sku: SKU
        warehouse_id: Armazém
    
    Returns:
        Quantidade disponível (on_hand - committed)
    """
    ws = stock_state.get_warehouse_stock(sku, warehouse_id)
    if ws is None:
        return 0.0
    return ws.quantity_available


def get_global_stock(stock_state: StockState, sku: str) -> float:
    """
    Obtém stock global (soma de todos os armazéns) de um SKU.
    
    Args:
        stock_state: Estado de stock
        sku: SKU
    
    Returns:
        Quantidade total disponível em todos os armazéns
    """
    total = 0.0
    for warehouse_id in stock_state.get_all_warehouses():
        total += get_realtime_stock(stock_state, sku, warehouse_id)
    return total


def snapshot_stock_state(stock_state: StockState) -> Dict:
    """
    Cria snapshot do estado para persistência.
    
    Args:
        stock_state: Estado a snapshot
    
    Returns:
        Dicionário serializável
    """
    return {
        "warehouses": {
            wh_id: {
                sku: ws.to_dict()
                for sku, ws in stocks.items()
            }
            for wh_id, stocks in stock_state.warehouses.items()
        },
        "last_snapshot": datetime.now(timezone.utc).isoformat(),
        "total_events": len(stock_state.event_history),
    }


