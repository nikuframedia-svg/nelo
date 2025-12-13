"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    MULTI-WAREHOUSE OPTIMIZER (MILP)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Este módulo otimiza redistribuição de stock e compras entre múltiplos armazéns.

Mathematical Formulation (MILP):
─────────────────────────────────
    Variáveis:
        q_transfer[w1, w2, sku]: Quantidade transferida de w1 para w2
        q_order[w, sku]: Quantidade a encomendar para w
    
    Objetivo:
        min Σ(c_transfer * q_transfer) + Σ(c_order * q_order) + penalty_rupture
    
    Restrições:
        stock_final[w, sku] = stock_inicial + q_order + Σ_in q_transfer - Σ_out q_transfer - consumo_previsto
        stock_final[w, sku] >= safety_stock[w, sku]
        Σ_w q_order[w, sku] <= capacidade_fornecimento[sku]
        capacidade_armazenamento[w] >= Σ_sku stock_final[w, sku]

TODO[R&D]: Future enhancements:
    - Stochastic optimization (demanda incerta)
    - Multi-period optimization (horizonte estendido)
    - Integration with transportation routing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Warehouse:
    """
    Definição de um armazém.
    
    Attributes:
        warehouse_id: ID do armazém
        location: Localização (coordenadas ou nome)
        storage_capacity: Capacidade máxima de armazenamento
        transfer_cost_per_unit: Custo de transferência por unidade (para outros armazéns)
    """
    warehouse_id: str
    location: str = ""
    storage_capacity: float = float('inf')
    transfer_cost_per_unit: Dict[str, float] = field(default_factory=dict)  # {target_warehouse: cost}


@dataclass
class OptimizationConfig:
    """
    Configuração para otimização multi-armazém.
    
    Attributes:
        use_milp: Usar MILP (True) ou heurística (False)
        penalty_rupture: Penalização por ruptura (por unidade)
        max_transfers: Número máximo de transferências permitidas
    """
    use_milp: bool = True
    penalty_rupture: float = 1000.0
    max_transfers: int = 10


@dataclass
class MultiWarehousePlan:
    """
    Plano de otimização multi-armazém.
    
    Attributes:
        transfers: Lista de transferências [(from_wh, to_wh, sku, qty)]
        orders: Lista de encomendas [(warehouse_id, sku, qty)]
        total_cost: Custo total do plano
        risk_reduction: Redução de risco de ruptura (%)
    """
    transfers: List[tuple] = field(default_factory=list)  # (from_wh, to_wh, sku, qty)
    orders: List[tuple] = field(default_factory=list)  # (warehouse_id, sku, qty)
    total_cost: float = 0.0
    risk_reduction: float = 0.0
    
    def to_dict(self) -> dict:
        """Converte para dicionário."""
        return {
            "transfers": [
                {"from": t[0], "to": t[1], "sku": t[2], "quantity": t[3]}
                for t in self.transfers
            ],
            "orders": [
                {"warehouse_id": o[0], "sku": o[1], "quantity": o[2]}
                for o in self.orders
            ],
            "total_cost": self.total_cost,
            "risk_reduction": self.risk_reduction,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def optimize_multi_warehouse(
    warehouses: List[Warehouse],
    stock_state: Any,  # StockState
    forecast_results: Dict[str, Any],  # {sku: ForecastResult}
    rop_results: Dict[str, Any],  # {sku: ROPResult}
    config: Optional[OptimizationConfig] = None,
) -> MultiWarehousePlan:
    """
    Otimiza redistribuição e compras multi-armazém.
    
    Args:
        warehouses: Lista de armazéns
        stock_state: Estado atual de stock
        forecast_results: Forecasts por SKU
        rop_results: ROPs por SKU
        config: Configuração
    
    Returns:
        MultiWarehousePlan com transferências e encomendas sugeridas
    """
    config = config or OptimizationConfig()
    
    if config.use_milp:
        try:
            return _optimize_milp(warehouses, stock_state, forecast_results, rop_results, config)
        except Exception as e:
            logger.warning(f"MILP falhou: {e}, usando heurística")
            return _optimize_heuristic(warehouses, stock_state, forecast_results, rop_results, config)
    else:
        return _optimize_heuristic(warehouses, stock_state, forecast_results, rop_results, config)


def _optimize_milp(
    warehouses: List[Warehouse],
    stock_state: Any,
    forecast_results: Dict[str, Any],
    rop_results: Dict[str, Any],
    config: OptimizationConfig,
) -> MultiWarehousePlan:
    """
    Otimização usando MILP (OR-Tools).
    
    TODO[R&D]: Implementar MILP completo com OR-Tools.
    Por agora, retorna heurística.
    """
    logger.warning("MILP não implementado, usando heurística")
    return _optimize_heuristic(warehouses, stock_state, forecast_results, rop_results, config)


def _optimize_heuristic(
    warehouses: List[Warehouse],
    stock_state: Any,
    forecast_results: Dict[str, Any],
    rop_results: Dict[str, Any],
    config: OptimizationConfig,
) -> MultiWarehousePlan:
    """
    Otimização heurística (greedy).
    
    Estratégia:
        1. Identificar armazéns com stock abaixo do ROP
        2. Identificar armazéns com stock excessivo
        3. Transferir stock de excedentes para défices
        4. Encomendar stock para armazéns que ainda estão abaixo do ROP
    """
    transfers = []
    orders = []
    total_cost = 0.0
    
    # Para cada SKU
    for sku in forecast_results.keys():
        if sku not in rop_results:
            continue
        
        rop_result = rop_results[sku]
        rop = rop_result.rop
        
        # Stock por armazém
        warehouse_stocks = {}
        for wh in warehouses:
            ws = stock_state.get_warehouse_stock(sku, wh.warehouse_id)
            if ws:
                warehouse_stocks[wh.warehouse_id] = ws.quantity_available
            else:
                warehouse_stocks[wh.warehouse_id] = 0.0
        
        # Identificar défices e excedentes
        deficits = {}  # {warehouse_id: deficit}
        surpluses = {}  # {warehouse_id: surplus}
        
        for wh_id, stock in warehouse_stocks.items():
            if stock < rop:
                deficits[wh_id] = rop - stock
            elif stock > rop * 1.5:  # Excedente se > 150% do ROP
                surpluses[wh_id] = stock - rop * 1.2
        
        # Transferir de excedentes para défices
        for deficit_wh, deficit_qty in deficits.items():
            for surplus_wh, surplus_qty in list(surpluses.items()):
                if surplus_qty <= 0:
                    continue
                
                transfer_qty = min(deficit_qty, surplus_qty)
                if transfer_qty > 0:
                    transfers.append((surplus_wh, deficit_wh, sku, transfer_qty))
                    deficits[deficit_wh] -= transfer_qty
                    surpluses[surplus_wh] -= transfer_qty
                    total_cost += transfer_qty * 10.0  # Custo simplificado
        
        # Encomendar para défices restantes
        for deficit_wh, deficit_qty in deficits.items():
            if deficit_qty > 0:
                orders.append((deficit_wh, sku, deficit_qty))
                total_cost += deficit_qty * 50.0  # Custo de compra simplificado
    
    return MultiWarehousePlan(
        transfers=transfers,
        orders=orders,
        total_cost=total_cost,
        risk_reduction=10.0,  # Placeholder
    )



