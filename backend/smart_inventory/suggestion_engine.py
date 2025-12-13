"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    SUGGESTION ENGINE (Sugestões Inteligentes)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Este módulo gera sugestões inteligentes baseadas em:
- ROP e risco de ruptura
- Forecast de demanda
- Sinais externos (preços, notícias)
- Otimização multi-armazém

Tipos de Sugestões:
───────────────────
    - Comprar: Stock abaixo do ROP
    - Transferir: Redistribuição entre armazéns
    - Reduzir: Stock excessivo
    - Antecipar: Preço em alta ou notícias negativas
    - Alerta: Risco de ruptura elevado
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class SuggestionType(str, Enum):
    """Tipos de sugestões."""
    BUY = "BUY"
    TRANSFER = "TRANSFER"
    REDUCE = "REDUCE"
    ANTICIPATE = "ANTICIPATE"
    ALERT = "ALERT"


class SuggestionPriority(str, Enum):
    """Prioridade da sugestão."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class InventorySuggestion:
    """
    Sugestão de inventário.
    
    Attributes:
        suggestion_type: Tipo de sugestão
        priority: Prioridade
        sku: SKU afetado
        warehouse_id: Armazém (se aplicável)
        title: Título da sugestão (PT-PT)
        description: Descrição detalhada (PT-PT)
        quantity: Quantidade sugerida
        due_date: Data limite (se aplicável)
        risk_level: Nível de risco associado
        explanation: Explicação técnica (PT-PT)
        metadata: Dados adicionais
    """
    suggestion_type: SuggestionType
    priority: SuggestionPriority
    sku: str
    warehouse_id: Optional[str] = None
    title: str = ""
    description: str = ""
    quantity: float = 0.0
    due_date: Optional[datetime] = None
    risk_level: float = 0.0
    explanation: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Converte para dicionário."""
        return {
            "suggestion_type": self.suggestion_type.value,
            "priority": self.priority.value,
            "sku": self.sku,
            "warehouse_id": self.warehouse_id,
            "title": self.title,
            "description": self.description,
            "quantity": self.quantity,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "risk_level": self.risk_level,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SUGGESTION GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_inventory_suggestions(
    stock_state: Any,  # StockState
    rop_results: Dict[str, Any],  # {sku: ROPResult}
    forecast_results: Dict[str, Any],  # {sku: ForecastResult}
    external_signals: Optional[Dict[str, Any]] = None,
    multi_warehouse_plan: Optional[Any] = None,  # MultiWarehousePlan
) -> List[InventorySuggestion]:
    """
    Gera sugestões inteligentes de inventário.
    
    Args:
        stock_state: Estado atual de stock
        rop_results: Resultados de ROP por SKU
        forecast_results: Forecasts por SKU
        external_signals: Sinais externos (opcional)
        multi_warehouse_plan: Plano de otimização (opcional)
    
    Returns:
        Lista de sugestões ordenadas por prioridade
    """
    suggestions = []
    
    # 1. Sugestões de compra (stock abaixo do ROP)
    for sku, rop_result in rop_results.items():
        # Obter stock atual do Digital Twin se não fornecido
        current_stock = rop_result.current_stock
        if current_stock is None:
            # Tentar obter do stock_state
            try:
                from stock_state import get_realtime_stock
                # Tentar primeiro armazém disponível
                warehouses = stock_state.get_all_warehouses()
                if warehouses:
                    current_stock = get_realtime_stock(stock_state, sku, list(warehouses)[0])
                else:
                    current_stock = 0.0
            except:
                current_stock = 0.0
        
        if current_stock < rop_result.rop:
            deficit = rop_result.rop - rop_result.current_stock
            
            # Prioridade baseada no risco
            if rop_result.risk_30d > 50:
                priority = SuggestionPriority.HIGH
            elif rop_result.risk_30d > 20:
                priority = SuggestionPriority.MEDIUM
            else:
                priority = SuggestionPriority.LOW
            
            suggestion = InventorySuggestion(
                suggestion_type=SuggestionType.BUY,
                priority=priority,
                sku=sku,
                title=f"Comprar {rop_result.reorder_quantity:.0f} unidades de {sku}",
                description=(
                    f"Stock atual ({current_stock:.0f}) está abaixo do ROP ({rop_result.rop:.0f}). "
                    f"Risco de ruptura: {rop_result.risk_30d:.1f}% nos próximos 30 dias."
                ),
                quantity=rop_result.reorder_quantity,
                due_date=datetime.now() + timedelta(days=rop_result.days_until_rop or 0),
                risk_level=rop_result.risk_30d,
                explanation=rop_result.explanation,
            )
            suggestions.append(suggestion)
    
    # 2. Alertas de risco elevado
    for sku, rop_result in rop_results.items():
        if rop_result.risk_30d > 80:
            # Obter stock atual
            current_stock_alert = rop_result.current_stock
            if current_stock_alert is None:
                try:
                    from stock_state import get_realtime_stock
                    warehouses = stock_state.get_all_warehouses()
                    if warehouses:
                        current_stock_alert = get_realtime_stock(stock_state, sku, list(warehouses)[0])
                    else:
                        current_stock_alert = 0.0
                except:
                    current_stock_alert = 0.0
            
            suggestion = InventorySuggestion(
                suggestion_type=SuggestionType.ALERT,
                priority=SuggestionPriority.HIGH,
                sku=sku,
                title=f"⚠️ Alerta: Risco de ruptura crítico para {sku}",
                description=(
                    f"Risco de ruptura muito elevado: {rop_result.risk_30d:.1f}% nos próximos 30 dias. "
                    f"Ação imediata recomendada."
                ),
                risk_level=rop_result.risk_30d,
                explanation=f"Stock atual: {current_stock_alert:.0f}, ROP: {rop_result.rop:.0f}",
            )
            suggestions.append(suggestion)
    
    # 3. Sugestões de transferência (do plano multi-armazém)
    if multi_warehouse_plan:
        for transfer in multi_warehouse_plan.transfers:
            from_wh, to_wh, sku, qty = transfer
            suggestion = InventorySuggestion(
                suggestion_type=SuggestionType.TRANSFER,
                priority=SuggestionPriority.MEDIUM,
                sku=sku,
                warehouse_id=to_wh,
                title=f"Transferir {qty:.0f} unidades de {sku} de {from_wh} para {to_wh}",
                description=(
                    f"Redistribuição recomendada para reduzir risco de ruptura em {to_wh} "
                    f"e otimizar utilização de stock."
                ),
                quantity=qty,
                metadata={"from_warehouse": from_wh, "to_warehouse": to_wh},
            )
            suggestions.append(suggestion)
    
    # 4. Sugestões de redução (stock excessivo)
    for sku, rop_result in rop_results.items():
        # Obter stock atual
        current_stock_reduce = rop_result.current_stock
        if current_stock_reduce is None:
            try:
                from stock_state import get_realtime_stock
                warehouses = stock_state.get_all_warehouses()
                if warehouses:
                    current_stock_reduce = get_realtime_stock(stock_state, sku, list(warehouses)[0])
                else:
                    current_stock_reduce = 0.0
            except:
                current_stock_reduce = 0.0
        
        if current_stock_reduce > rop_result.rop * 2:
            excess = current_stock_reduce - rop_result.rop * 1.5
            suggestion = InventorySuggestion(
                suggestion_type=SuggestionType.REDUCE,
                priority=SuggestionPriority.LOW,
                sku=sku,
                title=f"Reduzir stock de {sku}",
                description=(
                    f"Stock atual ({current_stock_reduce:.0f}) está significativamente acima do ROP. "
                    f"Considerar reduzir encomendas futuras."
                ),
                quantity=excess,
                explanation=f"Stock excessivo: {excess:.0f} unidades acima do recomendado",
            )
            suggestions.append(suggestion)
    
    # 5. Sugestões de antecipação (sinais externos)
    if external_signals:
        # TODO: Analisar sinais externos e gerar sugestões
        pass
    
    # 5. Se não há sugestões, gerar sugestões genéricas baseadas em ROP
    if len(suggestions) == 0 and len(rop_results) > 0:
        # Gerar sugestões informativas para todos os SKUs
        for sku, rop_result in list(rop_results.items())[:5]:  # Limitar a 5
            # Obter stock atual
            current_stock_info = rop_result.current_stock
            if current_stock_info is None:
                try:
                    from stock_state import get_realtime_stock
                    warehouses = stock_state.get_all_warehouses()
                    if warehouses:
                        current_stock_info = get_realtime_stock(stock_state, sku, list(warehouses)[0])
                    else:
                        current_stock_info = rop_result.rop * 1.2  # Assumir stock acima do ROP
                except:
                    current_stock_info = rop_result.rop * 1.2
            
            # Sugestão baseada na relação stock/ROP
            if current_stock_info < rop_result.rop * 0.8:
                # Stock baixo
                suggestions.append(InventorySuggestion(
                    suggestion_type=SuggestionType.BUY,
                    priority=SuggestionPriority.MEDIUM,
                    sku=sku,
                    title=f"Monitorizar stock de {sku}",
                    description=(
                        f"Stock atual ({current_stock_info:.0f}) está próximo do ROP ({rop_result.rop:.0f}). "
                        f"Considerar encomendar {rop_result.reorder_quantity:.0f} unidades."
                    ),
                    quantity=rop_result.reorder_quantity,
                    risk_level=rop_result.risk_30d,
                    explanation=f"ROP: {rop_result.rop:.0f}, Safety Stock: {rop_result.safety_stock:.0f}",
                ))
            elif current_stock_info > rop_result.rop * 2.5:
                # Stock excessivo
                suggestions.append(InventorySuggestion(
                    suggestion_type=SuggestionType.REDUCE,
                    priority=SuggestionPriority.LOW,
                    sku=sku,
                    title=f"Stock elevado para {sku}",
                    description=(
                        f"Stock atual ({current_stock_info:.0f}) está significativamente acima do ROP. "
                        f"Considerar reduzir encomendas futuras."
                    ),
                    quantity=current_stock_info - rop_result.rop * 1.5,
                    risk_level=5.0,
                    explanation=f"Stock excessivo: {current_stock_info - rop_result.rop * 1.5:.0f} unidades acima do recomendado",
                ))
            else:
                # Stock normal - sugestão informativa
                suggestions.append(InventorySuggestion(
                    suggestion_type=SuggestionType.ALERT,
                    priority=SuggestionPriority.LOW,
                    sku=sku,
                    title=f"Stock normal para {sku}",
                    description=(
                        f"Stock atual ({current_stock_info:.0f}) está dentro dos parâmetros normais. "
                        f"ROP: {rop_result.rop:.0f}, Risco: {rop_result.risk_30d:.1f}%."
                    ),
                    risk_level=rop_result.risk_30d,
                    explanation=f"Cobertura: {rop_result.coverage_days:.0f} dias",
                ))
    
    # Ordenar por prioridade (HIGH primeiro)
    priority_order = {SuggestionPriority.HIGH: 0, SuggestionPriority.MEDIUM: 1, SuggestionPriority.LOW: 2}
    suggestions.sort(key=lambda s: (priority_order[s.priority], -s.risk_level))
    
    return suggestions

