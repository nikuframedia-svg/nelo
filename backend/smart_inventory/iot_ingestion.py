"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    IoT INGESTION (RFID / Vision / Manual)
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Este módulo processa eventos de stock provenientes de múltiplas fontes:
- RFID: Leitura automática de tags
- Vision: Sistemas de visão computacional (câmaras + ML)
- Manual: Entrada manual por operadores
- Scan: Leituras de códigos de barras/QR

Cada evento é normalizado para um formato comum (StockEvent) que pode ser
aplicado ao Digital Twin de stock.

TODO[R&D]: Future enhancements:
    - Real-time video stream processing
    - Edge device integration (Raspberry Pi, Jetson)
    - Anomaly detection on ingestion (spikes, negative stock)
    - Blockchain-based audit trail for critical events
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class StockEventSource(str, Enum):
    """Fonte do evento de stock."""
    RFID = "RFID"
    VISION = "VISION"
    SCAN = "SCAN"
    MANUAL = "MANUAL"
    API = "API"  # Integração com sistemas externos
    ERP = "ERP"  # Sincronização com ERP


@dataclass
class StockEvent:
    """
    Evento de mudança de stock.
    
    Representa uma transação de inventário que altera a quantidade
    de um SKU num armazém específico.
    
    Attributes:
        timestamp: Momento do evento (UTC)
        warehouse_id: Identificador do armazém
        location_id: Localização específica (prateleira, zona, etc.)
        sku: Stock Keeping Unit (identificador do produto)
        quantity_change: Mudança de quantidade (+ entrada, - saída)
        source: Fonte do evento (RFID, VISION, etc.)
        operator_id: ID do operador (se aplicável)
        order_id: ID da encomenda associada (se aplicável)
        metadata: Dados adicionais (JSON)
    """
    timestamp: datetime
    warehouse_id: str
    location_id: Optional[str]
    sku: str
    quantity_change: float
    source: StockEventSource
    operator_id: Optional[str] = None
    order_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validação pós-inicialização."""
        if self.metadata is None:
            self.metadata = {}
        
        # Validar quantidade
        if self.quantity_change == 0:
            logger.warning(f"StockEvent com quantity_change=0 para SKU {self.sku}")
        
        # Validar timestamp
        if self.timestamp.tzinfo is None:
            logger.warning(f"StockEvent sem timezone, assumindo UTC")
            from datetime import timezone
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário (serialização)."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "warehouse_id": self.warehouse_id,
            "location_id": self.location_id,
            "sku": self.sku,
            "quantity_change": self.quantity_change,
            "source": self.source.value,
            "operator_id": self.operator_id,
            "order_id": self.order_id,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> StockEvent:
        """Cria StockEvent a partir de dicionário."""
        from datetime import datetime
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            timestamp=timestamp,
            warehouse_id=data["warehouse_id"],
            location_id=data.get("location_id"),
            sku=data["sku"],
            quantity_change=float(data["quantity_change"]),
            source=StockEventSource(data["source"]),
            operator_id=data.get("operator_id"),
            order_id=data.get("order_id"),
            metadata=data.get("metadata", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_rfid_event(payload: Dict[str, Any]) -> StockEvent:
    """
    Processa evento de RFID.
    
    Formato esperado do payload:
        {
            "timestamp": "2025-12-05T10:30:00Z",
            "warehouse_id": "WH-001",
            "location_id": "A-12-3",
            "sku": "SKU-12345",
            "quantity": 10,
            "direction": "IN" | "OUT",
            "tag_id": "RFID-ABC123",
            "reader_id": "READER-01",
        }
    
    Args:
        payload: Dados do evento RFID
    
    Returns:
        StockEvent normalizado
    """
    from datetime import datetime, timezone
    
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    quantity = float(payload.get("quantity", 0))
    direction = payload.get("direction", "IN").upper()
    
    # Converter direção em mudança de quantidade
    if direction == "OUT":
        quantity_change = -abs(quantity)
    else:
        quantity_change = abs(quantity)
    
    event = StockEvent(
        timestamp=timestamp,
        warehouse_id=str(payload["warehouse_id"]),
        location_id=payload.get("location_id"),
        sku=str(payload["sku"]),
        quantity_change=quantity_change,
        source=StockEventSource.RFID,
        metadata={
            "tag_id": payload.get("tag_id"),
            "reader_id": payload.get("reader_id"),
            "raw_payload": payload,
        },
    )
    
    logger.debug(f"Ingested RFID event: {event.sku} {quantity_change:+g} at {event.warehouse_id}")
    
    return event


def ingest_vision_event(payload: Dict[str, Any]) -> StockEvent:
    """
    Processa evento de sistema de visão computacional.
    
    Formato esperado do payload:
        {
            "timestamp": "2025-12-05T10:30:00Z",
            "warehouse_id": "WH-001",
            "location_id": "ZONE-B",
            "sku": "SKU-12345",
            "detected_quantity": 15,
            "previous_quantity": 10,
            "camera_id": "CAM-01",
            "confidence": 0.95,
            "model_version": "v2.1",
        }
    
    Args:
        payload: Dados do evento de visão
    
    Returns:
        StockEvent normalizado
    """
    from datetime import datetime, timezone
    
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    detected_qty = float(payload.get("detected_quantity", 0))
    previous_qty = float(payload.get("previous_quantity", 0))
    quantity_change = detected_qty - previous_qty
    
    event = StockEvent(
        timestamp=timestamp,
        warehouse_id=str(payload["warehouse_id"]),
        location_id=payload.get("location_id"),
        sku=str(payload["sku"]),
        quantity_change=quantity_change,
        source=StockEventSource.VISION,
        metadata={
            "detected_quantity": detected_qty,
            "previous_quantity": previous_qty,
            "camera_id": payload.get("camera_id"),
            "confidence": payload.get("confidence"),
            "model_version": payload.get("model_version"),
            "raw_payload": payload,
        },
    )
    
    logger.debug(f"Ingested Vision event: {event.sku} {quantity_change:+g} at {event.warehouse_id} (confidence: {payload.get('confidence', 0):.2f})")
    
    return event


def ingest_scan_event(payload: Dict[str, Any]) -> StockEvent:
    """
    Processa evento de leitura de código de barras/QR.
    
    Formato esperado do payload:
        {
            "timestamp": "2025-12-05T10:30:00Z",
            "warehouse_id": "WH-001",
            "sku": "SKU-12345",
            "quantity": 5,
            "scan_type": "IN" | "OUT",
            "scanner_id": "SCAN-01",
        }
    
    Args:
        payload: Dados do evento de scan
    
    Returns:
        StockEvent normalizado
    """
    from datetime import datetime, timezone
    
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    quantity = float(payload.get("quantity", 0))
    scan_type = payload.get("scan_type", "IN").upper()
    
    if scan_type == "OUT":
        quantity_change = -abs(quantity)
    else:
        quantity_change = abs(quantity)
    
    event = StockEvent(
        timestamp=timestamp,
        warehouse_id=str(payload["warehouse_id"]),
        location_id=payload.get("location_id"),
        sku=str(payload["sku"]),
        quantity_change=quantity_change,
        source=StockEventSource.SCAN,
        operator_id=payload.get("operator_id"),
        order_id=payload.get("order_id"),
        metadata={
            "scanner_id": payload.get("scanner_id"),
            "raw_payload": payload,
        },
    )
    
    logger.debug(f"Ingested Scan event: {event.sku} {quantity_change:+g} at {event.warehouse_id}")
    
    return event


def ingest_manual_event(
    warehouse_id: str,
    sku: str,
    quantity_change: float,
    location_id: Optional[str] = None,
    operator_id: Optional[str] = None,
    order_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
) -> StockEvent:
    """
    Cria evento manual (entrada por operador).
    
    Args:
        warehouse_id: ID do armazém
        sku: SKU do produto
        quantity_change: Mudança de quantidade (+ entrada, - saída)
        location_id: Localização (opcional)
        operator_id: ID do operador
        order_id: ID da encomenda (opcional)
        timestamp: Timestamp do evento (default: agora)
    
    Returns:
        StockEvent normalizado
    """
    from datetime import timezone
    
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    event = StockEvent(
        timestamp=timestamp,
        warehouse_id=warehouse_id,
        location_id=location_id,
        sku=sku,
        quantity_change=quantity_change,
        source=StockEventSource.MANUAL,
        operator_id=operator_id,
        order_id=order_id,
        metadata={
            "manual_entry": True,
        },
    )
    
    logger.info(f"Manual stock event: {sku} {quantity_change:+g} at {warehouse_id} by {operator_id or 'unknown'}")
    
    return event


def apply_stock_event(stock_state: Any, event: StockEvent) -> Any:
    """
    Aplica um evento de stock ao estado atual.
    
    Esta função é um wrapper que delega para o método do StockState.
    Ver stock_state.py para implementação completa.
    
    Args:
        stock_state: Estado atual de stock (StockState)
        event: Evento a aplicar
    
    Returns:
        Novo estado de stock (imutável)
    """
    # Esta função será implementada em stock_state.py
    # Por agora, apenas validação
    if event.quantity_change == 0:
        logger.warning(f"Evento com quantity_change=0 ignorado: {event.sku}")
        return stock_state
    
    # Delegar para StockState.apply_event()
    return stock_state.apply_event(event)


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def validate_event(event: StockEvent) -> tuple[bool, Optional[str]]:
    """
    Valida um evento de stock.
    
    Returns:
        (is_valid, error_message)
    """
    if not event.warehouse_id:
        return False, "warehouse_id é obrigatório"
    
    if not event.sku:
        return False, "sku é obrigatório"
    
    if event.quantity_change == 0:
        return False, "quantity_change não pode ser zero"
    
    if not isinstance(event.timestamp, datetime):
        return False, "timestamp inválido"
    
    return True, None


def batch_ingest_events(payloads: list[Dict[str, Any]], source: StockEventSource) -> list[StockEvent]:
    """
    Processa múltiplos eventos em batch.
    
    Args:
        payloads: Lista de payloads de eventos
        source: Fonte dos eventos
    
    Returns:
        Lista de StockEvents normalizados
    """
    events = []
    
    for payload in payloads:
        try:
            if source == StockEventSource.RFID:
                event = ingest_rfid_event(payload)
            elif source == StockEventSource.VISION:
                event = ingest_vision_event(payload)
            elif source == StockEventSource.SCAN:
                event = ingest_scan_event(payload)
            else:
                logger.warning(f"Batch ingest não suportado para source={source}")
                continue
            
            is_valid, error = validate_event(event)
            if is_valid:
                events.append(event)
            else:
                logger.warning(f"Evento inválido ignorado: {error}")
        
        except Exception as e:
            logger.error(f"Erro ao processar evento: {e}")
            continue
    
    logger.info(f"Batch ingest: {len(events)}/{len(payloads)} eventos processados")
    
    return events



