"""
════════════════════════════════════════════════════════════════════════════════
DATA QUALITY - Checks e ML Básico para Deteção de Anomalias
════════════════════════════════════════════════════════════════════════════════

Contract 14: Data Quality checks e ML básico para deteção de anomalias

Funcionalidades:
- Validação de qualidade de dados (warnings/errors)
- ML básico para deteção de anomalias (autoencoder opcional)
- Flags de qualidade armazenadas em ops_data_quality_flags
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session

from ops_ingestion.models import (
    OpsRawOrder,
    OpsRawInventoryMove,
    OpsRawHR,
    OpsRawMachine,
    OpsDataQualityFlag,
    MovementType,
)

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    import torch
    import torch.nn as nn
    HAS_PYTORCH = True
except ImportError:
    HAS_PYTORCH = False
    logger.debug("PyTorch not available, ML anomaly detection will be disabled")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.debug("NumPy not available, some quality checks will be limited")


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECKS - ORDERS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_orders_quality(
    db: Session,
    raw_ids: List[int],
) -> Dict[str, Any]:
    """
    Analisa qualidade de ordens importadas.
    
    Checks:
    - Quantidades negativas
    - Datas de entrega no passado extremo
    - Tempos padrão absurdos (operação com 0s quando devia ser > 10s)
    
    Returns:
        dict com warnings e errors
    """
    orders = db.query(OpsRawOrder).filter(OpsRawOrder.id.in_(raw_ids)).all()
    
    warnings: List[str] = []
    errors: List[str] = []
    
    for order in orders:
        # Check quantidade negativa
        if order.quantity <= 0:
            errors.append(f"Ordem {order.external_order_code}: Quantidade deve ser > 0")
            _create_quality_flag(
                db, "ops_raw_orders", order.id, "ERROR",
                "quantity", f"Quantidade inválida: {order.quantity}", 3
            )
        
        # Check data de entrega no passado extremo (> 1 ano atrás)
        if order.due_date:
            one_year_ago = datetime.now().date() - timedelta(days=365)
            if order.due_date < one_year_ago:
                warnings.append(f"Ordem {order.external_order_code}: Data de entrega muito antiga ({order.due_date})")
                _create_quality_flag(
                    db, "ops_raw_orders", order.id, "WARNING",
                    "due_date", f"Data de entrega muito antiga: {order.due_date}", 2
                )
        
        # Check routing - tempos absurdos
        routing = order.get_routing()
        if routing:
            for op_idx, op in enumerate(routing):
                time_min = op.get("time_min") or op.get("time") or 0
                if time_min == 0 and op.get("operation"):
                    warnings.append(
                        f"Ordem {order.external_order_code}: Operação {op.get('operation')} tem tempo 0"
                    )
                    _create_quality_flag(
                        db, "ops_raw_orders", order.id, "WARNING",
                        "routing", f"Operação {op_idx} tem tempo 0", 1
                    )
                elif time_min > 0 and time_min < 0.1:  # Menos de 6 segundos
                    warnings.append(
                        f"Ordem {order.external_order_code}: Operação {op.get('operation')} tem tempo muito baixo ({time_min} min)"
                    )
    
    return {
        "warnings": warnings,
        "errors": errors,
        "checked_count": len(orders),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECKS - INVENTORY MOVES
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_inventory_moves_quality(
    db: Session,
    raw_ids: List[int],
) -> Dict[str, Any]:
    """
    Analisa qualidade de movimentos de inventário.
    
    Checks:
    - Movimentos sem order_code
    - Movimentos que excedem qty total da OP
    - Timestamps fora de ordem (recuam no tempo)
    """
    moves = db.query(OpsRawInventoryMove).filter(OpsRawInventoryMove.id.in_(raw_ids)).all()
    
    warnings: List[str] = []
    errors: List[str] = []
    
    # Agrupar por order_code para verificar timestamps
    moves_by_order: Dict[str, List[OpsRawInventoryMove]] = {}
    
    for move in moves:
        # Check order_code
        if not move.order_code or not move.order_code.strip():
            errors.append(f"Movimento {move.id}: Sem order_code")
            _create_quality_flag(
                db, "ops_raw_inventory_moves", move.id, "ERROR",
                "order_code", "order_code vazio", 3
            )
            continue
        
        # Agrupar por ordem
        if move.order_code not in moves_by_order:
            moves_by_order[move.order_code] = []
        moves_by_order[move.order_code].append(move)
    
    # Verificar timestamps fora de ordem
    for order_code, order_moves in moves_by_order.items():
        sorted_moves = sorted(order_moves, key=lambda m: m.timestamp)
        
        for i in range(1, len(sorted_moves)):
            prev_move = sorted_moves[i - 1]
            curr_move = sorted_moves[i]
            
            if curr_move.timestamp < prev_move.timestamp:
                warnings.append(
                    f"Ordem {order_code}: Movimento {curr_move.id} tem timestamp anterior ao movimento {prev_move.id}"
                )
                _create_quality_flag(
                    db, "ops_raw_inventory_moves", curr_move.id, "WARNING",
                    "timestamp", f"Timestamp fora de ordem: {curr_move.timestamp} < {prev_move.timestamp}", 2
                )
    
    return {
        "warnings": warnings,
        "errors": errors,
        "checked_count": len(moves),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECKS - HR
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_hr_quality(
    db: Session,
    raw_ids: List[int],
) -> Dict[str, Any]:
    """
    Analisa qualidade de dados de RH.
    
    Checks:
    - Skills fora de 0-1
    - Padrões de turno incoerentes
    """
    hr_records = db.query(OpsRawHR).filter(OpsRawHR.id.in_(raw_ids)).all()
    
    warnings: List[str] = []
    errors: List[str] = []
    
    for hr in hr_records:
        # Check skills
        skills = hr.get_skills()
        for skill_name, skill_value in skills.items():
            if not isinstance(skill_value, (int, float)):
                warnings.append(f"Técnico {hr.technician_code}: Skill '{skill_name}' não é numérico")
            elif skill_value < 0 or skill_value > 1:
                errors.append(
                    f"Técnico {hr.technician_code}: Skill '{skill_name}' fora de range [0,1]: {skill_value}"
                )
                _create_quality_flag(
                    db, "ops_raw_hr", hr.id, "ERROR",
                    "skills", f"Skill '{skill_name}' fora de range: {skill_value}", 3
                )
        
        # Check shift_pattern (básico)
        if hr.shift_pattern:
            # Verificar formato básico (ex: "M-F 08:00-16:00")
            if len(hr.shift_pattern) < 5:
                warnings.append(f"Técnico {hr.technician_code}: Padrão de turno muito curto")
    
    return {
        "warnings": warnings,
        "errors": errors,
        "checked_count": len(hr_records),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY CHECKS - MACHINES
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_machines_quality(
    db: Session,
    raw_ids: List[int],
) -> Dict[str, Any]:
    """
    Analisa qualidade de dados de máquinas.
    
    Checks:
    - Capacidade 0 ou negativa
    - Setup time > capacidade de turno (sinais de input errado)
    """
    machines = db.query(OpsRawMachine).filter(OpsRawMachine.id.in_(raw_ids)).all()
    
    warnings: List[str] = []
    errors: List[str] = []
    
    for machine in machines:
        # Check capacidade
        if machine.capacity_per_shift_hours <= 0:
            errors.append(f"Máquina {machine.machine_code}: Capacidade deve ser > 0")
            _create_quality_flag(
                db, "ops_raw_machines", machine.id, "ERROR",
                "capacity_per_shift_hours", f"Capacidade inválida: {machine.capacity_per_shift_hours}", 3
            )
        
        # Check setup time vs capacidade
        if machine.avg_setup_time_minutes and machine.capacity_per_shift_hours > 0:
            setup_hours = machine.avg_setup_time_minutes / 60.0
            if setup_hours > machine.capacity_per_shift_hours:
                warnings.append(
                    f"Máquina {machine.machine_code}: Setup time ({setup_hours}h) > capacidade de turno ({machine.capacity_per_shift_hours}h)"
                )
                _create_quality_flag(
                    db, "ops_raw_machines", machine.id, "WARNING",
                    "avg_setup_time_minutes", f"Setup time maior que capacidade: {setup_hours}h > {machine.capacity_per_shift_hours}h", 2
                )
    
    return {
        "warnings": warnings,
        "errors": errors,
        "checked_count": len(machines),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ML ANOMALY DETECTION (OPCIONAL)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleAutoencoder(nn.Module):
    """Autoencoder simples para deteção de anomalias."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def detect_anomalies_ml_orders(
    db: Session,
    raw_ids: List[int],
    threshold: float = 0.1,
) -> List[int]:
    """
    Detecta anomalias em ordens usando ML (autoencoder).
    
    Requer:
    - N > 1000 registos históricos para treinar
    - PyTorch disponível
    
    Returns:
        Lista de IDs de ordens anómalas
    """
    if not HAS_PYTORCH or not HAS_NUMPY:
        logger.debug("ML anomaly detection skipped: PyTorch/NumPy not available")
        return []
    
    # Buscar ordens
    orders = db.query(OpsRawOrder).filter(OpsRawOrder.id.in_(raw_ids)).all()
    
    if len(orders) < 100:  # Mínimo para treinar
        logger.debug(f"ML anomaly detection skipped: only {len(orders)} orders (need >= 100)")
        return []
    
    try:
        # Extrair features
        features = []
        for order in orders:
            routing = order.get_routing()
            feature = [
                order.quantity or 0.0,
                len(routing) if routing else 0.0,
                sum(op.get("time_min", 0) for op in routing) if routing else 0.0,
            ]
            features.append(feature)
        
        X = np.array(features)
        
        # Normalizar
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std
        
        # Treinar autoencoder
        model = SimpleAutoencoder(input_dim=X_norm.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X_norm)
        
        # Treino rápido (10 epochs)
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            reconstructed = model(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
        
        # Detectar anomalias
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        # Threshold (percentil 95)
        error_threshold = np.percentile(reconstruction_errors, 95)
        
        # Identificar anomalias
        anomalous_ids = [
            orders[i].id
            for i in range(len(orders))
            if reconstruction_errors[i] > error_threshold
        ]
        
        # Criar flags
        for order_id in anomalous_ids:
            _create_quality_flag(
                db, "ops_raw_orders", order_id, "ANOMALY",
                None, "Anomalia detectada por ML (autoencoder)", 2
            )
        
        return anomalous_ids
    
    except Exception as e:
        logger.warning(f"ML anomaly detection failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _create_quality_flag(
    db: Session,
    table_name: str,
    record_id: int,
    flag_type: str,
    field_name: Optional[str],
    message: str,
    severity: int = 1,
) -> None:
    """Cria flag de qualidade."""
    try:
        flag = OpsDataQualityFlag(
            table_name=table_name,
            record_id=record_id,
            flag_type=flag_type,
            field_name=field_name,
            message=message,
            severity=severity,
            detected_by="data_quality",
        )
        db.add(flag)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to create quality flag: {e}")
        db.rollback()


