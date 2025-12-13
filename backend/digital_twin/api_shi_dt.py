"""
════════════════════════════════════════════════════════════════════════════════════════════════════
SHI-DT API - Smart Health Index Digital Twin REST Endpoints
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints para o serviço SHI-DT:
- GET /shi-dt/machines - Lista máquinas monitorizadas
- GET /shi-dt/machines/{machine_id}/health - Health Index atual
- GET /shi-dt/machines/{machine_id}/rul - RUL estimado
- GET /shi-dt/machines/{machine_id}/status - Status completo
- POST /shi-dt/machines/{machine_id}/ingest - Ingerir dados de sensores
- GET /shi-dt/alerts - Alertas ativos
- GET /shi-dt/metrics - Métricas agregadas

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from .shi_dt import (
    SHIDT,
    get_shidt,
    SHIDTConfig,
    SensorSnapshot,
    OperationContext,
    HealthIndexReading,
    MachineHealthStatus,
    RULEstimate,
    HealthAlert,
    OperationalProfile,
    AlertSeverity,
)
from .health_indicator_cvae import SensorSnapshot as CVAESensorSnapshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/shi-dt", tags=["SHI-DT"])


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SensorDataRequest(BaseModel):
    """Request para ingestão de dados de sensores."""
    timestamp: Optional[str] = Field(None, description="ISO timestamp (usa agora se omitido)")
    
    # Vibration sensors
    vibration_x: float = Field(0.0, ge=0, le=10, description="Vibração eixo X (g)")
    vibration_y: float = Field(0.0, ge=0, le=10, description="Vibração eixo Y (g)")
    vibration_z: float = Field(0.0, ge=0, le=10, description="Vibração eixo Z (g)")
    vibration_rms: Optional[float] = Field(None, description="Vibração RMS (calculado se omitido)")
    
    # Current sensors
    current_phase_a: float = Field(0.0, ge=0, le=100, description="Corrente fase A (A)")
    current_phase_b: float = Field(0.0, ge=0, le=100, description="Corrente fase B (A)")
    current_phase_c: float = Field(0.0, ge=0, le=100, description="Corrente fase C (A)")
    current_rms: Optional[float] = Field(None, description="Corrente RMS (calculado se omitido)")
    
    # Temperature sensors
    temperature_bearing: float = Field(0.0, ge=-50, le=200, description="Temperatura rolamento (°C)")
    temperature_motor: float = Field(0.0, ge=-50, le=200, description="Temperatura motor (°C)")
    temperature_ambient: float = Field(25.0, ge=-50, le=100, description="Temperatura ambiente (°C)")
    
    # Other sensors
    acoustic_emission: float = Field(0.0, ge=0, le=100, description="Emissão acústica (dB)")
    oil_particle_count: float = Field(0.0, ge=0, le=1000, description="Partículas no óleo (ppm)")
    pressure: float = Field(0.0, ge=0, le=500, description="Pressão (bar)")
    
    # Operational parameters
    speed_rpm: float = Field(0.0, ge=0, le=50000, description="Velocidade (RPM)")
    load_percent: float = Field(0.0, ge=0, le=150, description="Carga (%)")
    power_factor: float = Field(0.85, ge=0, le=1, description="Fator de potência")
    
    # Context (optional)
    op_code: Optional[str] = Field(None, description="Código da operação atual")
    product_type: Optional[str] = Field(None, description="Tipo de produto")
    order_id: Optional[str] = Field(None, description="ID da ordem de produção")
    
    class Config:
        json_schema_extra = {
            "example": {
                "vibration_x": 0.15,
                "vibration_y": 0.12,
                "vibration_z": 0.18,
                "current_phase_a": 25.5,
                "current_phase_b": 26.0,
                "current_phase_c": 25.8,
                "temperature_bearing": 45.0,
                "temperature_motor": 52.0,
                "temperature_ambient": 22.0,
                "speed_rpm": 1500.0,
                "load_percent": 65.0,
                "op_code": "OP-MILL",
                "product_type": "PRODUCT-A",
            }
        }


class HealthIndexResponse(BaseModel):
    """Resposta com Health Index."""
    machine_id: str
    timestamp: str
    health_index: float = Field(..., description="Índice de saúde 0-100%")
    health_index_std: float = Field(..., description="Incerteza do HI")
    status: str = Field(..., description="HEALTHY, WARNING, CRITICAL")
    profile: str = Field(..., description="Perfil operacional atual")


class RULResponse(BaseModel):
    """Resposta com RUL estimado."""
    machine_id: str
    timestamp: str
    rul_hours: float = Field(..., description="RUL médio (horas)")
    rul_lower: float = Field(..., description="RUL limite inferior (horas)")
    rul_upper: float = Field(..., description="RUL limite superior (horas)")
    confidence: float = Field(..., description="Nível de confiança 0-1")
    degradation_rate: float = Field(..., description="Taxa de degradação (HI/hora)")
    method: str = Field(..., description="Método de estimação")


class MachineStatusResponse(BaseModel):
    """Resposta com status completo da máquina."""
    machine_id: str
    timestamp: str
    health_index: float
    health_index_std: float
    status: str
    rul: Optional[Dict[str, Any]]
    profile: str
    degradation: Dict[str, Any]
    top_contributors: List[Dict[str, Any]]
    active_alerts: List[Dict[str, Any]]
    last_sensor_update: str
    model_version: str


class AlertResponse(BaseModel):
    """Resposta de alerta."""
    alert_id: str
    machine_id: str
    timestamp: str
    severity: str
    title: str
    message: str
    hi_current: float
    rul: Optional[Dict[str, Any]]
    contributing_sensors: List[Dict[str, Any]]
    recommended_actions: List[str]


class IngestResponse(BaseModel):
    """Resposta da ingestão de dados."""
    success: bool
    machine_id: str
    timestamp: str
    health_index: float
    status: str
    profile: str
    alerts_generated: int


class MetricsResponse(BaseModel):
    """Resposta com métricas agregadas."""
    timestamp: str
    version: str
    machines_monitored: int
    machines: Dict[str, Dict[str, Any]]
    summary: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_sensor_value(value: float, min_val: float, max_val: float) -> float:
    """Normaliza valor do sensor para [0, 1]."""
    if max_val == min_val:
        return 0.5
    return max(0, min(1, (value - min_val) / (max_val - min_val)))


def _create_sensor_snapshot(machine_id: str, data: SensorDataRequest) -> SensorSnapshot:
    """Cria SensorSnapshot a partir do request."""
    # Parse timestamp
    if data.timestamp:
        try:
            ts = datetime.fromisoformat(data.timestamp.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)
    else:
        ts = datetime.now(timezone.utc)
    
    # Calcular RMS se não fornecido
    vib_rms = data.vibration_rms
    if vib_rms is None:
        vib_rms = (data.vibration_x**2 + data.vibration_y**2 + data.vibration_z**2)**0.5
    
    curr_rms = data.current_rms
    if curr_rms is None:
        curr_rms = (data.current_phase_a**2 + data.current_phase_b**2 + data.current_phase_c**2)**0.5 / 1.732
    
    # Normalizar valores para [0, 1]
    return SensorSnapshot(
        machine_id=machine_id,
        timestamp=ts,
        vibration_x=_normalize_sensor_value(data.vibration_x, 0, 2),
        vibration_y=_normalize_sensor_value(data.vibration_y, 0, 2),
        vibration_z=_normalize_sensor_value(data.vibration_z, 0, 2),
        vibration_rms=_normalize_sensor_value(vib_rms, 0, 3.5),
        current_phase_a=_normalize_sensor_value(data.current_phase_a, 0, 50),
        current_phase_b=_normalize_sensor_value(data.current_phase_b, 0, 50),
        current_phase_c=_normalize_sensor_value(data.current_phase_c, 0, 50),
        current_rms=_normalize_sensor_value(curr_rms, 0, 50),
        temperature_bearing=_normalize_sensor_value(data.temperature_bearing, 20, 100),
        temperature_motor=_normalize_sensor_value(data.temperature_motor, 20, 120),
        temperature_ambient=_normalize_sensor_value(data.temperature_ambient, 0, 50),
        acoustic_emission=_normalize_sensor_value(data.acoustic_emission, 0, 80),
        oil_particle_count=_normalize_sensor_value(data.oil_particle_count, 0, 500),
        pressure=_normalize_sensor_value(data.pressure, 0, 200),
        speed_rpm=_normalize_sensor_value(data.speed_rpm, 0, 3000),
        load_percent=data.load_percent / 100.0,
        power_factor=data.power_factor,
    )


def _create_operation_context(machine_id: str, data: SensorDataRequest) -> OperationContext:
    """Cria OperationContext a partir do request."""
    return OperationContext(
        machine_id=machine_id,
        op_code=data.op_code or "UNKNOWN",
        product_type=data.product_type or "UNKNOWN",
        order_id=data.order_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/machines")
async def list_machines() -> Dict[str, Any]:
    """
    Lista todas as máquinas monitorizadas pelo SHI-DT.
    
    Returns:
        Lista de machine_ids com status resumido
    """
    shi_dt = get_shidt()
    all_status = shi_dt.get_all_machines_status()
    
    machines = []
    for machine_id, status in all_status.items():
        machines.append({
            "machine_id": machine_id,
            "health_index": round(status.health_index, 1),
            "status": status.status,
            "profile": status.current_profile.value,
            "rul_hours": status.rul_estimate.rul_hours if status.rul_estimate else None,
            "alerts_count": len(status.active_alerts),
        })
    
    # Ordenar por HI (pior primeiro)
    machines.sort(key=lambda x: x["health_index"])
    
    return {
        "count": len(machines),
        "machines": machines,
    }


@router.get("/machines/{machine_id}/health")
async def get_health_index(machine_id: str) -> HealthIndexResponse:
    """
    Obtém o Health Index atual de uma máquina.
    
    Args:
        machine_id: ID da máquina
    
    Returns:
        HealthIndexResponse com índice de saúde 0-100%
    """
    shi_dt = get_shidt()
    status = shi_dt.get_full_status(machine_id)
    
    return HealthIndexResponse(
        machine_id=machine_id,
        timestamp=status.timestamp.isoformat(),
        health_index=round(status.health_index, 1),
        health_index_std=round(status.health_index_std, 2),
        status=status.status,
        profile=status.current_profile.value,
    )


@router.get("/machines/{machine_id}/rul")
async def get_rul(machine_id: str) -> RULResponse:
    """
    Obtém a Remaining Useful Life estimada de uma máquina.
    
    Args:
        machine_id: ID da máquina
    
    Returns:
        RULResponse com estimativa de vida útil
    """
    shi_dt = get_shidt()
    status = shi_dt.get_full_status(machine_id)
    
    if not status.rul_estimate:
        # Retornar valores default se não há estimativa
        return RULResponse(
            machine_id=machine_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            rul_hours=10000.0,
            rul_lower=5000.0,
            rul_upper=15000.0,
            confidence=0.3,
            degradation_rate=0.0,
            method="no_data",
        )
    
    rul = status.rul_estimate
    return RULResponse(
        machine_id=machine_id,
        timestamp=rul.timestamp.isoformat(),
        rul_hours=round(rul.rul_hours, 1),
        rul_lower=round(rul.rul_lower, 1),
        rul_upper=round(rul.rul_upper, 1),
        confidence=round(rul.confidence, 3),
        degradation_rate=round(rul.degradation_rate, 4),
        method=rul.extrapolation_method,
    )


@router.get("/machines/{machine_id}/status")
async def get_machine_status(machine_id: str) -> MachineStatusResponse:
    """
    Obtém o status completo de uma máquina.
    
    Inclui HI, RUL, alertas, contribuição de sensores, etc.
    
    Args:
        machine_id: ID da máquina
    
    Returns:
        MachineStatusResponse com estado completo
    """
    shi_dt = get_shidt()
    status = shi_dt.get_full_status(machine_id)
    
    return MachineStatusResponse(
        machine_id=status.machine_id,
        timestamp=status.timestamp.isoformat(),
        health_index=round(status.health_index, 1),
        health_index_std=round(status.health_index_std, 2),
        status=status.status,
        rul=status.rul_estimate.to_dict() if status.rul_estimate else None,
        profile=status.current_profile.value,
        degradation={
            "trend": status.degradation_trend,
            "rate_per_hour": round(status.degradation_rate, 4),
        },
        top_contributors=[c.to_dict() for c in status.top_contributors],
        active_alerts=[a.to_dict() for a in status.active_alerts],
        last_sensor_update=status.last_sensor_update.isoformat(),
        model_version=status.model_version,
    )


@router.post("/machines/{machine_id}/ingest")
async def ingest_sensor_data(
    machine_id: str,
    data: SensorDataRequest = Body(...),
) -> IngestResponse:
    """
    Ingere dados de sensores de uma máquina.
    
    Este endpoint recebe leituras de sensores em tempo real e atualiza
    o estado de saúde da máquina.
    
    Args:
        machine_id: ID da máquina
        data: Leituras dos sensores
    
    Returns:
        IngestResponse com resultado do processamento
    """
    shi_dt = get_shidt()
    
    try:
        # Criar objetos de dados
        snapshot = _create_sensor_snapshot(machine_id, data)
        context = _create_operation_context(machine_id, data)
        
        # Ingerir dados
        reading = shi_dt.ingest_sensor_data(machine_id, snapshot, context)
        
        # Obter status atualizado
        status = shi_dt.get_full_status(machine_id)
        
        return IngestResponse(
            success=True,
            machine_id=machine_id,
            timestamp=reading.timestamp.isoformat(),
            health_index=round(reading.hi_smoothed, 1),
            status=status.status,
            profile=reading.profile.value,
            alerts_generated=len(status.active_alerts),
        )
        
    except Exception as e:
        logger.error(f"Error ingesting sensor data for {machine_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filtrar por severidade"),
    machine_id: Optional[str] = Query(None, description="Filtrar por máquina"),
    acknowledged: Optional[bool] = Query(None, description="Filtrar por reconhecido"),
) -> Dict[str, Any]:
    """
    Lista alertas ativos do sistema.
    
    Args:
        severity: Filtrar por severidade (info, warning, critical, emergency)
        machine_id: Filtrar por máquina
        acknowledged: Filtrar por estado de reconhecimento
    
    Returns:
        Lista de alertas ativos
    """
    shi_dt = get_shidt()
    all_status = shi_dt.get_all_machines_status()
    
    alerts = []
    for status in all_status.values():
        for alert in status.active_alerts:
            # Aplicar filtros
            if severity and alert.severity.value != severity:
                continue
            if machine_id and alert.machine_id != machine_id:
                continue
            if acknowledged is not None and alert.acknowledged != acknowledged:
                continue
            
            alerts.append(alert.to_dict())
    
    # Ordenar por severidade e timestamp
    severity_order = {"emergency": 0, "critical": 1, "warning": 2, "info": 3}
    alerts.sort(key=lambda x: (severity_order.get(x["severity"], 4), x["timestamp"]))
    
    return {
        "count": len(alerts),
        "alerts": alerts,
    }


@router.get("/metrics")
async def get_metrics() -> MetricsResponse:
    """
    Obtém métricas agregadas do sistema SHI-DT.
    
    Útil para dashboards e monitorização global.
    
    Returns:
        MetricsResponse com métricas de todas as máquinas
    """
    shi_dt = get_shidt()
    metrics = shi_dt.export_metrics()
    
    # Calcular sumário
    machines = metrics.get("machines", {})
    total = len(machines)
    
    healthy = sum(1 for m in machines.values() if m.get("status") == "HEALTHY")
    warning = sum(1 for m in machines.values() if m.get("status") == "WARNING")
    critical = sum(1 for m in machines.values() if m.get("status") == "CRITICAL")
    
    avg_hi = sum(m.get("health_index", 100) for m in machines.values()) / max(1, total)
    
    total_alerts = sum(m.get("alerts_count", 0) for m in machines.values())
    
    summary = {
        "healthy_count": healthy,
        "warning_count": warning,
        "critical_count": critical,
        "average_health_index": round(avg_hi, 1),
        "total_alerts": total_alerts,
    }
    
    return MetricsResponse(
        timestamp=metrics.get("timestamp", datetime.now(timezone.utc).isoformat()),
        version=metrics.get("version", "1.0.0"),
        machines_monitored=total,
        machines=machines,
        summary=summary,
    )


@router.post("/demo/generate-data")
async def generate_demo_data(
    num_machines: int = Query(5, ge=1, le=50, description="Número de máquinas"),
    readings_per_machine: int = Query(10, ge=1, le=100, description="Leituras por máquina"),
) -> Dict[str, Any]:
    """
    Gera dados de demonstração para teste do sistema.
    
    Cria leituras simuladas de sensores para múltiplas máquinas.
    
    Args:
        num_machines: Número de máquinas a simular
        readings_per_machine: Leituras por máquina
    
    Returns:
        Resumo dos dados gerados
    """
    import numpy as np
    
    shi_dt = get_shidt()
    
    machines_created = []
    
    for i in range(num_machines):
        machine_id = f"DEMO-M-{i+1:03d}"
        
        # Simular degradação progressiva
        base_degradation = np.random.uniform(0, 0.5)
        
        for j in range(readings_per_machine):
            # Aumentar degradação ao longo do tempo
            degradation = base_degradation + j * 0.02
            degradation = min(degradation, 0.9)
            
            # Gerar leituras baseadas na degradação
            snapshot = SensorSnapshot(
                machine_id=machine_id,
                timestamp=datetime.now(timezone.utc),
                vibration_x=0.1 + degradation * 0.5 + np.random.normal(0, 0.02),
                vibration_y=0.1 + degradation * 0.5 + np.random.normal(0, 0.02),
                vibration_z=0.1 + degradation * 0.5 + np.random.normal(0, 0.02),
                vibration_rms=0.17 + degradation * 0.8 + np.random.normal(0, 0.03),
                current_phase_a=0.3 + degradation * 0.3 + np.random.normal(0, 0.02),
                current_phase_b=0.3 + degradation * 0.3 + np.random.normal(0, 0.02),
                current_phase_c=0.3 + degradation * 0.3 + np.random.normal(0, 0.02),
                current_rms=0.52 + degradation * 0.4 + np.random.normal(0, 0.03),
                temperature_bearing=0.25 + degradation * 0.4 + np.random.normal(0, 0.02),
                temperature_motor=0.20 + degradation * 0.5 + np.random.normal(0, 0.02),
                temperature_ambient=0.25 + np.random.normal(0, 0.01),
                acoustic_emission=0.15 + degradation * 0.5 + np.random.normal(0, 0.03),
                oil_particle_count=0.10 + degradation * 0.4 + np.random.normal(0, 0.02),
                pressure=0.50 + np.random.normal(0, 0.03),
                speed_rpm=0.70 + np.random.normal(0, 0.05),
                load_percent=0.50 + np.random.normal(0, 0.1),
                power_factor=0.85 - degradation * 0.1 + np.random.normal(0, 0.01),
            )
            
            context = OperationContext(
                machine_id=machine_id,
                op_code=np.random.choice(["OP-MILL", "OP-CUT", "OP-DRILL"]),
                product_type=np.random.choice(["PROD-A", "PROD-B", "PROD-C"]),
            )
            
            shi_dt.ingest_sensor_data(machine_id, snapshot, context)
        
        # Obter status final
        status = shi_dt.get_full_status(machine_id)
        machines_created.append({
            "machine_id": machine_id,
            "health_index": round(status.health_index, 1),
            "status": status.status,
            "rul_hours": round(status.rul_estimate.rul_hours, 0) if status.rul_estimate else None,
        })
    
    return {
        "success": True,
        "machines_created": num_machines,
        "readings_per_machine": readings_per_machine,
        "total_readings": num_machines * readings_per_machine,
        "machines": machines_created,
    }


@router.get("/status")
async def get_system_status() -> Dict[str, Any]:
    """
    Obtém status do sistema SHI-DT.
    
    Returns:
        Informações sobre o estado do serviço
    """
    shi_dt = get_shidt()
    
    return {
        "service": "SHI-DT",
        "version": SHIDT.VERSION,
        "status": "operational",
        "initialized": shi_dt._initialized,
        "machines_monitored": len(shi_dt._machine_states),
        "pytorch_available": True,  # Já verificado na inicialização
        "features": {
            "cvae_health_indicator": True,
            "rul_estimation": True,
            "profile_detection": shi_dt.config.profile_detection_enabled,
            "online_learning": shi_dt.config.online_learning_enabled,
            "explainability": True,
        },
        "config": {
            "threshold_healthy": shi_dt.config.threshold_healthy,
            "threshold_warning": shi_dt.config.threshold_warning,
            "threshold_critical": shi_dt.config.threshold_critical,
            "rul_method": shi_dt.config.rul_extrapolation_method,
        },
    }



