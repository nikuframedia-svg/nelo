"""
════════════════════════════════════════════════════════════════════════════════════════════════════
API IoT - Endpoints para Ingestão de Dados de Sensores
════════════════════════════════════════════════════════════════════════════════════════════════════

Endpoints para receber dados de sensores de máquinas.
Usado por integrações IoT (OPC-UA, MQTT bridges) e importações manuais.

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from digital_twin.iot_ingestion import (
    SensorReading,
    SensorType,
    DataSource,
    IoTIngestionService,
    parse_opc_ua_reading,
    parse_mqtt_reading,
    generate_demo_readings,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/digital-twin/iot", tags=["Digital Twin - IoT"])

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SensorReadingInput(BaseModel):
    """Input model for sensor reading."""
    machine_id: str = Field(..., description="Machine identifier")
    timestamp: Optional[datetime] = Field(None, description="Reading timestamp (UTC)")
    sensor_type: str = Field(..., description="Type of sensor (vibration, temperature, etc.)")
    sensor_id: Optional[str] = Field(None, description="Specific sensor identifier")
    value: float = Field(..., description="Sensor value")
    unit: str = Field("", description="Unit of measurement")
    source: str = Field("API", description="Data source")
    quality: float = Field(100.0, ge=0, le=100, description="Reading quality 0-100")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_schema_extra = {
            "example": {
                "machine_id": "MACH-001",
                "timestamp": "2025-12-11T10:30:00Z",
                "sensor_type": "vibration",
                "sensor_id": "MACH-001_vib_x",
                "value": 0.45,
                "unit": "mm/s",
                "source": "OPC_UA",
                "quality": 100.0,
            }
        }


class BatchReadingsInput(BaseModel):
    """Input model for batch sensor readings."""
    readings: List[SensorReadingInput] = Field(..., description="List of sensor readings")
    
    class Config:
        json_schema_extra = {
            "example": {
                "readings": [
                    {"machine_id": "MACH-001", "sensor_type": "vibration", "value": 0.45, "unit": "mm/s"},
                    {"machine_id": "MACH-001", "sensor_type": "temperature", "value": 55.2, "unit": "°C"},
                ]
            }
        }


class OpcUaReadingInput(BaseModel):
    """Input model for OPC-UA formatted reading."""
    nodeId: str = Field(..., description="OPC-UA node ID")
    value: float = Field(..., description="Value")
    quality: str = Field("Good", description="Quality status")
    timestamp: Optional[datetime] = None
    unit: str = Field("", description="Unit")


class MqttReadingInput(BaseModel):
    """Input model for MQTT formatted reading."""
    topic: str = Field(..., description="MQTT topic (e.g., sensors/machine1/vibration)")
    value: float = Field(..., description="Value")
    timestamp: Optional[datetime] = None
    unit: str = Field("", description="Unit")
    quality: float = Field(100.0)


class SensorReadingOutput(BaseModel):
    """Output model for sensor reading."""
    machine_id: str
    timestamp: datetime
    sensor_type: str
    sensor_id: Optional[str]
    value: float
    unit: str
    source: str
    quality: float


class IngestionStatsOutput(BaseModel):
    """Output model for ingestion statistics."""
    total_ingested: int
    total_persisted: int
    total_errors: int
    buffer_size: int
    by_machine: Dict[str, int]
    by_sensor_type: Dict[str, int]


class DemoGenerateInput(BaseModel):
    """Input for generating demo data."""
    machine_id: str = Field(..., description="Machine identifier")
    duration_hours: float = Field(24, ge=1, le=168, description="Duration in hours")
    interval_seconds: float = Field(60, ge=10, le=3600, description="Interval between readings")
    degradation_factor: float = Field(0.01, ge=0, le=1, description="Degradation factor")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL SERVICE INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_iot_service: Optional[IoTIngestionService] = None


def get_iot_service() -> IoTIngestionService:
    """Get or create the IoT ingestion service."""
    global _iot_service
    if _iot_service is None:
        _iot_service = IoTIngestionService(buffer_size=1000, auto_flush=True)
    return _iot_service


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/readings", summary="Ingest sensor readings")
async def ingest_readings(
    body: BatchReadingsInput,
    service: IoTIngestionService = Depends(get_iot_service),
) -> Dict[str, Any]:
    """
    Ingest multiple sensor readings.
    
    Use this endpoint to send sensor data from IoT devices, OPC-UA bridges,
    or manual imports.
    """
    readings = []
    for r in body.readings:
        timestamp = r.timestamp or datetime.now(timezone.utc)
        
        try:
            sensor_type = SensorType(r.sensor_type.lower())
        except ValueError:
            sensor_type = SensorType.OTHER
        
        try:
            source = DataSource(r.source.upper())
        except ValueError:
            source = DataSource.API
        
        readings.append(SensorReading(
            machine_id=r.machine_id,
            timestamp=timestamp,
            sensor_type=sensor_type,
            sensor_id=r.sensor_id,
            value=r.value,
            unit=r.unit,
            source=source,
            quality=r.quality,
            metadata=r.metadata,
        ))
    
    results = service.ingest_batch(readings)
    
    return {
        "status": "ok",
        "ingested": results["success"],
        "failed": results["failed"],
        "total": len(body.readings),
    }


@router.post("/readings/opc-ua", summary="Ingest OPC-UA formatted readings")
async def ingest_opc_ua(
    readings: List[OpcUaReadingInput],
    service: IoTIngestionService = Depends(get_iot_service),
) -> Dict[str, Any]:
    """
    Ingest readings in OPC-UA format.
    
    Expected nodeId format: ns=2;s=MachineId.SensorType
    """
    parsed_readings = []
    for r in readings:
        try:
            parsed = parse_opc_ua_reading({
                "nodeId": r.nodeId,
                "value": r.value,
                "quality": r.quality,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "unit": r.unit,
            })
            parsed_readings.append(parsed)
        except Exception as e:
            logger.warning(f"Failed to parse OPC-UA reading: {e}")
    
    results = service.ingest_batch(parsed_readings)
    
    return {
        "status": "ok",
        "ingested": results["success"],
        "failed": results["failed"] + (len(readings) - len(parsed_readings)),
    }


@router.post("/readings/mqtt", summary="Ingest MQTT formatted readings")
async def ingest_mqtt(
    readings: List[MqttReadingInput],
    service: IoTIngestionService = Depends(get_iot_service),
) -> Dict[str, Any]:
    """
    Ingest readings in MQTT format.
    
    Topic format: sensors/{machine_id}/{sensor_type}
    """
    parsed_readings = []
    for r in readings:
        try:
            parsed = parse_mqtt_reading({
                "value": r.value,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "unit": r.unit,
                "quality": r.quality,
            }, topic=r.topic)
            parsed_readings.append(parsed)
        except Exception as e:
            logger.warning(f"Failed to parse MQTT reading: {e}")
    
    results = service.ingest_batch(parsed_readings)
    
    return {
        "status": "ok",
        "ingested": results["success"],
        "failed": results["failed"] + (len(readings) - len(parsed_readings)),
    }


@router.get("/readings/{machine_id}", summary="Get recent readings for a machine")
async def get_readings(
    machine_id: str,
    sensor_type: Optional[str] = Query(None, description="Filter by sensor type"),
    limit: int = Query(100, ge=1, le=1000),
    service: IoTIngestionService = Depends(get_iot_service),
) -> Dict[str, Any]:
    """
    Get recent sensor readings for a machine.
    """
    st = None
    if sensor_type:
        try:
            st = SensorType(sensor_type.lower())
        except ValueError:
            pass
    
    readings = service.get_recent_readings(machine_id, sensor_type=st, limit=limit)
    
    return {
        "machine_id": machine_id,
        "count": len(readings),
        "readings": [r.to_dict() for r in readings],
    }


@router.get("/stats", summary="Get ingestion statistics")
async def get_stats(
    service: IoTIngestionService = Depends(get_iot_service),
) -> IngestionStatsOutput:
    """
    Get statistics about sensor data ingestion.
    """
    stats = service.get_statistics()
    return IngestionStatsOutput(**stats)


@router.post("/flush", summary="Flush buffer to database")
async def flush_buffer(
    service: IoTIngestionService = Depends(get_iot_service),
) -> Dict[str, Any]:
    """
    Force flush the internal buffer to the database.
    """
    count = service.flush()
    return {"status": "ok", "flushed": count}


@router.post("/demo/generate", summary="Generate demo sensor data")
async def generate_demo(
    body: DemoGenerateInput,
    service: IoTIngestionService = Depends(get_iot_service),
) -> Dict[str, Any]:
    """
    Generate demo sensor data for testing.
    
    Creates simulated sensor readings with gradual degradation pattern.
    """
    readings = generate_demo_readings(
        machine_id=body.machine_id,
        duration_hours=body.duration_hours,
        interval_seconds=body.interval_seconds,
        degradation_factor=body.degradation_factor,
    )
    
    results = service.ingest_batch(readings)
    
    return {
        "status": "ok",
        "machine_id": body.machine_id,
        "readings_generated": len(readings),
        "ingested": results["success"],
        "duration_hours": body.duration_hours,
    }


@router.get("/sensor-types", summary="List available sensor types")
async def list_sensor_types() -> List[str]:
    """
    List all supported sensor types.
    """
    return [st.value for st in SensorType]


@router.get("/sources", summary="List available data sources")
async def list_sources() -> List[str]:
    """
    List all supported data sources.
    """
    return [ds.value for ds in DataSource]


