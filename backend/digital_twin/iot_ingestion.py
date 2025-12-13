"""
════════════════════════════════════════════════════════════════════════════════════════════════════
IoT SENSOR INGESTION FOR DIGITAL TWIN / PREDICTIVECARE
════════════════════════════════════════════════════════════════════════════════════════════════════

Pipeline de ingestão de dados de sensores de máquinas para o Digital Twin.

Fontes suportadas:
- OPC-UA: Protocolo industrial standard
- MQTT: IoT lightweight messaging
- CSV/Excel: Importação batch
- API: Integração direta

Cada leitura de sensor é normalizada para SensorReading e persistida em dt_sensor_readings.
Os dados alimentam o SHI-DT (CVAE) para cálculo de Health Index e RUL.

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Literal
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class SensorType(str, Enum):
    """Tipos de sensores suportados."""
    VIBRATION = "vibration"
    TEMPERATURE = "temperature"
    CURRENT = "current"
    PRESSURE = "pressure"
    ACOUSTIC = "acoustic"
    SPEED = "speed"
    TORQUE = "torque"
    FLOW = "flow"
    HUMIDITY = "humidity"
    OTHER = "other"


class DataSource(str, Enum):
    """Fonte dos dados de sensor."""
    OPC_UA = "OPC_UA"
    MQTT = "MQTT"
    CSV = "CSV"
    EXCEL = "EXCEL"
    API = "API"
    SIMULATION = "SIMULATION"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorReading:
    """
    Leitura individual de um sensor de máquina.
    
    Attributes:
        machine_id: Identificador da máquina
        timestamp: Momento da leitura (UTC)
        sensor_type: Tipo de sensor (vibration, temperature, etc.)
        sensor_id: ID específico do sensor (opcional, para múltiplos sensores do mesmo tipo)
        value: Valor da leitura
        unit: Unidade de medida
        source: Fonte dos dados
        quality: Qualidade da leitura (0-100, 100=perfeito)
        metadata: Dados adicionais
    """
    machine_id: str
    timestamp: datetime
    sensor_type: SensorType
    value: float
    unit: str
    sensor_id: Optional[str] = None
    source: DataSource = DataSource.API
    quality: float = 100.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validação pós-inicialização."""
        # Garantir timezone
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Validar qualidade
        self.quality = max(0.0, min(100.0, self.quality))
        
        # Converter sensor_type se string
        if isinstance(self.sensor_type, str):
            try:
                self.sensor_type = SensorType(self.sensor_type.lower())
            except ValueError:
                self.sensor_type = SensorType.OTHER
        
        # Converter source se string
        if isinstance(self.source, str):
            try:
                self.source = DataSource(self.source.upper())
            except ValueError:
                self.source = DataSource.API
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "sensor_type": self.sensor_type.value,
            "sensor_id": self.sensor_id,
            "value": self.value,
            "unit": self.unit,
            "source": self.source.value,
            "quality": self.quality,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SensorReading:
        """Cria a partir de dicionário."""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        
        return cls(
            machine_id=data["machine_id"],
            timestamp=timestamp,
            sensor_type=data.get("sensor_type", "other"),
            sensor_id=data.get("sensor_id"),
            value=float(data["value"]),
            unit=data.get("unit", ""),
            source=data.get("source", "API"),
            quality=float(data.get("quality", 100.0)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SensorBatch:
    """Batch de leituras de sensores."""
    readings: List[SensorReading]
    batch_id: Optional[str] = None
    source: DataSource = DataSource.API
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def count(self) -> int:
        return len(self.readings)
    
    @property
    def machines(self) -> List[str]:
        return list(set(r.machine_id for r in self.readings))
    
    @property
    def time_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        if not self.readings:
            return None, None
        timestamps = [r.timestamp for r in self.readings]
        return min(timestamps), max(timestamps)


# ═══════════════════════════════════════════════════════════════════════════════
# SQLAlchemy MODEL (for persistence)
# ═══════════════════════════════════════════════════════════════════════════════

# Try to import SQLAlchemy for ORM model
try:
    from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index
    from sqlalchemy.ext.declarative import declarative_base
    
    # Use existing Base if available
    try:
        from models_common import Base
    except ImportError:
        Base = declarative_base()
    
    class DTSensorReading(Base):
        """SQLAlchemy model for sensor readings."""
        __tablename__ = "dt_sensor_readings"
        
        id = Column(Integer, primary_key=True, autoincrement=True)
        machine_id = Column(String(64), nullable=False, index=True)
        timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
        sensor_type = Column(String(32), nullable=False)
        sensor_id = Column(String(64), nullable=True)
        value = Column(Float, nullable=False)
        unit = Column(String(32), nullable=False, default="")
        source = Column(String(32), nullable=False, default="API")
        quality = Column(Float, nullable=False, default=100.0)
        metadata_json = Column(Text, nullable=True)
        created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
        
        # Composite index for efficient queries
        __table_args__ = (
            Index('ix_dt_sensor_machine_time', 'machine_id', 'timestamp'),
            Index('ix_dt_sensor_type_time', 'sensor_type', 'timestamp'),
        )
        
        def to_sensor_reading(self) -> SensorReading:
            """Convert to SensorReading dataclass."""
            return SensorReading(
                machine_id=self.machine_id,
                timestamp=self.timestamp,
                sensor_type=SensorType(self.sensor_type),
                sensor_id=self.sensor_id,
                value=self.value,
                unit=self.unit,
                source=DataSource(self.source),
                quality=self.quality,
                metadata=json.loads(self.metadata_json) if self.metadata_json else {},
            )
        
        @classmethod
        def from_sensor_reading(cls, reading: SensorReading) -> "DTSensorReading":
            """Create from SensorReading dataclass."""
            return cls(
                machine_id=reading.machine_id,
                timestamp=reading.timestamp,
                sensor_type=reading.sensor_type.value,
                sensor_id=reading.sensor_id,
                value=reading.value,
                unit=reading.unit,
                source=reading.source.value,
                quality=reading.quality,
                metadata_json=json.dumps(reading.metadata) if reading.metadata else None,
            )
    
    SQLALCHEMY_AVAILABLE = True
    
except ImportError:
    DTSensorReading = None
    SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available - sensor persistence disabled")


# ═══════════════════════════════════════════════════════════════════════════════
# INGESTION SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class IoTIngestionService:
    """
    Serviço de ingestão de dados de sensores de máquinas.
    
    Responsabilidades:
    - Receber e normalizar leituras de sensores
    - Persistir em base de dados
    - Agregar dados para o SHI-DT
    - Buffer para batch processing
    """
    
    def __init__(
        self,
        db_session=None,
        buffer_size: int = 100,
        auto_flush: bool = True,
    ):
        """
        Initialize the IoT Ingestion Service.
        
        Args:
            db_session: SQLAlchemy session (optional)
            buffer_size: Size of the internal buffer before auto-flush
            auto_flush: Whether to auto-flush when buffer is full
        """
        self.db_session = db_session
        self.buffer_size = buffer_size
        self.auto_flush = auto_flush
        
        # Internal buffer
        self._buffer: List[SensorReading] = []
        
        # Statistics
        self._stats = {
            "total_ingested": 0,
            "total_persisted": 0,
            "total_errors": 0,
            "by_machine": defaultdict(int),
            "by_sensor_type": defaultdict(int),
        }
        
        logger.info(f"IoTIngestionService initialized (buffer_size={buffer_size})")
    
    def ingest_reading(self, reading: SensorReading) -> bool:
        """
        Ingest a single sensor reading.
        
        Args:
            reading: The sensor reading to ingest
            
        Returns:
            True if successfully ingested
        """
        try:
            # Validate
            if not self._validate_reading(reading):
                return False
            
            # Add to buffer
            self._buffer.append(reading)
            
            # Update stats
            self._stats["total_ingested"] += 1
            self._stats["by_machine"][reading.machine_id] += 1
            self._stats["by_sensor_type"][reading.sensor_type.value] += 1
            
            # Auto-flush if needed
            if self.auto_flush and len(self._buffer) >= self.buffer_size:
                self.flush()
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting reading: {e}")
            self._stats["total_errors"] += 1
            return False
    
    def ingest_batch(self, readings: List[SensorReading]) -> Dict[str, int]:
        """
        Ingest multiple sensor readings.
        
        Args:
            readings: List of sensor readings
            
        Returns:
            Dict with success/failure counts
        """
        results = {"success": 0, "failed": 0}
        
        for reading in readings:
            if self.ingest_reading(reading):
                results["success"] += 1
            else:
                results["failed"] += 1
        
        logger.info(f"Batch ingestion: {results['success']} success, {results['failed']} failed")
        return results
    
    def flush(self) -> int:
        """
        Flush buffer to database.
        
        Returns:
            Number of readings persisted
        """
        if not self._buffer:
            return 0
        
        persisted = 0
        
        if SQLALCHEMY_AVAILABLE and self.db_session is not None:
            try:
                for reading in self._buffer:
                    db_reading = DTSensorReading.from_sensor_reading(reading)
                    self.db_session.add(db_reading)
                    persisted += 1
                
                self.db_session.commit()
                self._stats["total_persisted"] += persisted
                logger.info(f"Flushed {persisted} readings to database")
                
            except Exception as e:
                logger.error(f"Error flushing to database: {e}")
                self.db_session.rollback()
                persisted = 0
        else:
            # No database - just log
            persisted = len(self._buffer)
            logger.debug(f"No database session - {persisted} readings discarded from buffer")
        
        # Clear buffer
        self._buffer.clear()
        
        return persisted
    
    def _validate_reading(self, reading: SensorReading) -> bool:
        """Validate a sensor reading."""
        if not reading.machine_id:
            logger.warning("Reading rejected: missing machine_id")
            return False
        
        if reading.value is None:
            logger.warning(f"Reading rejected: missing value for {reading.machine_id}")
            return False
        
        # Check for NaN/Inf
        import math
        if math.isnan(reading.value) or math.isinf(reading.value):
            logger.warning(f"Reading rejected: invalid value {reading.value} for {reading.machine_id}")
            return False
        
        return True
    
    def get_recent_readings(
        self,
        machine_id: str,
        sensor_type: Optional[SensorType] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[SensorReading]:
        """
        Get recent readings for a machine.
        
        Args:
            machine_id: Machine identifier
            sensor_type: Filter by sensor type (optional)
            limit: Maximum number of readings
            since: Only readings after this timestamp
            
        Returns:
            List of sensor readings
        """
        if SQLALCHEMY_AVAILABLE and self.db_session is not None:
            query = self.db_session.query(DTSensorReading).filter(
                DTSensorReading.machine_id == machine_id
            )
            
            if sensor_type:
                query = query.filter(DTSensorReading.sensor_type == sensor_type.value)
            
            if since:
                query = query.filter(DTSensorReading.timestamp >= since)
            
            query = query.order_by(DTSensorReading.timestamp.desc()).limit(limit)
            
            return [r.to_sensor_reading() for r in query.all()]
        
        # Fallback: return from buffer
        readings = [
            r for r in self._buffer
            if r.machine_id == machine_id
            and (sensor_type is None or r.sensor_type == sensor_type)
            and (since is None or r.timestamp >= since)
        ]
        return sorted(readings, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            "total_ingested": self._stats["total_ingested"],
            "total_persisted": self._stats["total_persisted"],
            "total_errors": self._stats["total_errors"],
            "buffer_size": len(self._buffer),
            "by_machine": dict(self._stats["by_machine"]),
            "by_sensor_type": dict(self._stats["by_sensor_type"]),
        }
    
    def convert_to_sensor_snapshot(
        self,
        machine_id: str,
        timestamp: Optional[datetime] = None,
    ) -> Optional[Any]:
        """
        Convert recent readings to a SensorSnapshot for SHI-DT.
        
        Aggregates the latest readings from different sensor types
        into a single SensorSnapshot that can be processed by the CVAE.
        
        Args:
            machine_id: Machine identifier
            timestamp: Reference timestamp (default: now)
            
        Returns:
            SensorSnapshot or None if insufficient data
        """
        from digital_twin.health_indicator_cvae import SensorSnapshot
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Get recent readings (last 5 minutes)
        since = timestamp - timedelta(minutes=5)
        readings = self.get_recent_readings(machine_id, limit=500, since=since)
        
        if not readings:
            logger.debug(f"No recent readings for machine {machine_id}")
            return None
        
        # Group by sensor type and get latest
        latest_by_type: Dict[SensorType, SensorReading] = {}
        for reading in readings:
            if reading.sensor_type not in latest_by_type:
                latest_by_type[reading.sensor_type] = reading
        
        # Build SensorSnapshot
        snapshot = SensorSnapshot(
            machine_id=machine_id,
            timestamp=timestamp,
        )
        
        # Map sensor readings to snapshot fields
        if SensorType.VIBRATION in latest_by_type:
            snapshot.vibration_rms = latest_by_type[SensorType.VIBRATION].value
        
        if SensorType.TEMPERATURE in latest_by_type:
            snapshot.temperature_motor = latest_by_type[SensorType.TEMPERATURE].value
        
        if SensorType.CURRENT in latest_by_type:
            snapshot.current_rms = latest_by_type[SensorType.CURRENT].value
        
        if SensorType.PRESSURE in latest_by_type:
            snapshot.pressure_main = latest_by_type[SensorType.PRESSURE].value
        
        if SensorType.ACOUSTIC in latest_by_type:
            snapshot.acoustic_emission = latest_by_type[SensorType.ACOUSTIC].value
        
        return snapshot


# ═══════════════════════════════════════════════════════════════════════════════
# PARSERS FOR DIFFERENT SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

def parse_opc_ua_reading(payload: Dict[str, Any]) -> SensorReading:
    """
    Parse OPC-UA reading format.
    
    Expected payload:
        {
            "nodeId": "ns=2;s=Machine1.Vibration",
            "value": 0.45,
            "quality": "Good",
            "timestamp": "2025-12-05T10:30:00Z",
        }
    """
    # Extract machine_id and sensor_type from nodeId
    node_id = payload.get("nodeId", "")
    parts = node_id.split(".")
    
    machine_id = parts[0].split("=")[-1] if parts else "unknown"
    sensor_hint = parts[1].lower() if len(parts) > 1 else "other"
    
    # Map sensor hint to type
    type_map = {
        "vibration": SensorType.VIBRATION,
        "temperature": SensorType.TEMPERATURE,
        "temp": SensorType.TEMPERATURE,
        "current": SensorType.CURRENT,
        "pressure": SensorType.PRESSURE,
        "acoustic": SensorType.ACOUSTIC,
        "speed": SensorType.SPEED,
        "torque": SensorType.TORQUE,
    }
    sensor_type = type_map.get(sensor_hint, SensorType.OTHER)
    
    # Parse timestamp
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Parse quality
    quality = 100.0
    q = payload.get("quality", "Good")
    if isinstance(q, str):
        quality = 100.0 if q.lower() == "good" else 50.0
    elif isinstance(q, (int, float)):
        quality = float(q)
    
    return SensorReading(
        machine_id=machine_id,
        timestamp=timestamp,
        sensor_type=sensor_type,
        sensor_id=node_id,
        value=float(payload.get("value", 0)),
        unit=payload.get("unit", ""),
        source=DataSource.OPC_UA,
        quality=quality,
        metadata={"raw_payload": payload},
    )


def parse_mqtt_reading(payload: Dict[str, Any], topic: str = "") -> SensorReading:
    """
    Parse MQTT reading format.
    
    Expected payload (topic: sensors/machine1/vibration):
        {
            "value": 0.45,
            "timestamp": "2025-12-05T10:30:00Z",
            "unit": "mm/s",
        }
    """
    # Extract from topic: sensors/{machine_id}/{sensor_type}
    topic_parts = topic.split("/")
    machine_id = topic_parts[1] if len(topic_parts) > 1 else payload.get("machine_id", "unknown")
    sensor_hint = topic_parts[2] if len(topic_parts) > 2 else payload.get("sensor_type", "other")
    
    # Map sensor type
    type_map = {
        "vibration": SensorType.VIBRATION,
        "temperature": SensorType.TEMPERATURE,
        "current": SensorType.CURRENT,
        "pressure": SensorType.PRESSURE,
    }
    sensor_type = type_map.get(sensor_hint.lower(), SensorType.OTHER)
    
    # Parse timestamp
    timestamp = payload.get("timestamp")
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    elif timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    return SensorReading(
        machine_id=machine_id,
        timestamp=timestamp,
        sensor_type=sensor_type,
        value=float(payload.get("value", 0)),
        unit=payload.get("unit", ""),
        source=DataSource.MQTT,
        quality=float(payload.get("quality", 100.0)),
        metadata={"topic": topic, "raw_payload": payload},
    )


def parse_csv_readings(
    csv_content: str,
    machine_id: str,
    sensor_type: SensorType,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
) -> List[SensorReading]:
    """
    Parse CSV content into sensor readings.
    
    Args:
        csv_content: CSV string content
        machine_id: Machine identifier
        sensor_type: Type of sensor
        timestamp_col: Name of timestamp column
        value_col: Name of value column
        
    Returns:
        List of sensor readings
    """
    import csv
    from io import StringIO
    
    readings = []
    reader = csv.DictReader(StringIO(csv_content))
    
    for row in reader:
        try:
            timestamp = datetime.fromisoformat(row[timestamp_col].replace("Z", "+00:00"))
            value = float(row[value_col])
            
            readings.append(SensorReading(
                machine_id=machine_id,
                timestamp=timestamp,
                sensor_type=sensor_type,
                value=value,
                unit=row.get("unit", ""),
                source=DataSource.CSV,
                quality=float(row.get("quality", 100.0)),
            ))
        except Exception as e:
            logger.warning(f"Failed to parse CSV row: {e}")
            continue
    
    return readings


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION / DEMO DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_demo_readings(
    machine_id: str,
    duration_hours: float = 24,
    interval_seconds: float = 60,
    degradation_factor: float = 0.01,
) -> List[SensorReading]:
    """
    Generate demo sensor readings for testing.
    
    Simulates gradual degradation of a machine over time.
    
    Args:
        machine_id: Machine identifier
        duration_hours: Duration of data to generate
        interval_seconds: Interval between readings
        degradation_factor: How fast the machine degrades (0-1)
        
    Returns:
        List of simulated sensor readings
    """
    import random
    
    readings = []
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=duration_hours)
    
    num_readings = int(duration_hours * 3600 / interval_seconds)
    
    # Base values (healthy machine)
    base_vibration = 0.5  # mm/s
    base_temperature = 45.0  # °C
    base_current = 10.0  # A
    
    for i in range(num_readings):
        timestamp = start + timedelta(seconds=i * interval_seconds)
        
        # Progress through time (0 to 1)
        progress = i / num_readings
        
        # Apply degradation
        degradation = progress * degradation_factor * 10
        
        # Vibration increases with degradation
        vibration = base_vibration * (1 + degradation) + random.gauss(0, 0.05)
        readings.append(SensorReading(
            machine_id=machine_id,
            timestamp=timestamp,
            sensor_type=SensorType.VIBRATION,
            sensor_id=f"{machine_id}_vib_1",
            value=max(0, vibration),
            unit="mm/s",
            source=DataSource.SIMULATION,
        ))
        
        # Temperature increases with degradation
        temperature = base_temperature * (1 + degradation * 0.3) + random.gauss(0, 2)
        readings.append(SensorReading(
            machine_id=machine_id,
            timestamp=timestamp,
            sensor_type=SensorType.TEMPERATURE,
            sensor_id=f"{machine_id}_temp_motor",
            value=temperature,
            unit="°C",
            source=DataSource.SIMULATION,
        ))
        
        # Current fluctuates more with degradation
        current = base_current * (1 + degradation * 0.2) + random.gauss(0, 0.5 + degradation)
        readings.append(SensorReading(
            machine_id=machine_id,
            timestamp=timestamp,
            sensor_type=SensorType.CURRENT,
            sensor_id=f"{machine_id}_current_rms",
            value=max(0, current),
            unit="A",
            source=DataSource.SIMULATION,
        ))
    
    logger.info(f"Generated {len(readings)} demo readings for {machine_id}")
    return readings


