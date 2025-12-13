"""
════════════════════════════════════════════════════════════════════════════════════════════════════
Operation Execution Log - Unified Shopfloor Logs
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 9 Implementation: Unified logs that feed all mathematical models

This log captures every operation execution from the shopfloor and feeds:
- DataDrivenDurations (scheduling)
- GoldenRuns (process optimization)
- Digital Twin / RUL (machine health)
- Causal Analysis (effect estimation)
- ZDM (failure history)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Database path
EXECUTION_LOG_DB_PATH = Path(__file__).parent.parent.parent / "data" / "execution_logs.db"


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionLogStatus(str, Enum):
    """Status of an operation execution."""
    COMPLETED = "COMPLETED"
    ABORTED = "ABORTED"
    PARTIAL = "PARTIAL"


class ScrapReason(str, Enum):
    """Predefined scrap reasons."""
    MATERIAL_DEFECT = "material_defect"
    MACHINE_ERROR = "machine_error"
    OPERATOR_ERROR = "operator_error"
    TOOL_WEAR = "tool_wear"
    PROCESS_DEVIATION = "process_deviation"
    QUALITY_REJECT = "quality_reject"
    OTHER = "other"


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ProcessParams(BaseModel):
    """Process parameters applied during execution."""
    feed_rate: Optional[float] = None
    spindle_speed: Optional[float] = None
    depth_of_cut: Optional[float] = None
    coolant_flow: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    custom: Dict[str, Any] = Field(default_factory=dict)


class OperationExecutionLog(BaseModel):
    """
    Unified execution log for operations.
    
    This model captures all relevant data from shopfloor execution to feed
    mathematical models and ML pipelines.
    """
    id: Optional[int] = None
    
    # Order/Operation identifiers
    order_id: str
    operation_id: str  # Can be operation code or FK
    operation_code: Optional[str] = None
    
    # Resources
    machine_id: str
    operator_id: Optional[str] = None
    
    # Product linkage
    revision_id: Optional[int] = None
    article_id: Optional[str] = None
    digital_identity_id: Optional[int] = None  # For traceability
    
    # Timing
    start_time: datetime
    end_time: datetime
    setup_time_s: float = 0.0
    cycle_time_s: float = 0.0  # Average cycle time per piece
    pause_time_s: float = 0.0
    
    # Quantities
    qty_good: float = 0.0
    qty_scrap: float = 0.0
    qty_rework: float = 0.0
    scrap_reason: Optional[str] = None
    
    # Process parameters
    params: ProcessParams = Field(default_factory=ProcessParams)
    
    # Quality checkpoints
    quality_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Energy/Resources
    energy_kwh: Optional[float] = None
    tool_id: Optional[str] = None
    tool_wear_pct: Optional[float] = None
    
    # Downtime
    downtime_reason: Optional[str] = None
    downtime_minutes: float = 0.0
    
    # Status
    status: ExecutionLogStatus = ExecutionLogStatus.COMPLETED
    notes: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def total_time_s(self) -> float:
        """Total execution time in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def effective_time_s(self) -> float:
        """Effective working time (excluding pauses)."""
        return self.total_time_s - self.pause_time_s - (self.downtime_minutes * 60)
    
    @property
    def scrap_rate(self) -> float:
        """Scrap rate as percentage."""
        total = self.qty_good + self.qty_scrap
        return (self.qty_scrap / total * 100) if total > 0 else 0.0
    
    @property
    def oee_quality(self) -> float:
        """Quality component of OEE."""
        total = self.qty_good + self.qty_scrap
        return (self.qty_good / total) if total > 0 else 1.0


class ExecutionLogQuery(BaseModel):
    """Query parameters for execution logs."""
    operation_id: Optional[str] = None
    machine_id: Optional[str] = None
    revision_id: Optional[int] = None
    article_id: Optional[str] = None
    status: Optional[ExecutionLogStatus] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    limit: int = 100


class ExecutionLogStats(BaseModel):
    """Aggregated statistics from execution logs."""
    operation_id: str
    machine_id: Optional[str] = None
    
    # Count
    n_executions: int = 0
    
    # Time stats
    avg_cycle_time_s: float = 0.0
    std_cycle_time_s: float = 0.0
    min_cycle_time_s: float = 0.0
    max_cycle_time_s: float = 0.0
    avg_setup_time_s: float = 0.0
    
    # Quality stats
    avg_scrap_rate: float = 0.0
    total_qty_good: float = 0.0
    total_qty_scrap: float = 0.0
    
    # Energy stats
    avg_energy_kwh: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_db_exists() -> None:
    """Create execution logs database and tables."""
    EXECUTION_LOG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(EXECUTION_LOG_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS operation_execution_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT NOT NULL,
            operation_id TEXT NOT NULL,
            operation_code TEXT,
            machine_id TEXT NOT NULL,
            operator_id TEXT,
            revision_id INTEGER,
            article_id TEXT,
            digital_identity_id INTEGER,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            setup_time_s REAL DEFAULT 0,
            cycle_time_s REAL DEFAULT 0,
            pause_time_s REAL DEFAULT 0,
            qty_good REAL DEFAULT 0,
            qty_scrap REAL DEFAULT 0,
            qty_rework REAL DEFAULT 0,
            scrap_reason TEXT,
            params TEXT DEFAULT '{}',
            quality_results TEXT DEFAULT '{}',
            energy_kwh REAL,
            tool_id TEXT,
            tool_wear_pct REAL,
            downtime_reason TEXT,
            downtime_minutes REAL DEFAULT 0,
            status TEXT DEFAULT 'COMPLETED',
            notes TEXT,
            created_at TEXT NOT NULL,
            
            -- Indexes for common queries
            UNIQUE(order_id, operation_id, machine_id, start_time)
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_exec_log_operation 
        ON operation_execution_logs(operation_id, machine_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_exec_log_revision 
        ON operation_execution_logs(revision_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_exec_log_time 
        ON operation_execution_logs(start_time, end_time)
    """)
    
    conn.commit()
    conn.close()
    logger.info("Execution logs database initialized")


# Ensure DB exists on module load
_ensure_db_exists()


# ═══════════════════════════════════════════════════════════════════════════════
# CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_execution_log(log: OperationExecutionLog) -> int:
    """
    Create a new execution log entry.
    
    Args:
        log: The execution log to create
    
    Returns:
        ID of created log
    """
    conn = sqlite3.connect(EXECUTION_LOG_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO operation_execution_logs (
            order_id, operation_id, operation_code, machine_id, operator_id,
            revision_id, article_id, digital_identity_id,
            start_time, end_time, setup_time_s, cycle_time_s, pause_time_s,
            qty_good, qty_scrap, qty_rework, scrap_reason,
            params, quality_results, energy_kwh, tool_id, tool_wear_pct,
            downtime_reason, downtime_minutes, status, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        log.order_id,
        log.operation_id,
        log.operation_code,
        log.machine_id,
        log.operator_id,
        log.revision_id,
        log.article_id,
        log.digital_identity_id,
        log.start_time.isoformat(),
        log.end_time.isoformat(),
        log.setup_time_s,
        log.cycle_time_s,
        log.pause_time_s,
        log.qty_good,
        log.qty_scrap,
        log.qty_rework,
        log.scrap_reason,
        json.dumps(log.params.dict()),
        json.dumps(log.quality_results),
        log.energy_kwh,
        log.tool_id,
        log.tool_wear_pct,
        log.downtime_reason,
        log.downtime_minutes,
        log.status.value,
        log.notes,
        log.created_at.isoformat(),
    ))
    
    log_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    logger.info(f"Created execution log {log_id} for order {log.order_id}")
    return log_id


def get_execution_log(log_id: int) -> Optional[OperationExecutionLog]:
    """Get a single execution log by ID."""
    conn = sqlite3.connect(EXECUTION_LOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM operation_execution_logs WHERE id = ?", (log_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return _row_to_log(row)


def query_execution_logs(query: ExecutionLogQuery) -> List[OperationExecutionLog]:
    """
    Query execution logs with filters.
    
    Args:
        query: Query parameters
    
    Returns:
        List of matching execution logs
    """
    conn = sqlite3.connect(EXECUTION_LOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    sql = "SELECT * FROM operation_execution_logs WHERE 1=1"
    params = []
    
    if query.operation_id:
        sql += " AND operation_id = ?"
        params.append(query.operation_id)
    
    if query.machine_id:
        sql += " AND machine_id = ?"
        params.append(query.machine_id)
    
    if query.revision_id:
        sql += " AND revision_id = ?"
        params.append(query.revision_id)
    
    if query.article_id:
        sql += " AND article_id = ?"
        params.append(query.article_id)
    
    if query.status:
        sql += " AND status = ?"
        params.append(query.status.value)
    
    if query.from_date:
        sql += " AND start_time >= ?"
        params.append(query.from_date.isoformat())
    
    if query.to_date:
        sql += " AND end_time <= ?"
        params.append(query.to_date.isoformat())
    
    sql += " ORDER BY start_time DESC LIMIT ?"
    params.append(query.limit)
    
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    
    return [_row_to_log(row) for row in rows]


def get_execution_stats(
    operation_id: str,
    machine_id: Optional[str] = None,
    days: int = 30,
) -> ExecutionLogStats:
    """
    Get aggregated statistics for an operation.
    
    Args:
        operation_id: Operation to analyze
        machine_id: Optional machine filter
        days: Number of days to look back
    
    Returns:
        ExecutionLogStats with aggregated metrics
    """
    import numpy as np
    from datetime import timedelta
    
    conn = sqlite3.connect(EXECUTION_LOG_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    
    sql = """
        SELECT 
            cycle_time_s, setup_time_s, qty_good, qty_scrap, energy_kwh
        FROM operation_execution_logs 
        WHERE operation_id = ? 
        AND status = 'COMPLETED'
        AND start_time >= ?
    """
    params = [operation_id, cutoff]
    
    if machine_id:
        sql += " AND machine_id = ?"
        params.append(machine_id)
    
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return ExecutionLogStats(operation_id=operation_id, machine_id=machine_id)
    
    cycle_times = [r["cycle_time_s"] for r in rows if r["cycle_time_s"] > 0]
    setup_times = [r["setup_time_s"] for r in rows if r["setup_time_s"] > 0]
    qty_good = sum(r["qty_good"] or 0 for r in rows)
    qty_scrap = sum(r["qty_scrap"] or 0 for r in rows)
    energies = [r["energy_kwh"] for r in rows if r["energy_kwh"] is not None]
    
    total_qty = qty_good + qty_scrap
    
    return ExecutionLogStats(
        operation_id=operation_id,
        machine_id=machine_id,
        n_executions=len(rows),
        avg_cycle_time_s=float(np.mean(cycle_times)) if cycle_times else 0.0,
        std_cycle_time_s=float(np.std(cycle_times)) if len(cycle_times) > 1 else 0.0,
        min_cycle_time_s=float(np.min(cycle_times)) if cycle_times else 0.0,
        max_cycle_time_s=float(np.max(cycle_times)) if cycle_times else 0.0,
        avg_setup_time_s=float(np.mean(setup_times)) if setup_times else 0.0,
        avg_scrap_rate=(qty_scrap / total_qty * 100) if total_qty > 0 else 0.0,
        total_qty_good=qty_good,
        total_qty_scrap=qty_scrap,
        avg_energy_kwh=float(np.mean(energies)) if energies else None,
    )


def _row_to_log(row: sqlite3.Row) -> OperationExecutionLog:
    """Convert database row to OperationExecutionLog."""
    params_dict = json.loads(row["params"] or "{}")
    quality_dict = json.loads(row["quality_results"] or "{}")
    
    return OperationExecutionLog(
        id=row["id"],
        order_id=row["order_id"],
        operation_id=row["operation_id"],
        operation_code=row["operation_code"],
        machine_id=row["machine_id"],
        operator_id=row["operator_id"],
        revision_id=row["revision_id"],
        article_id=row["article_id"],
        digital_identity_id=row["digital_identity_id"],
        start_time=datetime.fromisoformat(row["start_time"]),
        end_time=datetime.fromisoformat(row["end_time"]),
        setup_time_s=row["setup_time_s"] or 0.0,
        cycle_time_s=row["cycle_time_s"] or 0.0,
        pause_time_s=row["pause_time_s"] or 0.0,
        qty_good=row["qty_good"] or 0.0,
        qty_scrap=row["qty_scrap"] or 0.0,
        qty_rework=row["qty_rework"] or 0.0,
        scrap_reason=row["scrap_reason"],
        params=ProcessParams(**params_dict),
        quality_results=quality_dict,
        energy_kwh=row["energy_kwh"],
        tool_id=row["tool_id"],
        tool_wear_pct=row["tool_wear_pct"],
        downtime_reason=row["downtime_reason"],
        downtime_minutes=row["downtime_minutes"] or 0.0,
        status=ExecutionLogStatus(row["status"]),
        notes=row["notes"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def log_operation_completion(
    order_id: str,
    operation_id: str,
    machine_id: str,
    start_time: datetime,
    end_time: datetime,
    qty_good: float,
    qty_scrap: float = 0.0,
    operator_id: Optional[str] = None,
    revision_id: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    scrap_reason: Optional[str] = None,
    downtime_minutes: float = 0.0,
    energy_kwh: Optional[float] = None,
) -> int:
    """
    Convenience function to log an operation completion.
    
    This is the main entry point from Shopfloor API.
    """
    total_qty = qty_good + qty_scrap
    total_time_s = (end_time - start_time).total_seconds()
    
    # Estimate cycle time per piece
    cycle_time_s = (total_time_s - downtime_minutes * 60) / total_qty if total_qty > 0 else 0.0
    
    process_params = ProcessParams()
    if params:
        process_params = ProcessParams(**params)
    
    log = OperationExecutionLog(
        order_id=order_id,
        operation_id=operation_id,
        machine_id=machine_id,
        operator_id=operator_id,
        revision_id=revision_id,
        start_time=start_time,
        end_time=end_time,
        cycle_time_s=cycle_time_s,
        qty_good=qty_good,
        qty_scrap=qty_scrap,
        scrap_reason=scrap_reason,
        params=process_params,
        downtime_minutes=downtime_minutes,
        energy_kwh=energy_kwh,
        status=ExecutionLogStatus.COMPLETED,
    )
    
    return create_execution_log(log)


def get_recent_logs_for_model(
    operation_id: str,
    machine_id: Optional[str] = None,
    days: int = 30,
    min_qty: float = 1.0,
) -> List[OperationExecutionLog]:
    """
    Get recent logs suitable for model training/inference.
    
    Filters out incomplete/aborted executions.
    """
    from datetime import timedelta
    
    query = ExecutionLogQuery(
        operation_id=operation_id,
        machine_id=machine_id,
        status=ExecutionLogStatus.COMPLETED,
        from_date=datetime.utcnow() - timedelta(days=days),
        limit=500,
    )
    
    logs = query_execution_logs(query)
    
    # Filter by minimum quantity
    return [log for log in logs if (log.qty_good + log.qty_scrap) >= min_qty]



