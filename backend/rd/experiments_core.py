"""
ProdPlan 4.0 - R&D Experiments Core
===================================

Núcleo de logging e gestão de experiências R&D.

Funcionalidades:
- Registo de experiências com contexto
- Tracking de métricas e KPIs
- Persistência em SQLite/JSON
- API para consulta de resultados

R&D / SIFIDE: Documentação de todas as experiências para relatórios.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class WorkPackage(str, Enum):
    """Work Packages do projeto R&D."""
    WP1_ROUTING = "WP1_ROUTING"
    WP2_SUGGESTIONS = "WP2_SUGGESTIONS"
    WP3_INVENTORY_CAPACITY = "WP3_INVENTORY_CAPACITY"
    WP3_INVENTORY = "WP3_INVENTORY_CAPACITY"  # Alias
    WP4_LEARNING_SCHEDULER = "WP4_LEARNING_SCHEDULER"
    WP4_LEARNING = "WP4_LEARNING_SCHEDULER"  # Alias
    WPX_TRUST_EVOLUTION = "WPX_TRUST_EVOLUTION"  # Trust Index evolution tracking
    WPX_GAPFILL_LITE = "WPX_GAPFILL_LITE"  # Gap Filling Lite experiments
    WPX_COMPLIANCE_EVOLUTION = "WPX_COMPLIANCE_EVOLUTION"  # Compliance evolution tracking
    WPX_DATA_INGESTION = "WPX_DATA_INGESTION"  # Operational data ingestion tracking
    WPX_PREDICTIVECARE = "WPX_PREDICTIVECARE"  # PredictiveCare / Maintenance tracking


class ExperimentStatus(str, Enum):
    """Status de uma experiência."""
    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RDExperimentCreate(BaseModel):
    """Request para criar experiência."""
    wp: WorkPackage
    name: str
    description: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class RDExperiment(BaseModel):
    """Modelo de experiência R&D."""
    id: int
    wp: WorkPackage
    name: str
    description: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.CREATED
    created_at: datetime
    updated_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    duration_sec: Optional[float] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    kpis: Dict[str, Any] = Field(default_factory=dict)
    conclusion: Optional[str] = None
    error_message: Optional[str] = None


class RDExperimentUpdate(BaseModel):
    """Request para atualizar experiência."""
    status: Optional[ExperimentStatus] = None
    summary: Optional[Dict[str, Any]] = None
    kpis: Optional[Dict[str, Any]] = None
    conclusion: Optional[str] = None
    error_message: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

# Path para base de dados
DB_PATH = Path(__file__).parent.parent.parent / "data" / "rd_experiments.db"


def _ensure_db_exists() -> None:
    """Cria base de dados e tabelas se não existirem."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tabela principal de experiências
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wp TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            context TEXT DEFAULT '{}',
            parameters TEXT DEFAULT '{}',
            status TEXT DEFAULT 'created',
            created_at TEXT NOT NULL,
            updated_at TEXT,
            finished_at TEXT,
            duration_sec REAL,
            summary TEXT DEFAULT '{}',
            kpis TEXT DEFAULT '{}',
            conclusion TEXT,
            error_message TEXT
        )
    """)
    
    # Tabela WP1 - Routing
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_wp1_routing (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            baseline_strategy TEXT DEFAULT 'FIFO',
            num_orders INTEGER,
            num_machines INTEGER,
            makespan_hours REAL,
            tardiness_hours REAL,
            setup_hours REAL,
            improvement_pct REAL,
            FOREIGN KEY (experiment_id) REFERENCES rd_experiments(id)
        )
    """)
    
    # Tabela WP2 - Suggestions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_wp2_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            suggestion_type TEXT,
            suggestion_count INTEGER,
            accepted_count INTEGER,
            rejected_count INTEGER,
            avg_confidence REAL,
            avg_impact REAL,
            precision REAL,
            recall REAL,
            FOREIGN KEY (experiment_id) REFERENCES rd_experiments(id)
        )
    """)
    
    # Tabela WP3 - Inventory
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_wp3_inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            policy TEXT NOT NULL,
            baseline_policy TEXT DEFAULT 'CLASSIC',
            num_skus INTEGER,
            horizon_days INTEGER,
            avg_stock REAL,
            stockout_days INTEGER,
            service_level REAL,
            cost_reduction_pct REAL,
            FOREIGN KEY (experiment_id) REFERENCES rd_experiments(id)
        )
    """)
    
    # Tabela WP4 - Learning Scheduler
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_wp4_scheduler (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            num_episodes INTEGER,
            total_timesteps INTEGER,
            avg_reward REAL,
            best_reward REAL,
            avg_makespan REAL,
            baseline_makespan REAL,
            improvement_pct REAL,
            FOREIGN KEY (experiment_id) REFERENCES rd_experiments(id)
        )
    """)
    
    # Tabela WPX - Trust Index Evolution
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_wpx_trust_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            dpp_id TEXT NOT NULL,
            trust_index_old REAL NOT NULL,
            trust_index_new REAL NOT NULL,
            change REAL NOT NULL,
            cause TEXT,
            field_scores TEXT DEFAULT '{}',
            timestamp TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES rd_experiments(id)
        )
    """)
    
    # Tabela WPX - Gap Filling Lite
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_wpx_gapfill_lite (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            dpp_id TEXT NOT NULL,
            filled_fields TEXT DEFAULT '[]',
            values_filled TEXT DEFAULT '{}',
            uncertainty TEXT DEFAULT '{}',
            context TEXT DEFAULT '{}',
            method TEXT DEFAULT 'lite',
            timestamp TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES rd_experiments(id)
        )
    """)
    
    # Tabela WPX - Compliance Evolution
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_wpx_compliance_evolution (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            dpp_id TEXT NOT NULL,
            espr_score_old REAL NOT NULL,
            espr_score_new REAL NOT NULL,
            cbam_score_old REAL,
            cbam_score_new REAL,
            csrd_score_old REAL NOT NULL,
            csrd_score_new REAL NOT NULL,
            critical_gaps TEXT DEFAULT '[]',
            timestamp TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES rd_experiments(id)
        )
    """)
    
    # Tabela WPX - PredictiveCare (Maintenance events)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rd_wpx_predictivecare (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            machine_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            work_order_number TEXT,
            shi_at_event REAL,
            rul_at_event REAL,
            risk_at_event REAL,
            priority TEXT,
            maintenance_type TEXT,
            failure_prevented INTEGER DEFAULT 0,
            duration_hours REAL,
            metadata_json TEXT DEFAULT '{}',
            timestamp TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES rd_experiments(id)
        )
    """)
    
    conn.commit()
    conn.close()


@contextmanager
def _get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Context manager para conexão à BD."""
    _ensure_db_exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_db_session():
    """
    Context manager para sessão de BD (alias para _get_db_connection).
    
    Usado por módulos externos como reporting.py.
    
    Yields:
        sqlite3.Connection com row_factory configurado
    """
    _ensure_db_exists()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CRUD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_experiment(data: RDExperimentCreate) -> RDExperiment:
    """
    Cria nova experiência.
    
    Returns:
        RDExperiment criado com ID
    """
    now = datetime.now(timezone.utc)
    
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO rd_experiments (wp, name, description, context, parameters, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            data.wp.value,
            data.name,
            data.description,
            json.dumps(data.context),
            json.dumps(data.parameters),
            ExperimentStatus.CREATED.value,
            now.isoformat(),
        ))
        conn.commit()
        
        experiment_id = cursor.lastrowid
    
    logger.info(f"Created experiment {experiment_id}: {data.name} ({data.wp.value})")
    
    return RDExperiment(
        id=experiment_id,
        wp=data.wp,
        name=data.name,
        description=data.description,
        context=data.context,
        parameters=data.parameters,
        status=ExperimentStatus.CREATED,
        created_at=now,
    )


def get_experiment(experiment_id: int) -> Optional[RDExperiment]:
    """Obtém experiência por ID."""
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rd_experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return _row_to_experiment(row)


def list_experiments(
    wp: Optional[WorkPackage] = None,
    status: Optional[ExperimentStatus] = None,
    limit: int = 100,
) -> List[RDExperiment]:
    """
    Lista experiências com filtros opcionais.
    """
    query = "SELECT * FROM rd_experiments WHERE 1=1"
    params = []
    
    if wp:
        query += " AND wp = ?"
        params.append(wp.value)
    
    if status:
        query += " AND status = ?"
        params.append(status.value)
    
    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)
    
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [_row_to_experiment(row) for row in rows]


def update_experiment_status(
    experiment_id: int,
    status: ExperimentStatus,
    summary: Optional[Dict[str, Any]] = None,
    kpis: Optional[Dict[str, Any]] = None,
    conclusion: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Optional[RDExperiment]:
    """
    Atualiza status e resultados de uma experiência.
    """
    now = datetime.now(timezone.utc)
    
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Obter experiência atual para calcular duração
        cursor.execute("SELECT created_at FROM rd_experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        created_at = datetime.fromisoformat(row["created_at"])
        duration_sec = (now - created_at).total_seconds() if status in (ExperimentStatus.FINISHED, ExperimentStatus.FAILED) else None
        finished_at = now.isoformat() if status in (ExperimentStatus.FINISHED, ExperimentStatus.FAILED) else None
        
        # Update
        cursor.execute("""
            UPDATE rd_experiments 
            SET status = ?, updated_at = ?, finished_at = ?, duration_sec = ?,
                summary = COALESCE(?, summary), kpis = COALESCE(?, kpis),
                conclusion = COALESCE(?, conclusion), error_message = COALESCE(?, error_message)
            WHERE id = ?
        """, (
            status.value,
            now.isoformat(),
            finished_at,
            duration_sec,
            json.dumps(summary) if summary else None,
            json.dumps(kpis) if kpis else None,
            conclusion,
            error_message,
            experiment_id,
        ))
        conn.commit()
    
    return get_experiment(experiment_id)


def _row_to_experiment(row: sqlite3.Row) -> RDExperiment:
    """Converte row para RDExperiment."""
    return RDExperiment(
        id=row["id"],
        wp=WorkPackage(row["wp"]),
        name=row["name"],
        description=row["description"],
        context=json.loads(row["context"] or "{}"),
        parameters=json.loads(row["parameters"] or "{}"),
        status=ExperimentStatus(row["status"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None,
        finished_at=datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None,
        duration_sec=row["duration_sec"],
        summary=json.loads(row["summary"] or "{}"),
        kpis=json.loads(row["kpis"] or "{}"),
        conclusion=row["conclusion"],
        error_message=row["error_message"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT LOGGER (Context Manager)
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentLogger:
    """
    Context manager para logging estruturado de experiências.
    
    Uso:
        with ExperimentLogger(WorkPackage.WP1_ROUTING, "test_spt_vs_edd") as exp:
            exp.log_parameter("strategy", "SPT")
            result = run_experiment()
            exp.log_kpi("makespan", result.makespan)
            exp.log_conclusion("SPT reduces makespan by 10%")
    """
    
    def __init__(
        self,
        wp: WorkPackage,
        name: str,
        description: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.wp = wp
        self.name = name
        self.description = description
        self.context = context or {}
        self.parameters: Dict[str, Any] = {}
        self.kpis: Dict[str, Any] = {}
        self.summary: Dict[str, Any] = {}
        self.conclusion: Optional[str] = None
        self._experiment: Optional[RDExperiment] = None
    
    def __enter__(self) -> "ExperimentLogger":
        """Cria experiência ao entrar no context."""
        self._experiment = create_experiment(RDExperimentCreate(
            wp=self.wp,
            name=self.name,
            description=self.description,
            context=self.context,
            parameters=self.parameters,
        ))
        
        # Marcar como running
        update_experiment_status(self._experiment.id, ExperimentStatus.RUNNING)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finaliza experiência ao sair do context."""
        if self._experiment is None:
            return
        
        if exc_type is not None:
            # Houve exceção
            update_experiment_status(
                self._experiment.id,
                ExperimentStatus.FAILED,
                summary=self.summary,
                kpis=self.kpis,
                error_message=str(exc_val),
            )
        else:
            # Sucesso
            update_experiment_status(
                self._experiment.id,
                ExperimentStatus.FINISHED,
                summary=self.summary,
                kpis=self.kpis,
                conclusion=self.conclusion,
            )
    
    @property
    def experiment_id(self) -> Optional[int]:
        """ID da experiência."""
        return self._experiment.id if self._experiment else None
    
    def log_parameter(self, name: str, value: Any) -> None:
        """Regista parâmetro."""
        self.parameters[name] = value
    
    def log_kpi(self, name: str, value: Any) -> None:
        """Regista KPI."""
        self.kpis[name] = value
    
    def log_summary(self, key: str, value: Any) -> None:
        """Regista item no summary."""
        self.summary[key] = value
    
    def log_conclusion(self, conclusion: str) -> None:
        """Regista conclusão."""
        self.conclusion = conclusion


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def log_experiment_event(
    experiment_type: str,
    event_data: Dict[str, Any],
) -> Optional[int]:
    """
    Log a simple experiment event (e.g., trust index evolution).
    
    As specified in Contract D1: Log WPX_TRUST_EVOLUTION events.
    
    Args:
        experiment_type: Type of experiment (e.g., "WPX_TRUST_EVOLUTION")
        event_data: Event data dictionary
    
    Returns:
        Experiment ID if created, None otherwise
    """
    try:
        wp = WorkPackage(experiment_type)
    except ValueError:
        # Unknown work package, skip logging
        logger.debug(f"Unknown experiment type: {experiment_type}, skipping log")
        return None
    
    # Create a simple experiment record
    name = event_data.get("name", f"{experiment_type}_event")
    description = event_data.get("description", f"Event: {experiment_type}")
    
    experiment = create_experiment(RDExperimentCreate(
        wp=wp,
        name=name,
        description=description,
        context=event_data,
        parameters=event_data,
    ))
    
    # Mark as finished immediately (it's an event, not a running experiment)
    update_experiment_status(
        experiment.id,
        ExperimentStatus.FINISHED,
        summary=event_data,
        kpis=event_data,
    )
    
    # For WPX_TRUST_EVOLUTION, save to specific table
    if wp == WorkPackage.WPX_TRUST_EVOLUTION:
        import json
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rd_wpx_trust_evolution (
                    experiment_id, dpp_id, trust_index_old, trust_index_new,
                    change, cause, field_scores, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                event_data.get("dpp_id", ""),
                event_data.get("trust_index_old", 0.0),
                event_data.get("trust_index_new", 0.0),
                event_data.get("change", 0.0),
                event_data.get("cause", "unknown"),
                json.dumps(event_data.get("field_scores", {})),
                event_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ))
            conn.commit()
    
    # For WPX_GAPFILL_LITE, save to specific table
    if wp == WorkPackage.WPX_GAPFILL_LITE:
        import json
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rd_wpx_gapfill_lite (
                    experiment_id, dpp_id, filled_fields, values_filled,
                    uncertainty, context, method, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                event_data.get("dpp_id", ""),
                json.dumps(event_data.get("filled_fields", [])),
                json.dumps(event_data.get("values", {})),
                json.dumps(event_data.get("uncertainty", {})),
                json.dumps(event_data.get("context", {})),
                event_data.get("method", "lite"),
                event_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ))
            conn.commit()
    
    # For WPX_COMPLIANCE_EVOLUTION, save to specific table
    if wp == WorkPackage.WPX_COMPLIANCE_EVOLUTION:
        import json
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rd_wpx_compliance_evolution (
                    experiment_id, dpp_id, espr_score_old, espr_score_new,
                    cbam_score_old, cbam_score_new, csrd_score_old, csrd_score_new,
                    critical_gaps, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                event_data.get("dpp_id", ""),
                event_data.get("espr_score_old", 0.0),
                event_data.get("espr_score_new", 0.0),
                event_data.get("cbam_score_old"),
                event_data.get("cbam_score_new"),
                event_data.get("csrd_score_old", 0.0),
                event_data.get("csrd_score_new", 0.0),
                json.dumps(event_data.get("critical_gaps", [])),
                event_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ))
            conn.commit()
    
    # For WPX_PREDICTIVECARE, save to specific table
    if wp == WorkPackage.WPX_PREDICTIVECARE:
        import json
        with _get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rd_wpx_predictivecare (
                    experiment_id, machine_id, event_type, work_order_number,
                    shi_at_event, rul_at_event, risk_at_event, priority,
                    maintenance_type, failure_prevented, duration_hours,
                    metadata_json, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                event_data.get("machine_id", ""),
                event_data.get("event_type", experiment_type),
                event_data.get("work_order_number"),
                event_data.get("shi"),
                event_data.get("rul"),
                event_data.get("risk_7d"),
                event_data.get("priority"),
                event_data.get("maintenance_type"),
                1 if event_data.get("failure_prevented", False) else 0,
                event_data.get("duration_hours"),
                json.dumps({k: v for k, v in event_data.items() if k not in (
                    "machine_id", "event_type", "work_order_number", "shi", "rul",
                    "risk_7d", "priority", "maintenance_type", "failure_prevented",
                    "duration_hours", "timestamp"
                )}),
                event_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            ))
            conn.commit()
    
    logger.info(f"Logged experiment event: {experiment_type} (ID: {experiment.id})")
    return experiment.id


def get_experiments_summary() -> Dict[str, Any]:
    """
    Obtém resumo de todas as experiências.
    """
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Count por WP
        cursor.execute("""
            SELECT wp, status, COUNT(*) as count
            FROM rd_experiments
            GROUP BY wp, status
        """)
        rows = cursor.fetchall()
        
        by_wp = {}
        for row in rows:
            wp = row["wp"]
            if wp not in by_wp:
                by_wp[wp] = {"total": 0, "by_status": {}}
            by_wp[wp]["by_status"][row["status"]] = row["count"]
            by_wp[wp]["total"] += row["count"]
        
        # Total
        cursor.execute("SELECT COUNT(*) as total FROM rd_experiments")
        total = cursor.fetchone()["total"]
        
        return {
            "total_experiments": total,
            "by_work_package": by_wp,
        }


def delete_experiment(experiment_id: int) -> bool:
    """Delete uma experiência."""
    with _get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rd_experiments WHERE id = ?", (experiment_id,))
        conn.commit()
        return cursor.rowcount > 0

