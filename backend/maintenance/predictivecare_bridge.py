"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PREDICTIVECARE BRIDGE - Geração Automática de Ordens de Manutenção
════════════════════════════════════════════════════════════════════════════════════════════════════

Ponte entre o módulo PredictiveCare (Digital Twin) e o sistema de ordens de manutenção.

Responsabilidades:
1. Avaliar estado de todas as máquinas periodicamente
2. Criar ordens de manutenção automaticamente quando risco é alto
3. Calcular janelas ótimas de manutenção
4. Sincronizar com CMMS externos
5. Log para R&D (WPX_PREDICTIVECARE)

R&D / SIFIDE: WP1 - Digital Twin para manutenção preditiva
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictiveCareBridgeConfig:
    """Configuration for the PredictiveCare bridge."""
    
    # Thresholds for automatic work order creation
    risk_threshold_high: float = 0.30      # Create WO if 7d risk >= 30%
    risk_threshold_critical: float = 0.50  # Create HIGH priority WO if >= 50%
    risk_threshold_emergency: float = 0.70  # Create EMERGENCY WO if >= 70%
    
    shi_threshold_critical: float = 40.0   # Create WO if SHI < 40%
    rul_threshold_hours: float = 168.0     # Create WO if RUL < 7 days
    
    # Maintenance window calculation
    default_maintenance_duration_hours: float = 4.0
    minimum_window_size_hours: float = 2.0
    lookahead_days: int = 7
    
    # CMMS integration
    cmms_enabled: bool = False
    cmms_system: str = ""  # "SAP_PM", "Odoo", etc.
    cmms_webhook_url: str = ""
    
    # R&D logging
    rd_logging_enabled: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class PredictiveCareBridge:
    """
    Bridge between PredictiveCare and Maintenance systems.
    
    Monitors machine health and automatically creates work orders
    when maintenance is needed.
    """
    
    def __init__(
        self,
        config: Optional[PredictiveCareBridgeConfig] = None,
        predictive_care_service=None,
        db_session=None,
    ):
        """
        Initialize the bridge.
        
        Args:
            config: Configuration options
            predictive_care_service: PredictiveCareService instance
            db_session: SQLAlchemy session for persistence
        """
        self.config = config or PredictiveCareBridgeConfig()
        self._pc_service = predictive_care_service
        self._db_session = db_session
        
        # Track created work orders to avoid duplicates
        self._created_wo_cache: Dict[str, datetime] = {}  # machine_id -> last created
        self._cache_ttl_hours = 24  # Don't create duplicate WO within 24h
        
        logger.info("PredictiveCareBridge initialized")
    
    def evaluate_and_create_workorders(self) -> List[Dict[str, Any]]:
        """
        Evaluate all machines and create work orders as needed.
        
        This should be called periodically (e.g., every hour or on demand).
        
        Returns:
            List of created work orders
        """
        from digital_twin.predictive_care import get_predictive_care_service, MachineHealthState
        from maintenance.models import (
            MaintenancePriority, MaintenanceType, MaintenanceStatus, WorkOrderSource
        )
        
        pc_service = self._pc_service or get_predictive_care_service()
        created_orders = []
        
        # Get all machine states
        machines = pc_service.get_all_machines_state()
        
        for state in machines:
            # Check if we should create a work order
            should_create, priority, reason = self._should_create_workorder(state)
            
            if not should_create:
                continue
            
            # Check cache to avoid duplicates
            if self._is_duplicate(state.machine_id):
                logger.debug(f"Skipping duplicate WO for {state.machine_id}")
                continue
            
            # Calculate optimal maintenance window
            window = self.suggest_maintenance_window(state.machine_id)
            
            # Create work order
            wo_data = {
                "work_order_number": self._generate_wo_number(),
                "machine_id": state.machine_id,
                "title": f"PredictiveCare: {reason}",
                "description": self._generate_description(state, reason),
                "priority": priority.value,
                "maintenance_type": MaintenanceType.PREDICTIVE.value,
                "status": MaintenanceStatus.OPEN.value,
                "source": WorkOrderSource.PREDICTIVECARE.value,
                "suggested_start": window.get("window_start") if window else None,
                "suggested_end": window.get("window_end") if window else None,
                "estimated_duration_hours": self.config.default_maintenance_duration_hours,
                "shi_at_creation": state.shi_percent,
                "rul_at_creation": state.rul_hours if state.rul_hours != float('inf') else None,
                "risk_at_creation": state.risk_next_7d,
                "created_at": datetime.now(timezone.utc),
            }
            
            # Persist if database available
            if self._db_session is not None:
                try:
                    from maintenance.models import MaintenanceWorkOrder, SQLALCHEMY_AVAILABLE
                    if SQLALCHEMY_AVAILABLE:
                        wo = MaintenanceWorkOrder(**{
                            k: v for k, v in wo_data.items()
                            if k not in ('created_at',)  # handled by default
                        })
                        self._db_session.add(wo)
                        self._db_session.commit()
                        wo_data["id"] = wo.id
                except Exception as e:
                    logger.error(f"Failed to persist work order: {e}")
            
            created_orders.append(wo_data)
            
            # Update cache
            self._created_wo_cache[state.machine_id] = datetime.now(timezone.utc)
            
            # Log to R&D
            self._log_to_rd(state, wo_data, "work_order_created")
            
            # Send to CMMS if enabled
            if self.config.cmms_enabled:
                self._send_to_cmms(wo_data)
            
            logger.info(f"Created WO {wo_data['work_order_number']} for {state.machine_id}: {reason}")
        
        return created_orders
    
    def suggest_maintenance_window(
        self,
        machine_id: str,
        horizon_days: int = 7,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate optimal maintenance window for a machine.
        
        Considers:
        - Production plan (Gantt) to find low-utilization periods
        - RUL to ensure maintenance happens before failure
        - Technician availability (future)
        
        Args:
            machine_id: Machine identifier
            horizon_days: How far ahead to look
            
        Returns:
            MaintenanceWindowSuggestion or None
        """
        from digital_twin.predictive_care import get_predictive_care_service
        
        pc_service = self._pc_service or get_predictive_care_service()
        state = pc_service.get_machine_state(machine_id)
        
        now = datetime.now(timezone.utc)
        duration = self.config.default_maintenance_duration_hours
        
        # Get RUL-based deadline
        if state.rul_hours != float('inf') and state.rul_hours > 0:
            deadline = now + timedelta(hours=state.rul_hours * 0.8)  # 80% of RUL
        else:
            deadline = now + timedelta(days=horizon_days)
        
        # Try to find window from production plan
        windows = self._find_plan_windows(machine_id, now, deadline, duration)
        
        if windows:
            best_window = windows[0]
            return {
                "machine_id": machine_id,
                "window_start": best_window["start"],
                "window_end": best_window["end"],
                "duration_hours": duration,
                "impact_on_plan": best_window.get("impact", "low"),
                "plan_delay_hours": best_window.get("delay_hours", 0),
                "risk_if_postponed": self._calculate_postponement_risk(state, best_window["start"]),
                "reason": f"Best window before RUL deadline ({state.rul_hours:.0f}h)",
                "alternative_windows": windows[1:3],  # Next 2 alternatives
            }
        
        # Fallback: suggest immediate or next available
        suggested_start = now + timedelta(hours=4)  # 4 hours from now
        return {
            "machine_id": machine_id,
            "window_start": suggested_start,
            "window_end": suggested_start + timedelta(hours=duration),
            "duration_hours": duration,
            "impact_on_plan": "medium",
            "plan_delay_hours": duration,
            "risk_if_postponed": state.risk_next_7d,
            "reason": "No optimal window found - immediate maintenance recommended",
            "alternative_windows": [],
        }
    
    def sync_from_cmms(self) -> Dict[str, int]:
        """
        Sync work orders from external CMMS.
        
        Returns:
            Dict with sync statistics
        """
        if not self.config.cmms_enabled:
            return {"status": "disabled"}
        
        # TODO: Implement actual CMMS integration
        # This is a stub for future implementation
        
        logger.info("CMMS sync: stub implementation")
        
        return {
            "status": "ok",
            "created": 0,
            "updated": 0,
            "errors": 0,
        }
    
    def complete_workorder(
        self,
        work_order_id: int,
        resolution_notes: str,
        failure_prevented: bool = True,
        parts_replaced: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Mark a work order as completed.
        
        Also logs to R&D for effectiveness tracking.
        
        Args:
            work_order_id: Work order ID
            resolution_notes: Description of work performed
            failure_prevented: Whether this prevented a failure
            parts_replaced: List of parts replaced
            
        Returns:
            Updated work order
        """
        from maintenance.models import MaintenanceStatus, SQLALCHEMY_AVAILABLE
        import json
        
        if not SQLALCHEMY_AVAILABLE or self._db_session is None:
            return {"error": "Database not available"}
        
        from maintenance.models import MaintenanceWorkOrder, MaintenanceHistory
        
        try:
            wo = self._db_session.query(MaintenanceWorkOrder).filter_by(id=work_order_id).first()
            
            if not wo:
                return {"error": "Work order not found"}
            
            # Update work order
            wo.status = MaintenanceStatus.COMPLETED.value
            wo.actual_end = datetime.now(timezone.utc)
            wo.resolution_notes = resolution_notes
            
            if wo.actual_start:
                wo.actual_duration_hours = (wo.actual_end - wo.actual_start).total_seconds() / 3600
            
            # Create history record
            history = MaintenanceHistory(
                work_order_id=wo.id,
                machine_id=wo.machine_id,
                maintenance_type=wo.maintenance_type,
                priority=wo.priority,
                started_at=wo.actual_start or wo.created_at,
                completed_at=wo.actual_end,
                duration_hours=wo.actual_duration_hours or wo.estimated_duration_hours or 0,
                shi_before=wo.shi_at_creation,
                failure_prevented=failure_prevented,
                was_planned=wo.source != "CORRECTIVE",
                was_successful=True,
                work_performed=resolution_notes,
                parts_replaced_json=json.dumps(parts_replaced) if parts_replaced else None,
            )
            
            self._db_session.add(history)
            self._db_session.commit()
            
            # Log to R&D
            self._log_to_rd_completion(wo, history, failure_prevented)
            
            return wo.to_dict()
            
        except Exception as e:
            logger.error(f"Failed to complete work order: {e}")
            self._db_session.rollback()
            return {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIVATE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _should_create_workorder(self, state) -> Tuple[bool, Any, str]:
        """
        Determine if a work order should be created for a machine.
        
        Returns:
            (should_create, priority, reason)
        """
        from maintenance.models import MaintenancePriority
        from digital_twin.predictive_care import MachineHealthState
        
        cfg = self.config
        
        # Check emergency conditions
        if state.risk_next_7d >= cfg.risk_threshold_emergency:
            return True, MaintenancePriority.EMERGENCY, f"Emergency risk: {state.risk_next_7d*100:.0f}%"
        
        if state.state == MachineHealthState.EMERGENCY:
            return True, MaintenancePriority.EMERGENCY, f"Emergency state (SHI={state.shi_percent:.0f}%)"
        
        # Check critical conditions
        if state.risk_next_7d >= cfg.risk_threshold_critical:
            return True, MaintenancePriority.CRITICAL, f"Critical risk: {state.risk_next_7d*100:.0f}%"
        
        if state.shi_percent < cfg.shi_threshold_critical:
            return True, MaintenancePriority.HIGH, f"Low SHI: {state.shi_percent:.0f}%"
        
        if state.rul_hours != float('inf') and state.rul_hours < cfg.rul_threshold_hours:
            return True, MaintenancePriority.HIGH, f"Low RUL: {state.rul_hours:.0f}h"
        
        # Check high risk
        if state.risk_next_7d >= cfg.risk_threshold_high:
            return True, MaintenancePriority.MEDIUM, f"High risk: {state.risk_next_7d*100:.0f}%"
        
        return False, None, ""
    
    def _is_duplicate(self, machine_id: str) -> bool:
        """Check if we recently created a WO for this machine."""
        if machine_id not in self._created_wo_cache:
            return False
        
        last_created = self._created_wo_cache[machine_id]
        hours_since = (datetime.now(timezone.utc) - last_created).total_seconds() / 3600
        
        return hours_since < self._cache_ttl_hours
    
    def _generate_wo_number(self) -> str:
        """Generate unique work order number."""
        now = datetime.now(timezone.utc)
        short_uuid = uuid.uuid4().hex[:6].upper()
        return f"WO-{now.strftime('%Y%m%d')}-{short_uuid}"
    
    def _generate_description(self, state, reason: str) -> str:
        """Generate detailed description for work order."""
        lines = [
            f"Automatically generated by PredictiveCare",
            f"",
            f"Machine: {state.machine_id}",
            f"Trigger: {reason}",
            f"",
            f"Current Metrics:",
            f"- Health Index (SHI): {state.shi_percent:.1f}%",
            f"- Remaining Useful Life (RUL): {state.rul_hours:.0f}h" if state.rul_hours != float('inf') else "- RUL: N/A",
            f"- 7-day Failure Risk: {state.risk_next_7d*100:.1f}%",
            f"- Anomaly Score: {state.anomaly_score:.2f}",
            f"",
            f"Top Contributing Sensors:",
        ]
        
        for c in state.top_contributors[:3]:
            lines.append(f"- {c.sensor_type}: {c.deviation_percent:.1f}% deviation")
        
        return "\n".join(lines)
    
    def _find_plan_windows(
        self,
        machine_id: str,
        start: datetime,
        end: datetime,
        duration: float,
    ) -> List[Dict[str, Any]]:
        """
        Find available maintenance windows in the production plan.
        
        TODO: Integrate with actual scheduling data
        """
        # Stub: return some default windows
        windows = []
        
        # Check for early morning slots (less production)
        current = start.replace(hour=6, minute=0, second=0, microsecond=0)
        if current < start:
            current += timedelta(days=1)
        
        while current < end:
            # Suggest early morning windows
            windows.append({
                "start": current,
                "end": current + timedelta(hours=duration),
                "impact": "low",
                "delay_hours": 0,
            })
            current += timedelta(days=1)
        
        return windows[:5]  # Return up to 5 windows
    
    def _calculate_postponement_risk(self, state, window_start: datetime) -> float:
        """Calculate increased risk if maintenance is postponed to window."""
        hours_until_window = (window_start - datetime.now(timezone.utc)).total_seconds() / 3600
        
        if hours_until_window <= 0:
            return state.risk_next_7d
        
        # Risk increases roughly linearly with time
        # This is simplified - real calculation would use survival curve
        daily_risk_increase = 0.05  # 5% per day
        additional_risk = (hours_until_window / 24) * daily_risk_increase
        
        return min(1.0, state.risk_next_7d + additional_risk)
    
    def _send_to_cmms(self, wo_data: Dict[str, Any]):
        """Send work order to external CMMS."""
        if not self.config.cmms_webhook_url:
            return
        
        # TODO: Implement actual CMMS integration
        logger.info(f"CMMS stub: would send WO {wo_data['work_order_number']} to {self.config.cmms_system}")
    
    def _log_to_rd(self, state, wo_data: Dict[str, Any], event_type: str):
        """Log event to R&D module."""
        if not self.config.rd_logging_enabled:
            return
        
        try:
            from rd.experiments_core import log_experiment_event, WorkPackage
            
            log_experiment_event(
                work_package=WorkPackage.WPX_PREDICTIVECARE,
                experiment_type=event_type,
                payload={
                    "machine_id": state.machine_id,
                    "work_order_number": wo_data["work_order_number"],
                    "shi": state.shi_percent,
                    "rul": state.rul_hours if state.rul_hours != float('inf') else None,
                    "risk_7d": state.risk_next_7d,
                    "priority": wo_data["priority"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            logger.debug(f"R&D logging not available: {e}")
    
    def _log_to_rd_completion(self, wo, history, failure_prevented: bool):
        """Log work order completion to R&D."""
        if not self.config.rd_logging_enabled:
            return
        
        try:
            from rd.experiments_core import log_experiment_event, WorkPackage
            
            log_experiment_event(
                work_package=WorkPackage.WPX_PREDICTIVECARE,
                experiment_type="work_order_completed",
                payload={
                    "machine_id": wo.machine_id,
                    "work_order_number": wo.work_order_number,
                    "failure_prevented": failure_prevented,
                    "duration_hours": history.duration_hours,
                    "shi_before": history.shi_before,
                    "maintenance_type": wo.maintenance_type,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as e:
            logger.debug(f"R&D logging not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

_bridge_instance: Optional[PredictiveCareBridge] = None


def get_predictivecare_bridge() -> PredictiveCareBridge:
    """Get or create the PredictiveCare bridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = PredictiveCareBridge()
    return _bridge_instance


