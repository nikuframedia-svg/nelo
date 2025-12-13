"""
Actions Engine for Nikufra Production OS

Industry 5.0 Human-Centric Execution Layer:
- System proposes actions (suggestions, commands, what-if results)
- Human approves or rejects
- Only after approval, changes are applied to the plan
- NEVER executes directly on machines/ERP

R&D: SIFIDE compliance - all execution is auditable and human-controlled.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any

import pandas as pd


# -------------------------
# Types
# -------------------------

ActionType = Literal[
    "SET_MACHINE_DOWN",
    "SET_MACHINE_UP",
    "CHANGE_ROUTE",
    "MOVE_OPERATION",
    "SET_VIP_ARTICLE",
    "CHANGE_HORIZON",
    "ADD_OVERTIME",
    "ADD_ORDER",
]

ActionStatus = Literal["PENDING", "APPROVED", "REJECTED", "APPLIED"]


@dataclass
class Action:
    """
    Represents a proposed action in the system.
    
    Lifecycle:
    1. Created with status="PENDING"
    2. Human reviews and sets status="APPROVED" or "REJECTED"
    3. If approved, system applies changes and sets status="APPLIED"
    """
    id: str
    type: ActionType
    payload: Dict[str, Any]
    source: str  # e.g. "suggestion_engine", "chat_command", "what_if", "manual"
    created_at: str
    status: ActionStatus = "PENDING"
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    applied_at: Optional[str] = None
    notes: Optional[str] = None
    
    # Human-readable description
    description: Optional[str] = None
    
    # Impact preview (optional)
    expected_impact: Optional[Dict[str, Any]] = None


def create_action(
    action_type: ActionType,
    payload: Dict[str, Any],
    source: str,
    description: Optional[str] = None,
    expected_impact: Optional[Dict[str, Any]] = None,
) -> Action:
    """Factory function to create a new Action."""
    return Action(
        id=str(uuid.uuid4())[:8],
        type=action_type,
        payload=payload,
        source=source,
        created_at=datetime.utcnow().isoformat() + "Z",
        status="PENDING",
        description=description or generate_action_description(action_type, payload),
        expected_impact=expected_impact,
    )


def generate_action_description(action_type: ActionType, payload: Dict[str, Any]) -> str:
    """Generate a human-readable description for an action."""
    
    if action_type == "SET_MACHINE_DOWN":
        machine = payload.get("machine_id", "?")
        start = payload.get("start_time", "?")
        end = payload.get("end_time", "?")
        reason = payload.get("reason", "manutenção")
        return f"Colocar {machine} offline de {start} a {end} ({reason})"
    
    elif action_type == "SET_MACHINE_UP":
        machine = payload.get("machine_id", "?")
        return f"Reativar {machine} (remover paragem programada)"
    
    elif action_type == "CHANGE_ROUTE":
        order = payload.get("order_id", "?")
        article = payload.get("article_id", "?")
        new_route = payload.get("new_route_label", "?")
        return f"Alterar rota de {order} ({article}) para Rota {new_route}"
    
    elif action_type == "MOVE_OPERATION":
        op_code = payload.get("op_code", "?")
        from_machine = payload.get("from_machine_id", "?")
        to_machine = payload.get("to_machine_id", "?")
        return f"Mover {op_code} de {from_machine} para {to_machine}"
    
    elif action_type == "SET_VIP_ARTICLE":
        article = payload.get("article_id", "?")
        return f"Definir {article} como VIP (prioridade máxima)"
    
    elif action_type == "CHANGE_HORIZON":
        days = payload.get("horizon_days", "?")
        return f"Alterar horizonte de planeamento para {days} dias"
    
    elif action_type == "ADD_OVERTIME":
        machine = payload.get("machine_id", "?")
        hours = payload.get("extra_hours", "?")
        date = payload.get("date", "?")
        return f"Adicionar {hours}h extra em {machine} ({date})"
    
    elif action_type == "ADD_ORDER":
        order_id = payload.get("order_id", "?")
        article = payload.get("article_id", "?")
        qty = payload.get("qty", "?")
        return f"Adicionar ordem {order_id}: {qty}x {article}"
    
    return f"Ação {action_type}"


# -------------------------
# Action Store (In-Memory + File Persistence)
# -------------------------

class ActionStore:
    """
    Simple action store with file persistence.
    
    For MVP, uses JSON file. Can be upgraded to database later.
    """
    
    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or Path(__file__).parent.parent / "data" / "actions.json"
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._actions: Dict[str, Action] = {}
        self._load()
    
    def _load(self) -> None:
        """Load actions from file."""
        if self.store_path.exists():
            try:
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        action = Action(**item)
                        self._actions[action.id] = action
            except (json.JSONDecodeError, TypeError):
                self._actions = {}
    
    def _save(self) -> None:
        """Save actions to file."""
        data = [asdict(a) for a in self._actions.values()]
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def list_actions(self, status: Optional[str] = None) -> List[Action]:
        """List all actions, optionally filtered by status."""
        actions = list(self._actions.values())
        if status:
            actions = [a for a in actions if a.status == status]
        # Sort by created_at descending (newest first)
        actions.sort(key=lambda a: a.created_at, reverse=True)
        return actions
    
    def get_action(self, action_id: str) -> Optional[Action]:
        """Get a specific action by ID."""
        return self._actions.get(action_id)
    
    def add_action(self, action: Action) -> Action:
        """Add a new action to the store."""
        self._actions[action.id] = action
        self._save()
        return action
    
    def update_action_status(
        self,
        action_id: str,
        status: ActionStatus,
        notes: Optional[str] = None,
        approved_by: Optional[str] = None,
    ) -> Optional[Action]:
        """Update the status of an action."""
        action = self._actions.get(action_id)
        if not action:
            return None
        
        action.status = status
        if notes:
            action.notes = notes
        
        if status == "APPROVED":
            action.approved_by = approved_by
            action.approved_at = datetime.utcnow().isoformat() + "Z"
        elif status == "APPLIED":
            action.applied_at = datetime.utcnow().isoformat() + "Z"
        elif status == "REJECTED":
            action.approved_by = approved_by
            action.approved_at = datetime.utcnow().isoformat() + "Z"
        
        self._save()
        return action
    
    def clear_all(self) -> None:
        """Clear all actions (for testing)."""
        self._actions = {}
        self._save()


# Global store instance
_action_store: Optional[ActionStore] = None


def get_action_store() -> ActionStore:
    """Get the global action store instance."""
    global _action_store
    if _action_store is None:
        _action_store = ActionStore()
    return _action_store


# -------------------------
# Action Application (Plan Modification)
# -------------------------

def apply_action_to_plan(action: Action, data: "DataBundle") -> "DataBundle":
    """
    Apply an approved action to the DataBundle.
    
    This modifies the planning DATA (not external systems).
    After this, the scheduler should be re-run to generate a new plan.
    
    Args:
        action: The approved action to apply
        data: The current DataBundle
    
    Returns:
        Modified DataBundle
    """
    from data_loader import DataBundle
    
    action_type = action.type
    payload = action.payload
    
    if action_type == "SET_MACHINE_DOWN":
        return _apply_machine_down(data, payload)
    
    elif action_type == "SET_MACHINE_UP":
        return _apply_machine_up(data, payload)
    
    elif action_type == "MOVE_OPERATION":
        return _apply_move_operation(data, payload)
    
    elif action_type == "SET_VIP_ARTICLE":
        return _apply_vip_article(data, payload)
    
    elif action_type == "CHANGE_ROUTE":
        return _apply_change_route(data, payload)
    
    elif action_type == "ADD_ORDER":
        return _apply_add_order(data, payload)
    
    elif action_type == "ADD_OVERTIME":
        return _apply_add_overtime(data, payload)
    
    # For unimplemented actions, return data unchanged
    return data


def _apply_machine_down(data: "DataBundle", payload: Dict[str, Any]) -> "DataBundle":
    """
    Add a downtime entry for a machine.
    
    Payload:
        - machine_id: str
        - start_time: str (ISO datetime)
        - end_time: str (ISO datetime)
        - reason: str (optional)
    """
    machine_id = payload.get("machine_id")
    start_time = payload.get("start_time")
    end_time = payload.get("end_time")
    reason = payload.get("reason", "Paragem manual")
    
    if not all([machine_id, start_time, end_time]):
        return data
    
    # Create new downtime entry
    new_downtime = pd.DataFrame([{
        "machine_id": machine_id,
        "downtime_id": f"DT-ACTION-{uuid.uuid4().hex[:6]}",
        "down_start": pd.to_datetime(start_time),
        "down_end": pd.to_datetime(end_time),
        "reason": reason,
        "planned": True,
    }])
    
    # Append to existing downtime
    if hasattr(data, 'downtime') and data.downtime is not None and not data.downtime.empty:
        data.downtime = pd.concat([data.downtime, new_downtime], ignore_index=True)
    else:
        data.downtime = new_downtime
    
    return data


def _apply_machine_up(data: "DataBundle", payload: Dict[str, Any]) -> "DataBundle":
    """
    Remove or reduce downtime for a machine.
    
    Payload:
        - machine_id: str
        - downtime_id: str (optional, specific downtime to remove)
    """
    machine_id = payload.get("machine_id")
    downtime_id = payload.get("downtime_id")
    
    if not machine_id:
        return data
    
    if not hasattr(data, 'downtime') or data.downtime is None or data.downtime.empty:
        return data
    
    if downtime_id:
        # Remove specific downtime
        data.downtime = data.downtime[data.downtime["downtime_id"] != downtime_id]
    else:
        # Remove all future downtime for this machine
        now = datetime.now()
        mask = ~(
            (data.downtime["machine_id"] == machine_id) & 
            (pd.to_datetime(data.downtime["down_start"]) > now)
        )
        data.downtime = data.downtime[mask]
    
    return data


def _apply_move_operation(data: "DataBundle", payload: Dict[str, Any]) -> "DataBundle":
    """
    Mark an operation to use a different machine.
    
    This modifies the routing data so the scheduler will use the new machine.
    
    Payload:
        - article_id: str
        - op_code: str
        - from_machine_id: str
        - to_machine_id: str
    """
    article_id = payload.get("article_id")
    op_code = payload.get("op_code")
    to_machine = payload.get("to_machine_id")
    
    if not all([article_id, op_code, to_machine]):
        return data
    
    if not hasattr(data, 'routing') or data.routing is None:
        return data
    
    # Update routing to use new machine
    mask = (data.routing["article_id"] == article_id) & (data.routing["op_code"] == op_code)
    data.routing.loc[mask, "primary_machine_id"] = to_machine
    
    return data


def _apply_vip_article(data: "DataBundle", payload: Dict[str, Any]) -> "DataBundle":
    """
    Set an article as VIP (raise priority of all its orders).
    
    Payload:
        - article_id: str
        - priority_value: int (optional, default 999)
    """
    article_id = payload.get("article_id")
    priority_value = payload.get("priority_value", 999)
    
    if not article_id:
        return data
    
    if not hasattr(data, 'orders') or data.orders is None:
        return data
    
    # Ensure priority column exists
    if "priority" not in data.orders.columns:
        data.orders["priority"] = 1
    
    # Set high priority for all orders with this article
    mask = data.orders["article_id"] == article_id
    data.orders.loc[mask, "priority"] = priority_value
    
    return data


def _apply_change_route(data: "DataBundle", payload: Dict[str, Any]) -> "DataBundle":
    """
    Change the preferred route for an article.
    
    Payload:
        - article_id: str
        - new_route_label: str
    """
    article_id = payload.get("article_id")
    new_route = payload.get("new_route_label")
    
    # This is handled at scheduler level by passing route preference
    # For now, we mark the data with a preference
    if not hasattr(data, '_route_preferences'):
        data._route_preferences = {}
    
    data._route_preferences[article_id] = new_route
    
    return data


def _apply_add_order(data: "DataBundle", payload: Dict[str, Any]) -> "DataBundle":
    """
    Add a new order to the orders DataFrame.
    
    Payload:
        - order_id: str
        - article_id: str
        - qty: int
        - due_date: str (ISO date)
        - priority: int (optional)
    """
    order_id = payload.get("order_id")
    article_id = payload.get("article_id")
    qty = payload.get("qty")
    due_date = payload.get("due_date")
    priority = payload.get("priority", 1)
    
    if not all([order_id, article_id, qty]):
        return data
    
    new_order = pd.DataFrame([{
        "order_id": order_id,
        "article_id": article_id,
        "qty": qty,
        "due_date": pd.to_datetime(due_date) if due_date else None,
        "priority": priority,
    }])
    
    if hasattr(data, 'orders') and data.orders is not None:
        data.orders = pd.concat([data.orders, new_order], ignore_index=True)
    else:
        data.orders = new_order
    
    return data


def _apply_add_overtime(data: "DataBundle", payload: Dict[str, Any]) -> "DataBundle":
    """
    Add overtime shift for a machine.
    
    Payload:
        - machine_id: str
        - date: str (ISO date)
        - extra_hours: float
        - start_hour: int (optional, default 18)
    """
    machine_id = payload.get("machine_id")
    date = payload.get("date")
    extra_hours = payload.get("extra_hours", 2)
    start_hour = payload.get("start_hour", 18)
    
    if not all([machine_id, date]):
        return data
    
    # Create overtime shift entry
    shift_date = pd.to_datetime(date).date()
    start_time = datetime.combine(shift_date, datetime.min.time().replace(hour=start_hour))
    end_time = start_time + pd.Timedelta(hours=extra_hours)
    
    new_shift = pd.DataFrame([{
        "machine_id": machine_id,
        "shift_id": f"SHIFT-OT-{uuid.uuid4().hex[:6]}",
        "shift_date": shift_date,
        "start_time": start_time,
        "end_time": end_time,
        "shift_type": "overtime",
    }])
    
    if hasattr(data, 'shifts') and data.shifts is not None and not data.shifts.empty:
        data.shifts = pd.concat([data.shifts, new_shift], ignore_index=True)
    else:
        data.shifts = new_shift
    
    return data


# -------------------------
# High-Level Functions
# -------------------------

def propose_action(
    action_type: ActionType,
    payload: Dict[str, Any],
    source: str,
    description: Optional[str] = None,
    expected_impact: Optional[Dict[str, Any]] = None,
) -> Action:
    """
    Propose a new action (adds to pending queue).
    
    The action will need human approval before being applied.
    """
    action = create_action(action_type, payload, source, description, expected_impact)
    store = get_action_store()
    return store.add_action(action)


def approve_action(
    action_id: str,
    approved_by: str,
    notes: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Approve an action and apply it to the plan.
    
    Returns:
        Dict with result including new KPIs, or None if action not found.
    """
    from data_loader import load_dataset
    from scheduler import build_plan, compute_kpis, save_plan_to_csv
    
    store = get_action_store()
    action = store.get_action(action_id)
    
    if not action:
        return None
    
    if action.status != "PENDING":
        return {"error": f"Ação não está pendente (status: {action.status})"}
    
    # Update to approved
    store.update_action_status(action_id, "APPROVED", notes, approved_by)
    
    # Load data and apply action
    data = load_dataset()
    modified_data = apply_action_to_plan(action, data)
    
    # Rebuild plan
    new_plan = build_plan(modified_data, mode="NORMAL")
    save_plan_to_csv(new_plan)
    
    # Compute new KPIs
    kpis = compute_kpis(new_plan, modified_data.orders)
    
    # Update to applied
    store.update_action_status(action_id, "APPLIED")
    
    return {
        "action_id": action_id,
        "status": "APPLIED",
        "new_plan_operations": len(new_plan),
        "kpis": kpis,
        "message": f"Ação aplicada: {action.description}",
    }


def reject_action(
    action_id: str,
    rejected_by: str,
    reason: Optional[str] = None,
) -> Optional[Action]:
    """
    Reject an action.
    
    Returns:
        Updated Action or None if not found.
    """
    store = get_action_store()
    action = store.get_action(action_id)
    
    if not action:
        return None
    
    if action.status != "PENDING":
        return action  # Already processed
    
    return store.update_action_status(action_id, "REJECTED", reason, rejected_by)


def get_pending_actions() -> List[Action]:
    """Get all pending actions."""
    store = get_action_store()
    return store.list_actions(status="PENDING")


def get_action_history(limit: int = 50) -> List[Action]:
    """Get recent action history (all statuses)."""
    store = get_action_store()
    actions = store.list_actions()
    return actions[:limit]


# -------------------------
# Integration Helpers
# -------------------------

def create_action_from_suggestion(suggestion: Dict[str, Any]) -> Action:
    """
    Create an action from a suggestion_engine suggestion.
    
    Maps suggestion types to action types.
    """
    stype = suggestion.get("type", "")
    
    if stype == "overload_reduction":
        return propose_action(
            action_type="MOVE_OPERATION",
            payload={
                "article_id": suggestion.get("article_id"),
                "op_code": suggestion.get("candidate_op"),
                "from_machine_id": suggestion.get("machine"),
                "to_machine_id": suggestion.get("alternative_machine"),
            },
            source="suggestion_engine",
            description=suggestion.get("formatted_pt", suggestion.get("reason")),
            expected_impact={"gain_hours": suggestion.get("expected_gain_h", 0)},
        )
    
    elif stype == "idle_gap":
        # Idle gaps are informational, not directly actionable
        # But we could propose adding work to that gap
        return propose_action(
            action_type="ADD_OVERTIME",
            payload={
                "machine_id": suggestion.get("machine"),
                "date": suggestion.get("gap_start", "").split("T")[0] if suggestion.get("gap_start") else None,
                "extra_hours": suggestion.get("gap_min", 0) / 60,
            },
            source="suggestion_engine",
            description=f"Aproveitar gap ocioso em {suggestion.get('machine')}",
        )
    
    return None


def create_action_from_command(parsed_command: Dict[str, Any]) -> Optional[Action]:
    """
    Create an action from a parsed chat command.
    
    Maps command types to action types.
    """
    cmd_type = parsed_command.get("command_type", "")
    entities = parsed_command.get("entities", {})
    
    if cmd_type == "machine_downtime":
        return propose_action(
            action_type="SET_MACHINE_DOWN",
            payload={
                "machine_id": entities.get("machine"),
                "start_time": entities.get("start"),
                "end_time": entities.get("end"),
                "reason": "Comando via chat",
            },
            source="chat_command",
            description=parsed_command.get("suggested_action"),
        )
    
    elif cmd_type == "plan_priority":
        # VIP prioritization
        priority = entities.get("priority", "").upper()
        if priority == "VIP":
            return propose_action(
                action_type="SET_VIP_ARTICLE",
                payload={"article_id": entities.get("article_id", "ALL_VIP")},
                source="chat_command",
                description=parsed_command.get("suggested_action"),
            )
    
    return None



