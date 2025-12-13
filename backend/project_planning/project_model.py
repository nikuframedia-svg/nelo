"""
═══════════════════════════════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — PROJECT MODEL
═══════════════════════════════════════════════════════════════════════════════════════════════════════

Data model for projects in production planning.

DEFINITION
══════════

A PROJECT is a logical grouping of production orders that share:
- Common delivery commitment (client promise)
- Resource budget (allocated machine-hours)
- Priority weight (importance relative to other projects)

Mathematical Notation:
─────────────────────

Let:
    P = {p₁, p₂, ..., pₙ} be the set of projects
    O = {o₁, o₂, ..., oₘ} be the set of orders
    
    O(p) ⊆ O        Orders belonging to project p
    d_p             Due date of project p
    w_p             Priority weight of project p
    B_p             Budget (machine-hours) for project p

AGGREGATION MODES
─────────────────

Projects can be built from orders using:

1. EXPLICIT: Orders have a project_id column
   O(p) = {o ∈ O : project_id(o) = p}

2. BY_CLIENT: Group by customer
   O(p) = {o ∈ O : client_id(o) = p}

3. BY_ARTICLE_FAMILY: Group by product family
   O(p) = {o ∈ O : family(article(o)) = p}

4. BY_DUE_DATE: Group by delivery window
   O(p) = {o ∈ O : due_date(o) ∈ [start_p, end_p]}

R&D / SIFIDE: WP5 - Project Planning
────────────────────────────────────
- Experiment E5.1: Compare project-level vs order-level OTD
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# ENUMS
# ════════════════════════════════════════════════════════════════════════════════════════════════════

class ProjectStatus(str, Enum):
    """Project lifecycle status."""
    PLANNING = "planning"           # Orders not yet scheduled
    IN_PROGRESS = "in_progress"     # Some operations started
    AT_RISK = "at_risk"             # Behind schedule
    COMPLETED = "completed"         # All orders delivered
    DELAYED = "delayed"             # Past due date, not complete


class AggregationMode(str, Enum):
    """How to group orders into projects."""
    EXPLICIT = "explicit"           # Use project_id column
    BY_CLIENT = "by_client"         # Group by client_id
    BY_ARTICLE_FAMILY = "by_family" # Group by article family
    BY_DUE_WEEK = "by_due_week"     # Group by due date week


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# PROJECT DATA MODEL
# ════════════════════════════════════════════════════════════════════════════════════════════════════

@dataclass
class Project:
    """
    Represents a production project.
    
    A project is a logical grouping of orders with:
    - Common delivery commitment
    - Shared priority
    - Resource budget
    
    Attributes:
        project_id: Unique identifier
        name: Human-readable name
        client: Customer name (optional)
        order_ids: Set of orders belonging to this project
        due_date: Promised delivery date
        budget_hours: Allocated machine-hours (optional)
        priority_weight: Importance weight (default 1.0)
        status: Current project status
        metadata: Additional attributes
    
    Mathematical Properties:
    ───────────────────────
    - |O(p)| = len(order_ids) = number of orders
    - d_p = due_date
    - w_p = priority_weight
    - B_p = budget_hours
    """
    project_id: str
    name: str
    client: Optional[str] = None
    order_ids: Set[str] = field(default_factory=set)
    due_date: Optional[datetime] = None
    budget_hours: Optional[float] = None
    priority_weight: float = 1.0
    status: ProjectStatus = ProjectStatus.PLANNING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed fields (populated by load/kpi engines)
    min_start: Optional[datetime] = None
    max_end: Optional[datetime] = None
    total_load_hours: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.order_ids, list):
            self.order_ids = set(self.order_ids)
    
    @property
    def num_orders(self) -> int:
        """Number of orders in this project."""
        return len(self.order_ids)
    
    @property
    def has_due_date(self) -> bool:
        """Whether project has a defined due date."""
        return self.due_date is not None
    
    @property
    def is_late(self) -> bool:
        """Whether project is past due date without completion."""
        if not self.due_date:
            return False
        if self.status == ProjectStatus.COMPLETED:
            return False
        return datetime.now() > self.due_date
    
    def add_order(self, order_id: str) -> None:
        """Add an order to this project."""
        self.order_ids.add(order_id)
    
    def remove_order(self, order_id: str) -> None:
        """Remove an order from this project."""
        self.order_ids.discard(order_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'project_id': self.project_id,
            'name': self.name,
            'client': self.client,
            'order_ids': list(self.order_ids),
            'num_orders': self.num_orders,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'budget_hours': self.budget_hours,
            'priority_weight': self.priority_weight,
            'status': self.status.value,
            'min_start': self.min_start.isoformat() if self.min_start else None,
            'max_end': self.max_end.isoformat() if self.max_end else None,
            'total_load_hours': round(self.total_load_hours, 2),
            'is_late': self.is_late,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Project':
        """Deserialize from dictionary."""
        return cls(
            project_id=data['project_id'],
            name=data['name'],
            client=data.get('client'),
            order_ids=set(data.get('order_ids', [])),
            due_date=datetime.fromisoformat(data['due_date']) if data.get('due_date') else None,
            budget_hours=data.get('budget_hours'),
            priority_weight=data.get('priority_weight', 1.0),
            status=ProjectStatus(data.get('status', 'planning')),
            metadata=data.get('metadata', {}),
        )


# ════════════════════════════════════════════════════════════════════════════════════════════════════
# PROJECT BUILDER
# ════════════════════════════════════════════════════════════════════════════════════════════════════

def group_orders_by_project(
    orders_df: pd.DataFrame,
    mode: AggregationMode = AggregationMode.EXPLICIT,
    project_col: str = 'project_id',
    client_col: str = 'client_id',
    article_col: str = 'article_id',
    due_col: str = 'due_date'
) -> Dict[str, List[str]]:
    """
    Group orders into projects based on aggregation mode.
    
    Mathematical Definition:
    ───────────────────────
    Returns a mapping π: P → 2^O
    where π(p) = O(p) = set of orders belonging to project p
    
    Args:
        orders_df: DataFrame with order data
        mode: Aggregation strategy
        project_col: Column for explicit project ID
        client_col: Column for client grouping
        article_col: Column for article family grouping
        due_col: Column for due date grouping
    
    Returns:
        Dict mapping project_id -> list of order_ids
    """
    if orders_df.empty:
        return {}
    
    order_id_col = 'order_id'
    if order_id_col not in orders_df.columns:
        # Try alternative names
        for alt in ['id', 'OrderID', 'order']:
            if alt in orders_df.columns:
                order_id_col = alt
                break
    
    groups: Dict[str, List[str]] = {}
    
    if mode == AggregationMode.EXPLICIT:
        if project_col in orders_df.columns:
            for proj_id, group in orders_df.groupby(project_col):
                if pd.notna(proj_id):
                    groups[str(proj_id)] = list(group[order_id_col].astype(str))
        else:
            # Fallback: each order is its own "project"
            logger.warning(f"Column '{project_col}' not found. Treating each order as separate project.")
            for _, row in orders_df.iterrows():
                oid = str(row[order_id_col])
                groups[f"PROJ-{oid}"] = [oid]
    
    elif mode == AggregationMode.BY_CLIENT:
        if client_col in orders_df.columns:
            for client, group in orders_df.groupby(client_col):
                if pd.notna(client):
                    groups[f"CLIENT-{client}"] = list(group[order_id_col].astype(str))
        else:
            logger.warning(f"Column '{client_col}' not found.")
    
    elif mode == AggregationMode.BY_ARTICLE_FAMILY:
        if article_col in orders_df.columns:
            # Extract family from article_id (e.g., ART-100 -> ART-1xx)
            def get_family(art):
                if pd.isna(art):
                    return "UNKNOWN"
                art_str = str(art)
                if '-' in art_str:
                    parts = art_str.split('-')
                    if len(parts) >= 2 and parts[1]:
                        # Take first digit for family
                        return f"{parts[0]}-{parts[1][0]}xx"
                return art_str[:5]
            
            orders_df = orders_df.copy()
            orders_df['_family'] = orders_df[article_col].apply(get_family)
            
            for family, group in orders_df.groupby('_family'):
                groups[f"FAM-{family}"] = list(group[order_id_col].astype(str))
    
    elif mode == AggregationMode.BY_DUE_WEEK:
        if due_col in orders_df.columns:
            orders_df = orders_df.copy()
            orders_df['_due'] = pd.to_datetime(orders_df[due_col], errors='coerce')
            orders_df['_week'] = orders_df['_due'].dt.isocalendar().week
            orders_df['_year'] = orders_df['_due'].dt.year
            
            for (year, week), group in orders_df.groupby(['_year', '_week']):
                if pd.notna(year) and pd.notna(week):
                    groups[f"WEEK-{int(year)}-W{int(week):02d}"] = list(group[order_id_col].astype(str))
    
    return groups


def build_projects_from_orders(
    orders_df: pd.DataFrame,
    mode: AggregationMode = AggregationMode.EXPLICIT,
    project_col: str = 'project_id',
    client_col: str = 'client_id',
    article_col: str = 'article_id',
    due_col: str = 'due_date',
    priority_col: str = 'priority',
    default_priority: float = 1.0
) -> List[Project]:
    """
    Build Project objects from orders DataFrame.
    
    This is the main entry point for creating projects from raw order data.
    
    Mathematical Process:
    ────────────────────
    1. Group orders: O → P (using aggregation mode)
    2. For each project p:
       - O(p) = orders in group
       - d_p = max{d_o : o ∈ O(p)} (latest due date)
       - w_p = avg{priority(o) : o ∈ O(p)} (average priority)
    
    Args:
        orders_df: DataFrame with columns [order_id, article_id, due_date, priority, ...]
        mode: How to group orders into projects
        project_col: Column name for explicit project ID
        client_col: Column name for client
        article_col: Column name for article
        due_col: Column name for due date
        priority_col: Column name for order priority
        default_priority: Default priority if column missing
    
    Returns:
        List of Project objects
    
    Example:
        >>> orders = pd.DataFrame({
        ...     'order_id': ['O1', 'O2', 'O3'],
        ...     'project_id': ['P1', 'P1', 'P2'],
        ...     'due_date': ['2024-01-15', '2024-01-15', '2024-01-20']
        ... })
        >>> projects = build_projects_from_orders(orders)
        >>> len(projects)
        2
    """
    if orders_df.empty:
        return []
    
    # Determine order_id column
    order_id_col = 'order_id'
    for alt in ['order_id', 'id', 'OrderID', 'order']:
        if alt in orders_df.columns:
            order_id_col = alt
            break
    
    # Group orders
    groupings = group_orders_by_project(
        orders_df, mode, project_col, client_col, article_col, due_col
    )
    
    projects = []
    
    for proj_id, order_ids in groupings.items():
        # Filter orders for this project
        proj_orders = orders_df[orders_df[order_id_col].astype(str).isin(order_ids)]
        
        if proj_orders.empty:
            continue
        
        # Compute project due date (latest among orders)
        due_date = None
        if due_col in proj_orders.columns:
            due_dates = pd.to_datetime(proj_orders[due_col], errors='coerce')
            valid_dates = due_dates.dropna()
            if not valid_dates.empty:
                due_date = valid_dates.max().to_pydatetime()
        
        # Compute priority weight (average of order priorities)
        priority_weight = default_priority
        if priority_col in proj_orders.columns:
            priorities = pd.to_numeric(proj_orders[priority_col], errors='coerce')
            valid_priorities = priorities.dropna()
            if not valid_priorities.empty:
                priority_weight = float(valid_priorities.mean())
        
        # Determine client (most common or first)
        client = None
        if client_col in proj_orders.columns:
            clients = proj_orders[client_col].dropna()
            if not clients.empty:
                client = str(clients.mode().iloc[0]) if len(clients.mode()) > 0 else str(clients.iloc[0])
        
        # Create project name
        if mode == AggregationMode.EXPLICIT:
            name = f"Projeto {proj_id}"
        elif mode == AggregationMode.BY_CLIENT:
            name = f"Cliente {client}" if client else f"Projeto {proj_id}"
        elif mode == AggregationMode.BY_ARTICLE_FAMILY:
            name = f"Família {proj_id.replace('FAM-', '')}"
        elif mode == AggregationMode.BY_DUE_WEEK:
            name = f"Semana {proj_id.replace('WEEK-', '')}"
        else:
            name = proj_id
        
        project = Project(
            project_id=proj_id,
            name=name,
            client=client,
            order_ids=set(order_ids),
            due_date=due_date,
            priority_weight=priority_weight,
            status=ProjectStatus.PLANNING,
            metadata={
                'aggregation_mode': mode.value,
                'num_articles': proj_orders[article_col].nunique() if article_col in proj_orders.columns else 0,
            }
        )
        
        projects.append(project)
    
    # Sort by due date, then priority
    projects.sort(
        key=lambda p: (
            p.due_date if p.due_date else datetime.max,
            -p.priority_weight
        )
    )
    
    logger.info(f"Built {len(projects)} projects from {len(orders_df)} orders using mode={mode.value}")
    
    return projects


def merge_projects(projects: List[Project], new_project_id: str, new_name: str) -> Project:
    """
    Merge multiple projects into one.
    
    Useful for combining related projects.
    
    Args:
        projects: Projects to merge
        new_project_id: ID for merged project
        new_name: Name for merged project
    
    Returns:
        Merged Project
    """
    if not projects:
        raise ValueError("Cannot merge empty project list")
    
    # Combine order IDs
    all_orders = set()
    for p in projects:
        all_orders.update(p.order_ids)
    
    # Latest due date
    due_dates = [p.due_date for p in projects if p.due_date]
    due_date = max(due_dates) if due_dates else None
    
    # Sum budget
    budgets = [p.budget_hours for p in projects if p.budget_hours]
    budget = sum(budgets) if budgets else None
    
    # Average priority weighted by order count
    total_orders = sum(p.num_orders for p in projects)
    if total_orders > 0:
        priority = sum(p.priority_weight * p.num_orders for p in projects) / total_orders
    else:
        priority = 1.0
    
    # Collect clients
    clients = [p.client for p in projects if p.client]
    client = clients[0] if clients else None
    
    return Project(
        project_id=new_project_id,
        name=new_name,
        client=client,
        order_ids=all_orders,
        due_date=due_date,
        budget_hours=budget,
        priority_weight=priority,
        status=ProjectStatus.PLANNING,
        metadata={'merged_from': [p.project_id for p in projects]},
    )


def filter_projects_by_status(
    projects: List[Project],
    statuses: List[ProjectStatus]
) -> List[Project]:
    """Filter projects by status."""
    return [p for p in projects if p.status in statuses]


def filter_projects_by_date_range(
    projects: List[Project],
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Project]:
    """Filter projects by due date range."""
    filtered = []
    for p in projects:
        if not p.due_date:
            continue
        if start_date and p.due_date < start_date:
            continue
        if end_date and p.due_date > end_date:
            continue
        filtered.append(p)
    return filtered



