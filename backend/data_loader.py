"""
Utilities for loading the Excel-based MVP dataset into pandas DataFrames.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from env if present
load_dotenv()

# Environment variable that may override the Excel path
DATA_FILE_ENV = "DATA_WORKBOOK_PATH"

# Default Excel path (relative to project root)
DEFAULT_WORKBOOK = (
    Path(__file__).resolve().parents[1] / "data" / "production_os_data_MVP.xlsx"
)

# Sheets that MUST exist in the workbook
REQUIRED_SHEETS = (
    "orders",
    "operations",
    "machines",
    "routing",
    "shifts",
    "downtime",
    "setup_matrix",
)


# ---------------------------------------------------
# Data container
# ---------------------------------------------------

@dataclass(frozen=True)
class DataBundle:
    """Typed container for all Excel sheets and metadata."""

    orders: pd.DataFrame
    operations: pd.DataFrame
    machines: pd.DataFrame
    routing: pd.DataFrame
    shifts: pd.DataFrame
    downtime: pd.DataFrame
    setup_matrix: pd.DataFrame

    raw_path: Path
    loaded_at: datetime


# Cache: Excel is only read once
_CACHE: Optional[DataBundle] = None


# ---------------------------------------------------
# Cleaning utilities
# ---------------------------------------------------

def _coerce_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "due_date" in df.columns:
        df["due_date"] = _coerce_datetime(df["due_date"])
    if "priority" in df.columns:
        df["priority"] = _coerce_numeric(df["priority"]).fillna(0).astype(int)
    return df


def _clean_shifts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "shift_date" in df.columns:
        df["shift_date"] = _coerce_datetime(df["shift_date"])
    return df


def _clean_downtime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "down_start" in df.columns:
        df["down_start"] = _coerce_datetime(df["down_start"])
    if "down_end" in df.columns:
        df["down_end"] = _coerce_datetime(df["down_end"])
    return df


_CLEANERS: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "orders": _clean_orders,
    "shifts": _clean_shifts,
    "downtime": _clean_downtime,
}


# ---------------------------------------------------
# Workbook resolution
# ---------------------------------------------------

def _resolve_workbook_path() -> Path:
    """Return the Excel path, prioritizing environment override."""
    env_path = os.getenv(DATA_FILE_ENV)
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    return DEFAULT_WORKBOOK


# ---------------------------------------------------
# Sheet reader
# ---------------------------------------------------

def _read_sheet(excel_file: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    """Read a sheet and apply cleaning if needed."""
    frame = excel_file.parse(sheet)
    cleaner = _CLEANERS.get(sheet)
    return cleaner(frame) if cleaner else frame


# ---------------------------------------------------
# Main dataset loader
# ---------------------------------------------------

def load_dataset() -> DataBundle:
    """Load all Excel sheets into a cached DataBundle."""
    global _CACHE

    if _CACHE is not None:
        return _CACHE

    workbook_path = _resolve_workbook_path()

    if not workbook_path.exists():
        raise FileNotFoundError(f"❌ Excel workbook not found at: {workbook_path}")

    excel = pd.ExcelFile(workbook_path)

    # Validate required sheets
    for sheet in REQUIRED_SHEETS:
        if sheet not in excel.sheet_names:
            raise ValueError(
                f"❌ Missing required sheet '{sheet}'. Available: {excel.sheet_names}"
            )

    # Load sheets
    orders = _read_sheet(excel, "orders")
    operations = _read_sheet(excel, "operations")
    machines = _read_sheet(excel, "machines")
    routing = _read_sheet(excel, "routing")
    shifts = _read_sheet(excel, "shifts")
    downtime = _read_sheet(excel, "downtime")
    setup_matrix = _read_sheet(excel, "setup_matrix")

    # Create the bundle
    _CACHE = DataBundle(
        orders=orders,
        operations=operations,
        machines=machines,
        routing=routing,
        shifts=shifts,
        downtime=downtime,
        setup_matrix=setup_matrix,
        raw_path=workbook_path,
        loaded_at=datetime.now(),
    )

    return _CACHE


def refresh_dataset() -> DataBundle:
    """
    Recarrega o Excel ignorando o cache atual.
    Útil após operações What-If ou quando o ficheiro muda.
    """
    global _CACHE
    _CACHE = None
    return load_dataset()


def as_records(df: pd.DataFrame) -> list[dict]:
    """
    Converte DataFrames em listas de dicts serializáveis (ISO para datas).
    """
    if df.empty:
        return []
    serialisable = df.copy()
    for column in serialisable.columns:
        if pd.api.types.is_datetime64_any_dtype(serialisable[column]):
            serialisable[column] = serialisable[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return serialisable.to_dict(orient="records")

