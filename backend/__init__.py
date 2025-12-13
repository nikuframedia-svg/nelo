"""
Nikufra Production OS MVP backend package.
"""
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

__all__ = ["BACKEND_ROOT", "PROJECT_ROOT"]



