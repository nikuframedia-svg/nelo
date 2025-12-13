"""
Estruturas base para planeamento encadeado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class MachineChain:
    """
    Representa uma cadeia de máquinas (A -> B -> C) que deve ser tratada como fluxo contínuo.
    """

    name: str
    machines: List[str]


# TODO[PLANEAMENTO_ENCADEADO]:
# - Carregar definições de cadeias a partir de configuração ou Excel.
# - Injetar estas cadeias no scheduler quando mode == "ENCADEADO".



