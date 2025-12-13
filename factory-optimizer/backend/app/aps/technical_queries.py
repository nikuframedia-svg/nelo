"""
Consultas Técnicas - Backend determinístico para perguntas sobre o modelo APS

Este módulo fornece funções determinísticas para consultar:
- Alternativas de máquinas
- Rotas disponíveis
- Operações por máquina
- Famílias por máquina
- Etc.

TODAS estas consultas vêm diretamente do modelo Order → OpRef → OpAlternative.
O LLM NUNCA deve inventar respostas - apenas chamar estas funções e reformular.
"""

import logging
from typing import Dict, List, Optional, Set

from app.aps.models import Order, OpAlternative, OpRef
from app.aps.parser import ProductionDataParser
from app.etl.loader import get_loader

logger = logging.getLogger(__name__)


class TechnicalQueries:
    """
    Consultas técnicas determinísticas sobre o modelo de produção.
    
    Todas as respostas vêm diretamente do Excel parseado, nunca inventadas.
    """
    
    def __init__(self, orders: Optional[List[Order]] = None):
        """
        Inicializa com lista de ordens.
        
        Se orders=None, carrega do Excel atual.
        """
        if orders is None:
            self.orders = self._load_orders_from_excel()
        else:
            self.orders = orders
        
        # Cache de estruturas para consultas rápidas
        self._machine_to_operations: Dict[str, Set[str]] = {}
        self._machine_to_families: Dict[str, Set[str]] = {}
        self._article_to_routes: Dict[str, Set[str]] = {}
        self._route_to_operations: Dict[tuple, Set[str]] = {}  # (artigo, rota) -> operações
        self._operation_to_alternatives: Dict[tuple, List[OpAlternative]] = {}  # (artigo, rota, op) -> alternativas
        
        self._build_indexes()
    
    def _load_orders_from_excel(self) -> List[Order]:
        """Carrega ordens do Excel atual."""
        try:
            loader = get_loader()
            data_dir = loader.data_dir
            excel_file = data_dir / "Nikufra DadosProducao (2).xlsx"
            if not excel_file.exists():
                excel_file = data_dir / "Nikufra DadosProducao.xlsx"
            
            if not excel_file.exists():
                logger.warning("Ficheiro Excel não encontrado")
                return []
            
            parser = ProductionDataParser()
            return parser.parse_dadosnikufra2(str(excel_file))
        except Exception as exc:
            logger.error(f"Erro ao carregar ordens do Excel: {exc}")
            return []
    
    def _build_indexes(self):
        """Constrói índices para consultas rápidas."""
        for order in self.orders:
            artigo = order.artigo
            
            # Indexar rotas por artigo
            if artigo not in self._article_to_routes:
                self._article_to_routes[artigo] = set()
            
            for op_ref in order.operations:
                rota = op_ref.rota
                op_id = op_ref.op_id
                
                # Indexar rota
                self._article_to_routes[artigo].add(rota)
                
                # Indexar operações por rota
                key = (artigo, rota)
                if key not in self._route_to_operations:
                    self._route_to_operations[key] = set()
                self._route_to_operations[key].add(op_id)
                
                # Indexar alternativas por operação
                op_key = (artigo, rota, op_id)
                if op_key not in self._operation_to_alternatives:
                    self._operation_to_alternatives[op_key] = []
                
                for alt in op_ref.alternatives:
                    machine_id = alt.maquina_id
                    family = alt.family
                    
                    # Indexar máquinas -> operações
                    if machine_id not in self._machine_to_operations:
                        self._machine_to_operations[machine_id] = set()
                    self._machine_to_operations[machine_id].add(f"{artigo}/{rota}/{op_id}")
                    
                    # Indexar máquinas -> famílias
                    if machine_id not in self._machine_to_families:
                        self._machine_to_families[machine_id] = set()
                    if family:
                        self._machine_to_families[machine_id].add(family)
                    
                    # Indexar alternativas
                    self._operation_to_alternatives[op_key].append(alt)
    
    def get_alternatives(
        self, 
        artigo: str, 
        rota: str, 
        operacao: str
    ) -> List[Dict]:
        """
        Retorna alternativas de máquinas para uma operação específica.
        
        Args:
            artigo: Artigo (ex: "GO Artigo 6")
            rota: Rota (ex: "Rota A")
            operacao: ID da operação (ex: "OP-001")
        
        Returns:
            Lista de alternativas com máquina, ratio, pessoas, etc.
        """
        key = (artigo, rota, operacao)
        alternatives = self._operation_to_alternatives.get(key, [])
        
        return [
            {
                "maquina_id": alt.maquina_id,
                "ratio_pch": alt.ratio_pch,
                "pessoas": alt.pessoas,
                "family": alt.family,
                "setup_h": alt.setup_h,
                "overlap": alt.overlap,
            }
            for alt in alternatives
        ]
    
    def get_routes(self, artigo: str) -> List[str]:
        """
        Retorna rotas disponíveis para um artigo.
        
        Args:
            artigo: Artigo (ex: "GO Artigo 6")
        
        Returns:
            Lista de rotas (ex: ["Rota A", "Rota B"])
        """
        return sorted(list(self._article_to_routes.get(artigo, set())))
    
    def get_operations_on_machine(self, machine_id: str) -> List[Dict]:
        """
        Retorna operações que podem ser feitas numa máquina.
        
        Args:
            machine_id: ID da máquina (ex: "300")
        
        Returns:
            Lista de operações com artigo, rota, op_id
        """
        operations = self._machine_to_operations.get(machine_id, set())
        
        result = []
        for op_str in operations:
            parts = op_str.split("/")
            if len(parts) == 3:
                artigo, rota, op_id = parts
                result.append({
                    "artigo": artigo,
                    "rota": rota,
                    "op_id": op_id,
                })
        
        return sorted(result, key=lambda x: (x["artigo"], x["rota"], x["op_id"]))
    
    def get_families_on_machine(self, machine_id: str) -> List[str]:
        """
        Retorna famílias que usam uma máquina.
        
        Args:
            machine_id: ID da máquina (ex: "300")
        
        Returns:
            Lista de famílias
        """
        return sorted(list(self._machine_to_families.get(machine_id, set())))
    
    def get_operations_in_route(self, artigo: str, rota: str) -> List[str]:
        """
        Retorna operações numa rota específica.
        
        Args:
            artigo: Artigo (ex: "GO Artigo 6")
            rota: Rota (ex: "Rota A")
        
        Returns:
            Lista de IDs de operações
        """
        key = (artigo, rota)
        return sorted(list(self._route_to_operations.get(key, set())))
    
    def validate_machine(self, machine_id: str) -> bool:
        """Valida se uma máquina existe no modelo."""
        return machine_id in self._machine_to_operations
    
    def validate_article(self, artigo: str) -> bool:
        """Valida se um artigo existe no modelo."""
        return artigo in self._article_to_routes
    
    def validate_route(self, artigo: str, rota: str) -> bool:
        """Valida se uma rota existe para um artigo."""
        routes = self._article_to_routes.get(artigo, set())
        return rota in routes
    
    def validate_operation(self, artigo: str, rota: str, operacao: str) -> bool:
        """Valida se uma operação existe numa rota."""
        key = (artigo, rota)
        operations = self._route_to_operations.get(key, set())
        return operacao in operations
    
    def get_all_machines(self) -> List[str]:
        """Retorna todas as máquinas disponíveis."""
        return sorted(list(self._machine_to_operations.keys()))
    
    def get_all_articles(self) -> List[str]:
        """Retorna todos os artigos disponíveis."""
        return sorted(list(self._article_to_routes.keys()))


# Instância global (lazy loading)
_technical_queries: Optional[TechnicalQueries] = None


def get_technical_queries() -> TechnicalQueries:
    """Retorna instância global de TechnicalQueries."""
    global _technical_queries
    if _technical_queries is None:
        _technical_queries = TechnicalQueries()
    return _technical_queries

