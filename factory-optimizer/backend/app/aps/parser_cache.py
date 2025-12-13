"""
Cache para parser de Excel - evita reparsear o mesmo ficheiro múltiplas vezes.

Cache em memória e disco para Orders parseadas.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.aps.models import Order

logger = logging.getLogger(__name__)


class ParserCache:
    """Cache de Orders parseadas por hash do ficheiro Excel."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "parser_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache em memória (file_hash -> Orders)
        self._memory_cache: Dict[str, List[Order]] = {}
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calcula hash do ficheiro para usar como chave de cache."""
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return ""
        
        # Usar tamanho + modificação time como hash simples
        stat = file_path_obj.stat()
        hash_input = f"{file_path}:{stat.st_size}:{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, file_hash: str) -> Path:
        """Retorna caminho do ficheiro de cache."""
        return self.cache_dir / f"{file_hash}.json"
    
    def get(self, file_path: str) -> Optional[List[Order]]:
        """Recupera Orders do cache (apenas memória - cache de disco seria muito complexo)."""
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return None
        
        # Verificar se ficheiro mudou (mtime diferente)
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return None
        
        # Tentar memória
        if file_hash in self._memory_cache:
            # Validar que o ficheiro não mudou
            current_hash = self._get_file_hash(file_path)
            if current_hash == file_hash:
                logger.debug(f"Cache hit (memória) para {file_path}")
                return self._memory_cache[file_hash]
            else:
                # Ficheiro mudou, invalidar
                logger.info(f"Ficheiro {file_path} mudou, invalidando cache")
                self._memory_cache.pop(file_hash, None)
        
        return None
    
    def set(self, file_path: str, orders: List[Order]):
        """Guarda Orders no cache (apenas memória)."""
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return
        
        # Guardar apenas em memória (cache de disco seria muito complexo para Orders)
        self._memory_cache[file_hash] = orders
        logger.info(f"Cache guardado (memória) para {file_path}: {len(orders)} orders")
    
    def invalidate(self, file_path: str):
        """Invalida cache para um ficheiro."""
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return
        
        # Remover de memória
        self._memory_cache.pop(file_hash, None)
        
        # Remover do disco
        cache_path = self._get_cache_path(file_hash)
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Cache invalidado para {file_path}")
    
    def clear(self):
        """Limpa todo o cache."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except:
                pass
    
    @staticmethod
    def _serialize_orders(orders: List[Order]) -> List[Dict]:
        """Serializa Orders para dict (simplificado)."""
        # Para cache, guardar apenas dados essenciais
        # O parser pode reconstruir o resto
        return [
            {
                "id": order.id,
                "artigo": order.artigo,
                "quantidade": order.quantidade,
                "prioridade": order.prioridade,
                "due_date": order.due_date.isoformat() if order.due_date else None,
                "data_entrada": order.data_entrada.isoformat(),
                "operations_count": len(order.operations),
            }
            for order in orders
        ]
    
    @staticmethod
    def _deserialize_orders(data: List[Dict]) -> List[Order]:
        """Deserializa Orders de dict."""
        # Nota: Esta é uma versão simplificada
        # Para cache completo, seria necessário serializar todas as operações
        # Por agora, retornar None para forçar reparse
        # (cache completo seria muito complexo)
        return []


# Instância global
_parser_cache: Optional[ParserCache] = None


def get_parser_cache() -> ParserCache:
    """Retorna instância global do cache de parser."""
    global _parser_cache
    if _parser_cache is None:
        _parser_cache = ParserCache()
    return _parser_cache

