"""Sistema de cache para insights gerados pelo LLM."""

import hashlib
import json
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class InsightCache:
    """Cache de insights por batch_id e mode."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "insight_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache em memória para acesso rápido
        self._memory_cache: Dict[Tuple[str, str], str] = {}
    
    def _get_cache_key(self, batch_id: str, mode: str) -> str:
        """Gera chave de cache baseada em batch_id e mode."""
        key_str = f"{batch_id}:{mode}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, batch_id: str, mode: str) -> Path:
        """Retorna caminho do ficheiro de cache."""
        cache_key = self._get_cache_key(batch_id, mode)
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, batch_id: str, mode: str) -> Optional[str]:
        """Recupera insight do cache."""
        # Tentar memória primeiro
        memory_key = (batch_id, mode)
        if memory_key in self._memory_cache:
            return self._memory_cache[memory_key]
        
        # Tentar disco
        cache_path = self._get_cache_path(batch_id, mode)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    insight = data.get("insight")
                    # Carregar em memória
                    self._memory_cache[memory_key] = insight
                    return insight
            except Exception as exc:
                logger.warning(f"Erro ao ler cache de {batch_id}:{mode}: {exc}")
        
        return None
    
    def set(self, batch_id: str, mode: str, insight: str, metadata: Optional[Dict] = None):
        """Guarda insight no cache."""
        memory_key = (batch_id, mode)
        self._memory_cache[memory_key] = insight
        
        cache_path = self._get_cache_path(batch_id, mode)
        try:
            data = {
                "batch_id": batch_id,
                "mode": mode,
                "insight": insight,
                "metadata": metadata or {},
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"Erro ao escrever cache de {batch_id}:{mode}: {exc}")
    
    def invalidate(self, batch_id: str, mode: Optional[str] = None):
        """Invalida cache para um batch_id (e opcionalmente um mode específico)."""
        if mode:
            # Invalidar apenas um mode
            memory_key = (batch_id, mode)
            self._memory_cache.pop(memory_key, None)
            cache_path = self._get_cache_path(batch_id, mode)
            if cache_path.exists():
                cache_path.unlink()
        else:
            # Invalidar todos os modes deste batch_id
            # Remover da memória
            keys_to_remove = [k for k in self._memory_cache.keys() if k[0] == batch_id]
            for key in keys_to_remove:
                del self._memory_cache[key]
            
            # Remover do disco (todos os ficheiros que começam com o hash do batch_id)
            # Como não sabemos todos os modes, vamos procurar
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("batch_id") == batch_id:
                            cache_file.unlink()
                except:
                    pass
    
    def clear(self):
        """Limpa todo o cache."""
        self._memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
            except:
                pass


# Instância global do cache
_insight_cache: Optional[InsightCache] = None


def get_insight_cache() -> InsightCache:
    """Retorna instância global do cache."""
    global _insight_cache
    if _insight_cache is None:
        _insight_cache = InsightCache()
    return _insight_cache

