"""
Sistema de cache para planos APS.

Cache em 3 níveis:
1. Memória (dict Python) - acesso < 10ms
2. Disco (JSON) - persistência entre restarts
3. Recalcular - se cache miss
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.aps.models import Plan

logger = logging.getLogger(__name__)


class PlanCache:
    """Cache de planos APS por (batch_id, horizon_hours)."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "plan_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Limpar caches antigos (formato antes/depois) ao iniciar
        self._clean_old_caches()
        
        # Cache em memória
        self._memory_cache: Dict[tuple, Plan] = {}
    
    def _clean_old_caches(self):
        """Remove caches com formato antigo (antes/depois)."""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Se tiver formato antigo, remover
                    if "antes" in data or "depois" in data:
                        logger.info(f"Removendo cache antigo: {cache_file.name}")
                        cache_file.unlink()
            except:
                # Se houver erro ao ler, remover também
                logger.warning(f"Removendo cache corrompido: {cache_file.name}")
                cache_file.unlink()
    
    def _get_cache_key(self, batch_id: str, horizon_hours: int) -> str:
        """Gera chave de cache."""
        key_str = f"{batch_id}:{horizon_hours}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self, batch_id: str, horizon_hours: int) -> Path:
        """Retorna caminho do ficheiro de cache."""
        cache_key = self._get_cache_key(batch_id, horizon_hours)
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, batch_id: str, horizon_hours: int) -> Optional[Plan]:
        """Recupera plano do cache."""
        memory_key = (batch_id, horizon_hours)
        
        # Tentar memória primeiro
        if memory_key in self._memory_cache:
            plan = self._memory_cache[memory_key]
            # Validar que o plano tem operações
            if plan.baseline and len(plan.baseline.operations) == 0:
                logger.warning(f"Plano em cache sem operações deserializadas. Invalidando.")
                self._memory_cache.pop(memory_key, None)
                return None
            return plan
        
        # Tentar disco
        cache_path = self._get_cache_path(batch_id, horizon_hours)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # VALIDAÇÃO: Se for formato antigo (antes/depois), rejeitar
                    if "antes" in data or "depois" in data:
                        logger.warning(f"Cache antigo detectado (formato antes/depois). Invalidando.")
                        cache_path.unlink()
                        return None
                    
                    plan = self._deserialize_plan(data)
                    
                    # Validar que o plano tem operações
                    if plan.baseline and len(plan.baseline.operations) == 0:
                        logger.warning(f"Plano deserializado sem operações. Invalidando cache.")
                        cache_path.unlink()
                        return None
                    
                    # Carregar em memória
                    self._memory_cache[memory_key] = plan
                    return plan
            except Exception as exc:
                logger.warning(f"Erro ao ler cache de plano: {exc}")
                # Se houver erro, remover cache corrompido
                if cache_path.exists():
                    cache_path.unlink()
        
        return None
    
    def set(self, batch_id: str, horizon_hours: int, plan: Plan, all_machines: Optional[List[str]] = None):
        """
        Guarda plano no cache.
        
        Args:
            batch_id: ID do batch
            horizon_hours: Horizonte em horas
            plan: Plano a guardar
            all_machines: Lista opcional de TODAS as máquinas (para garantir serialização completa)
        """
        memory_key = (batch_id, horizon_hours)
        self._memory_cache[memory_key] = plan
        
        cache_path = self._get_cache_path(batch_id, horizon_hours)
        try:
            data = plan.to_dict(all_machines_from_engine=all_machines)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            logger.warning(f"Erro ao escrever cache de plano: {exc}")
    
    def invalidate(self, batch_id: str, horizon_hours: Optional[int] = None):
        """Invalida cache para um batch_id (e opcionalmente horizon_hours específico)."""
        if horizon_hours is not None:
            # Invalidar apenas um horizon_hours
            memory_key = (batch_id, horizon_hours)
            self._memory_cache.pop(memory_key, None)
            cache_path = self._get_cache_path(batch_id, horizon_hours)
            if cache_path.exists():
                cache_path.unlink()
        else:
            # Invalidar todos os horizon_hours deste batch_id
            keys_to_remove = [
                k for k in self._memory_cache.keys() if k[0] == batch_id
            ]
            for key in keys_to_remove:
                del self._memory_cache[key]
            
            # Remover do disco
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
    
    @staticmethod
    def _deserialize_plan(data: Dict) -> Plan:
        """Deserializa Plan de dict."""
        from app.aps.models import APSConfig, PlanResult, ScheduledOperation, OpRef, OpAlternative
        from datetime import datetime
        
        # Deserializar config
        config = APSConfig.from_dict(data.get("config", {}))
        
        def _deserialize_operations(ops_data: List[Dict]) -> List[ScheduledOperation]:
            """Deserializa lista de operações."""
            operations = []
            for op_data in ops_data:
                try:
                    # Criar OpAlternative mínimo (dados básicos)
                    alt = OpAlternative(
                        maquina_id=op_data.get("maquina_id", ""),
                        ratio_pch=0.0,  # Não está no JSON serializado
                        pessoas=1.0,
                        family=op_data.get("family", ""),
                        setup_h=0.0,
                        overlap_pct=0.0,
                    )
                    
                    # Criar OpRef mínimo
                    # CRÍTICO: Usar rota do JSON, não fallback "A" (isso estava a sobrescrever rotas B)
                    rota_from_json = op_data.get("rota")
                    if not rota_from_json or rota_from_json == "":
                        rota_from_json = "A"  # Só usar A se realmente não houver rota no JSON
                        logger.debug(f"⚠️ Cache: rota não encontrada no JSON para op {op_data.get('op_id')}, usando fallback 'A'")
                    else:
                        logger.debug(f"✅ Cache: rota '{rota_from_json}' lida do JSON para op {op_data.get('op_id')}")
                    op_ref = OpRef(
                        op_id=op_data.get("op_id", ""),
                        rota=rota_from_json,  # Usar rota real do JSON
                        stage_index=0,  # Não está no JSON
                        precedencias=[],
                        operacao_logica=op_data.get("family", ""),
                        alternatives=[alt],
                    )
                    
                    # Criar ScheduledOperation
                    start_time_str = op_data.get("start_time")
                    if isinstance(start_time_str, str):
                        start_time = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
                    else:
                        start_time = datetime.utcnow()
                    
                    end_time_str = op_data.get("end_time")
                    if isinstance(end_time_str, str):
                        end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
                    else:
                        end_time = datetime.utcnow()
                    
                    scheduled_op = ScheduledOperation(
                        order_id=op_data.get("order_id", ""),
                        op_ref=op_ref,
                        alternative_chosen=alt,
                        start_time=start_time,
                        end_time=end_time,
                        quantidade=op_data.get("quantidade", 0),
                        duracao_h=op_data.get("duracao_h", 0.0),
                    )
                    operations.append(scheduled_op)
                except Exception as exc:
                    logger.warning(f"Erro ao deserializar operação: {exc}")
                    continue
            return operations
        
        # CRÍTICO: Obter lista de todas as máquinas do JSON (all_machines)
        # Isto garante que mesmo máquinas sem operações são incluídas
        all_machines_from_json = set()
        for result_data in [data.get("baseline"), data.get("optimized")]:
            if result_data:
                all_machines_list = result_data.get("all_machines", [])
                if all_machines_list:
                    all_machines_from_json.update(all_machines_list)
                # Também adicionar máquinas que aparecem no gantt_by_machine
                gantt = result_data.get("gantt_by_machine", {})
                all_machines_from_json.update(gantt.keys())
        
        logger.info(f"📋 Cache: Deserializando plano com {len(all_machines_from_json)} máquinas: {sorted(all_machines_from_json)}")
        
        # Deserializar baseline
        baseline_data = data.get("baseline")
        baseline = None
        if baseline_data:
            operations = _deserialize_operations(baseline_data.get("operations", []))
            gantt_by_machine = baseline_data.get("gantt_by_machine", {})
            
            # CRÍTICO: Garantir que TODAS as máquinas aparecem no gantt_by_machine
            for machine_id in all_machines_from_json:
                if machine_id not in gantt_by_machine:
                    gantt_by_machine[machine_id] = []
            
            baseline = PlanResult(
                makespan_h=baseline_data.get("makespan_h", 0.0),
                total_setup_h=baseline_data.get("total_setup_h", 0.0),
                kpis=baseline_data.get("kpis", {}),
                operations=operations,  # ✅ AGORA DESERIALIZA AS OPERAÇÕES
                gantt_by_machine=gantt_by_machine,
            )
            # Reconstruir Gantt passando todas as máquinas
            baseline.build_gantt(all_machines=sorted(list(all_machines_from_json)))
            
            # Garantir novamente após build_gantt (pode ter removido máquinas vazias)
            for machine_id in all_machines_from_json:
                if machine_id not in baseline.gantt_by_machine:
                    baseline.gantt_by_machine[machine_id] = []
                    logger.debug(f"📋 Cache: Adicionada máquina {machine_id} ao baseline.gantt_by_machine")
        
        # Deserializar optimized
        optimized_data = data.get("optimized")
        optimized = None
        if optimized_data:
            operations = _deserialize_operations(optimized_data.get("operations", []))
            gantt_by_machine = optimized_data.get("gantt_by_machine", {})
            
            # CRÍTICO: Garantir que TODAS as máquinas aparecem no gantt_by_machine
            for machine_id in all_machines_from_json:
                if machine_id not in gantt_by_machine:
                    gantt_by_machine[machine_id] = []
            
            optimized = PlanResult(
                makespan_h=optimized_data.get("makespan_h", 0.0),
                total_setup_h=optimized_data.get("total_setup_h", 0.0),
                kpis=optimized_data.get("kpis", {}),
                operations=operations,  # ✅ AGORA DESERIALIZA AS OPERAÇÕES
                gantt_by_machine=gantt_by_machine,
            )
            # Reconstruir Gantt passando todas as máquinas
            optimized.build_gantt(all_machines=sorted(list(all_machines_from_json)))
            
            # Garantir novamente após build_gantt (pode ter removido máquinas vazias)
            for machine_id in all_machines_from_json:
                if machine_id not in optimized.gantt_by_machine:
                    optimized.gantt_by_machine[machine_id] = []
                    logger.debug(f"📋 Cache: Adicionada máquina {machine_id} ao optimized.gantt_by_machine")
        
        created_at_str = data.get("created_at", datetime.utcnow().isoformat())
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except:
            created_at = datetime.utcnow()
        
        plan = Plan(
            batch_id=data.get("batch_id", ""),
            horizon_hours=data.get("horizon_hours", 4),
            created_at=created_at,
            baseline=baseline,
            optimized=optimized,
            config=config,
        )
        
        return plan


# Instância global do cache
_plan_cache: Optional[PlanCache] = None


def get_plan_cache() -> PlanCache:
    """Retorna instância global do cache."""
    global _plan_cache
    if _plan_cache is None:
        _plan_cache = PlanCache()
    return _plan_cache

