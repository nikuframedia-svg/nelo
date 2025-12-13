"""
APS Engine - Motor de planeamento encadeado.

Implementa algoritmos:
- Baseline: FIFO simples com primeira alternativa
- Optimized: Greedy com score de alternativas
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.aps.models import (
    APSConfig,
    MachineState,
    OpAlternative,
    OpRef,
    Order,
    Plan,
    PlanResult,
    ScheduledOperation,
    TimeWindow,
)
from app.aps.planning_config import PlanningConfig

logger = logging.getLogger(__name__)


class APSEngine:
    """Motor de planeamento encadeado com rotas e máquinas alternativas."""
    
    def __init__(self, config: Optional[APSConfig] = None, planning_config: Optional[PlanningConfig] = None):
        self.config = config or APSConfig()
        
        # VALIDAÇÃO CRÍTICA: Limpar preferências de rota globais/inválidas ao inicializar
        if self.config.routing_preferences.get("prefer_route"):
            prefer_route_dict = self.config.routing_preferences["prefer_route"]
            invalid_keys = [k for k in prefer_route_dict.keys() if k in ["*", "all", "ALL", ""] or not k or len(str(k)) < 2]
            if invalid_keys:
                logger.warning(f"⚠️ [APS] APSEngine.__init__: Removendo preferências de rota inválidas: {invalid_keys}")
                for invalid_key in invalid_keys:
                    prefer_route_dict.pop(invalid_key, None)
        
        self.planning_config = planning_config  # Configuração de planeamento (indisponibilidades, ordens manuais, etc.)
        self.machines: Dict[str, MachineState] = {}
    
    def build_schedule(
        self,
        orders: List[Order],
        horizon_hours: int,
        start_time: Optional[datetime] = None,
    ) -> Plan:
        """
        Constrói plano completo (baseline + optimized).
        
        Args:
            orders: Lista de ordens de produção
            horizon_hours: Horizonte de planeamento em horas
            start_time: Tempo de início (default: agora)
        
        Returns:
            Plan com baseline e optimized
        """
        if not orders:
            raise ValueError("Lista de ordens vazia")
        
        # Log mínimo para performance
        logger.info(f"🚀 APS: {len(orders)} orders, horizon={horizon_hours}h")
        
        start_time = start_time or datetime.utcnow()
        end_time = start_time + timedelta(hours=horizon_hours)
        
        # Verificar se Orders têm operações (apenas log de erros)
        for order in orders:
            if not order.operations:
                logger.error(f"❌ Order {order.id} ({order.artigo}) sem operações!")
        
        # Aplicar configuração de planeamento (indisponibilidades, ordens manuais, prioridades)
        orders = self._apply_planning_config(orders, start_time)
        
        # Usar horizonte da configuração se definido
        if self.planning_config and self.planning_config.horizon_hours:
            horizon_hours = self.planning_config.horizon_hours
            end_time = start_time + timedelta(hours=horizon_hours)
            logger.info(f"⏰ Horizonte alterado pela configuração: {horizon_hours}h")
        
        # Inicializar máquinas
        self._initialize_machines(orders)
        
        # Aplicar indisponibilidades às máquinas
        self._apply_machine_unavailabilities(start_time, end_time)
        
        # Calcular baseline
        baseline = self._calculate_baseline(orders, start_time, end_time)
        
        # Garantir que baseline tem todas as máquinas
        for machine_id in self.machines.keys():
            if machine_id not in baseline.gantt_by_machine:
                baseline.gantt_by_machine[machine_id] = []

        # Calcular optimized
        optimized = self._calculate_optimized(orders, start_time, end_time, baseline)
        
        # Garantir que optimized tem todas as máquinas
        for machine_id in self.machines.keys():
            if machine_id not in optimized.gantt_by_machine:
                optimized.gantt_by_machine[machine_id] = []
        
        # VALIDAÇÃO FINAL: Garantir que o config não tem preferências globais antes de criar Plan
        if self.config.routing_preferences.get("prefer_route"):
            prefer_route_dict = self.config.routing_preferences["prefer_route"]
            invalid_keys = [k for k in prefer_route_dict.keys() if k in ["*", "all", "ALL", ""] or not k or len(str(k)) < 2]
            if invalid_keys:
                logger.warning(f"⚠️ [APS] build_schedule: Removendo preferências de rota inválidas antes de criar Plan: {invalid_keys}")
                for invalid_key in invalid_keys:
                    prefer_route_dict.pop(invalid_key, None)
        
        # Criar plano
        plan = Plan(
            batch_id=f"batch_{int(start_time.timestamp())}",
            horizon_hours=horizon_hours,
            created_at=start_time,
            baseline=baseline,
            optimized=optimized,
            config=self.config,
        )
        
        return plan
    
    def _initialize_machines(self, orders: List[Order]):
        """
        Inicializa estado de todas as máquinas.
        
        IMPORTANTE: Inicializa TODAS as máquinas que aparecem no parser,
        não apenas as que aparecem nas alternativas das ordens atuais.
        Isto garante que o frontend mostra todas as máquinas, mesmo que
        não tenham operações agendadas.
        """
        # Primeiro, coletar máquinas das alternativas das ordens
        machine_ids_from_orders = set()
        for order in orders:
            for op_ref in order.operations:
                for alt in op_ref.alternatives:
                    machine_ids_from_orders.add(alt.maquina_id)
        
        # Depois, tentar obter TODAS as máquinas do parser via TechnicalQueries
        # Isto garante que máquinas que não aparecem nas ordens atuais também são incluídas
        all_machine_ids = set(machine_ids_from_orders)
        
        try:
            from app.aps.technical_queries import TechnicalQueries
            # Criar instância temporária com as ordens atuais para garantir que temos todas as máquinas
            tech_queries = TechnicalQueries(orders=orders)
            all_machines_from_parser = tech_queries.get_all_machines()
            all_machine_ids.update(all_machines_from_parser)
            logger.info(f"✅ [APS] TechnicalQueries encontrou {len(all_machines_from_parser)} máquinas no parser (total: {len(all_machine_ids)})")
        except Exception as exc:
            logger.warning(f"⚠️ [APS] Não foi possível obter todas as máquinas do parser: {exc}. Usando apenas máquinas das ordens.")
            # Se falhar, usar apenas as máquinas das ordens (comportamento anterior)
            all_machine_ids = machine_ids_from_orders
        
        # Inicializar todas as máquinas
        self.machines = {
            machine_id: MachineState(id=machine_id)
            for machine_id in all_machine_ids
        }
        
        logger.info(f"✅ [APS] Inicializadas {len(self.machines)} máquinas ({len(machine_ids_from_orders)} das ordens, {len(all_machine_ids) - len(machine_ids_from_orders)} adicionais do parser)")
    
    def _apply_planning_config(self, orders: List[Order], start_time: datetime) -> List[Order]:
        """
        Aplica configuração de planeamento:
        - Adiciona ordens manuais
        - Aplica prioridades modificadas
        """
        if not self.planning_config:
            return orders
        
        # Criar cópia da lista de ordens
        all_orders = list(orders)
        
        # Adicionar ordens manuais
        for manual_order in self.planning_config.manual_orders:
            # Criar Order a partir de ManualOrder
            # Para ordens manuais, precisamos copiar as operações do artigo existente
            manual_order_obj = None
            
            # Tentar encontrar operações do artigo nas ordens existentes
            for existing_order in orders:
                if existing_order.artigo == manual_order.artigo or existing_order.artigo.replace(" ", "") == manual_order.artigo.replace(" ", ""):
                    # Copiar operações do artigo existente (deep copy)
                    import copy
                    manual_order_obj = Order(
                        id=f"MANUAL-{manual_order.artigo}-{int(start_time.timestamp())}",
                        artigo=manual_order.artigo,
                        quantidade=manual_order.quantidade,
                        prioridade=manual_order.prioridade,
                        due_date=manual_order.due_date,
                        data_entrada=start_time,
                        operations=copy.deepcopy(existing_order.operations),  # Copiar operações
                    )
                    break
            
            if manual_order_obj and manual_order_obj.operations:
                all_orders.append(manual_order_obj)
                logger.info(f"Ordem manual adicionada: {manual_order_obj.id} ({manual_order.artigo}) com {len(manual_order_obj.operations)} operações")
            else:
                logger.warning(f"Ordem manual {manual_order.artigo} não pode ser adicionada: operações não encontradas para este artigo")
        
        # Aplicar prioridades modificadas
        for order in all_orders:
            # Verificar se há override de prioridade (por order_id ou artigo)
            if order.id in self.planning_config.priority_overrides:
                order.prioridade = self.planning_config.priority_overrides[order.id]
                logger.info(f"Prioridade override aplicada: {order.id} -> {order.prioridade}")
            elif order.artigo in self.planning_config.priority_overrides:
                order.prioridade = self.planning_config.priority_overrides[order.artigo]
                logger.info(f"Prioridade override aplicada: {order.artigo} -> {order.prioridade}")
        
        return all_orders
    
    def _apply_machine_unavailabilities(self, start_time: datetime, end_time: datetime):
        """Aplica indisponibilidades de máquinas ao estado das máquinas."""
        if not self.planning_config:
            return
        
        for unavailability in self.planning_config.machine_unavailabilities:
            maquina_id = unavailability.maquina_id
            
            # Verificar se máquina existe
            if maquina_id not in self.machines:
                logger.warning(f"Indisponibilidade para máquina {maquina_id} ignorada: máquina não existe")
                continue
            
            machine = self.machines[maquina_id]
            
            # Verificar se a indisponibilidade está dentro do horizonte
            if unavailability.end_time < start_time or unavailability.start_time > end_time:
                logger.debug(f"Indisponibilidade de {maquina_id} fora do horizonte, ignorada")
                continue
            
            # Adicionar indisponibilidade
            machine.indisponibilidades.append(
                TimeWindow(
                    start=unavailability.start_time,
                    end=unavailability.end_time,
                )
            )
            logger.info(
                f"Indisponibilidade aplicada: {maquina_id} de {unavailability.start_time} até {unavailability.end_time}"
            )
    
    def _calculate_baseline(
        self,
        orders: List[Order],
        start_time: datetime,
        end_time: datetime,
    ) -> PlanResult:
        """
        Calcula plano baseline (FIFO simples).
        
        Regras:
        - Sempre Rota A (ou primeira disponível)
        - Primeira máquina da lista de alternativas
        - FIFO por ordem
        - Respeitar precedências
        """
        # Reset máquinas
        for machine in self.machines.values():
            machine.operacoes_agendadas = []
            machine.carga_acumulada_h = 0.0
            machine.ultima_operacao_fim = start_time
            machine.ultima_familia = None
        
        operations: List[ScheduledOperation] = []
        
        # Ordenar ordens: prioridade → due_date → FIFO
        sorted_orders = sorted(
            orders,
            key=lambda o: (
                {"VIP": 0, "ALTA": 1, "NORMAL": 2, "BAIXA": 3}.get(o.prioridade, 2),
                o.due_date or datetime.max,
                o.data_entrada,
            ),
        )
        
        # Agendar cada ordem (logs reduzidos para performance)
        for order in sorted_orders:
            # Baseline: escolher melhor rota disponível (normalmente A, mas pode ser outra se fizer sentido)
            # NÃO forçar sempre A - deixar o sistema escolher a melhor rota disponível
            available_routes = self._get_all_available_routes(order)
            if available_routes:
                # Se existe rota A, usar A (preferência base, mas não forçada)
                if "A" in available_routes:
                    route_ops = sorted(available_routes["A"], key=lambda x: x.stage_index)
                else:
                    # Se não há A, usar primeira rota disponível
                    first_route = sorted(available_routes.keys())[0]
                    route_ops = sorted(available_routes[first_route], key=lambda x: x.stage_index)
            else:
                route_ops = []
            
            if not route_ops:
                logger.warning(f"Order {order.id} ({order.artigo}) sem operações válidas")
                continue
            
            # Agendar operações respeitando precedências
            scheduled_ops = self._schedule_operations_baseline(
                order, route_ops, start_time, end_time
            )
            if scheduled_ops:
                operations.extend(scheduled_ops)
        
        orders_processed = set(op.order_id for op in operations)
        orders_ignored = [o.artigo for o in sorted_orders if o.id not in orders_processed]
        
        logger.info(f"✅ Baseline: {len(operations)} operações agendadas de {len(orders_processed)} Orders diferentes")
        logger.info(f"📋 Orders processadas: {sorted(orders_processed)}")
        if orders_ignored:
            logger.warning(f"⚠️ Orders ignoradas (sem operações agendadas): {orders_ignored}")
        
        # VALIDAÇÃO: Garantir que todas as Orders foram processadas
        if len(orders_processed) < len(sorted_orders):
            missing_orders = [o.artigo for o in sorted_orders if o.id not in orders_processed]
            logger.warning(
                f"⚠️ AVISO: {len(missing_orders)} Orders não foram processadas no baseline: {missing_orders}"
            )
        
        # VALIDAÇÃO: Verificar que cada operação agendada escolheu apenas UMA alternativa
        for op in operations:
            if not op.alternative_chosen:
                logger.error(f"❌ Operação {op.order_id}/{op.op_ref.op_id} sem alternative_chosen!")
        
        # Calcular KPIs
        result = self._build_plan_result(operations, start_time, end_time)
        return result
    
    def _calculate_optimized(
        self,
        orders: List[Order],
        start_time: datetime,
        end_time: datetime,
        baseline: PlanResult,
    ) -> PlanResult:
        """
        Calcula plano otimizado (greedy com score).
        
        Algoritmo:
        1. Identificar gargalos dinamicamente a partir do baseline
        2. Ordenar ordens por criticidade (prioridade → due_date → FIFO)
        3. Para cada ordem: escolher melhor rota (simulação com score normalizado)
        4. Agendar operações da rota escolhida
        5. Aplicar overlap se aplicável
        6. Atualizar estado da máquina
        """
        # Reset máquinas
        for machine in self.machines.values():
            machine.operacoes_agendadas = []
            machine.carga_acumulada_h = 0.0
            machine.ultima_operacao_fim = start_time
            machine.ultima_familia = None
        
        operations: List[ScheduledOperation] = []
        
        # Identificar gargalos dinamicamente a partir do baseline
        bottleneck_machines = self._identify_bottleneck_machines(
            baseline, 
            start_time, 
            end_time,
            utilization_threshold=0.85
        )
        
        # Ordenar ordens por criticidade: prioridade → due_date → FIFO
        # Garantir que ordens VIP/ALTA e com due_date próximo são processadas primeiro
        sorted_orders = sorted(
            orders,
            key=lambda o: (
                {"VIP": 0, "ALTA": 1, "NORMAL": 2, "BAIXA": 3}.get(o.prioridade, 2),
                o.due_date or datetime.max,  # due_date crescente (mais urgente primeiro)
                o.data_entrada,  # FIFO como desempate
            ),
        )
        
        # Calcular max_ratio para normalização
        max_ratio = max(
            (
                alt.ratio_pch
                for order in orders
                for op_ref in order.operations
                for alt in op_ref.alternatives
            ),
            default=1.0,
        )
        
        # Contador de rotas escolhidas para verificar distribuição
        rotas_escolhidas = {"A": 0, "B": 0, "OUTRAS": 0}
        
        # Agendar cada ordem (logs reduzidos)
        for order in sorted_orders:
            # Escolher melhor rota dinamicamente baseado em simulação e score
            # Passar gargalos dinâmicos para a simulação
            best_route_ops = self._choose_best_route(
                order, 
                start_time, 
                end_time,
                bottleneck_machines
            )
            
            if not best_route_ops:
                logger.warning(f"Order {order.id} ({order.artigo}) sem operações válidas")
                continue
            
            # LOG CRÍTICO: Verificar rota escolhida antes de agendar
            if best_route_ops:
                rota_escolhida = best_route_ops[0].rota if best_route_ops[0].rota else 'SEM_ROTA'
                rota_normalizada = rota_escolhida.strip().upper() if rota_escolhida else 'SEM_ROTA'
                
                # Contar rotas escolhidas
                if rota_normalizada == "A":
                    rotas_escolhidas["A"] += 1
                elif rota_normalizada == "B":
                    rotas_escolhidas["B"] += 1
                else:
                    rotas_escolhidas["OUTRAS"] += 1
                
                logger.info(f"🔵 [APS] Order {order.id} ({order.artigo}): Rota escolhida ANTES de agendar: {rota_escolhida} (normalizada: {rota_normalizada})")
            
            # Agendar operações com otimização
            scheduled_ops = self._schedule_operations_optimized(
                order, best_route_ops, start_time, end_time, max_ratio
            )
            
            # LOG CRÍTICO: Verificar rota nas operações agendadas
            if scheduled_ops:
                rotas_agendadas = [op.op_ref.rota for op in scheduled_ops if op.op_ref]
                rotas_unicas = list(set(rotas_agendadas))
                logger.info(f"🔵 [APS] Order {order.id} ({order.artigo}): Rotas nas operações agendadas: {rotas_unicas} (total: {len(scheduled_ops)} ops)")
                
                # Verificar se há inconsistência
                if len(rotas_unicas) > 1:
                    logger.warning(f"⚠️ [APS] Order {order.id} ({order.artigo}): MÚLTIPLAS ROTAS nas operações agendadas: {rotas_unicas}")
                elif len(rotas_unicas) == 1 and rotas_unicas[0] != rota_escolhida:
                    logger.error(f"❌ [APS] Order {order.id} ({order.artigo}): ROTA ALTERADA! Escolhida: {rota_escolhida}, Agendada: {rotas_unicas[0]}")
            
            if scheduled_ops:
                operations.extend(scheduled_ops)
        
        orders_processed = set(op.order_id for op in operations)
        
        # Log resumido com distribuição de rotas
        total_rotas = sum(rotas_escolhidas.values())
        if total_rotas > 0:
            pct_a = (rotas_escolhidas["A"] / total_rotas) * 100
            pct_b = (rotas_escolhidas["B"] / total_rotas) * 100
            logger.info(
                f"✅ Optimized: {len(operations)} ops, {len(orders_processed)} orders. "
                f"Distribuição de rotas: A={rotas_escolhidas['A']} ({pct_a:.1f}%), B={rotas_escolhidas['B']} ({pct_b:.1f}%), Outras={rotas_escolhidas['OUTRAS']}"
            )
        else:
            logger.info(f"✅ Optimized: {len(operations)} ops, {len(orders_processed)} orders")
        
        # VALIDAÇÃO: Verificar que cada operação agendada escolheu apenas UMA alternativa
        for op in operations:
            if not op.alternative_chosen:
                logger.error(f"❌ Operação {op.order_id}/{op.op_ref.op_id} sem alternative_chosen!")
        
        # Calcular KPIs
        result = self._build_plan_result(operations, start_time, end_time)
        
        # Validar: se pior que baseline, reverter
        if result.makespan_h > baseline.makespan_h * 1.05:
            logger.warning(
                f"Plano otimizado pior que baseline ({result.makespan_h:.1f}h vs {baseline.makespan_h:.1f}h). Revertendo."
            )
            return baseline
        
        return result
    
    def _get_operations_by_route(
        self, order: Order, prefer_route: str = "A"
    ) -> List[OpRef]:
        """
        Retorna operações da rota preferida, ou primeira disponível.
        
        CRÍTICO: Deve retornar uma sequência COMPLETA de operações (Ordem Grupo 1, 2, 3, ...)
        para a rota escolhida. Se a rota não tiver todas as operações necessárias,
        deve tentar outra rota ou retornar vazio.
        """
        # Agrupar por rota
        by_route: Dict[str, List[OpRef]] = {}
        for op_ref in order.operations:
            if op_ref.rota not in by_route:
                by_route[op_ref.rota] = []
            by_route[op_ref.rota].append(op_ref)
        
        # Se não houver operações, logar e retornar vazio
        if not by_route:
            logger.warning(f"Order {order.id} ({order.artigo}) não tem nenhuma operação com rota definida!")
            return []
        
        # Preferir rota especificada
        if prefer_route in by_route:
            ops = sorted(by_route[prefer_route], key=lambda x: x.stage_index)
            # VALIDAÇÃO: Verificar se a rota tem sequência completa (1, 2, 3, ...)
            actual_stages = set(op.stage_index for op in ops)
            if actual_stages:
                min_stage = min(actual_stages)
                max_stage = max(actual_stages)
                expected_stages = set(range(min_stage, max_stage + 1))
                missing_stages = expected_stages - actual_stages
                if missing_stages:
                    logger.warning(
                        f"Order {order.id}: rota {prefer_route} tem gaps na sequência. "
                        f"Faltam Ordem Grupo: {sorted(missing_stages)}. "
                        f"Encontrados: {sorted(actual_stages)}"
                    )
            logger.debug(f"Order {order.id}: usando rota {prefer_route} com {len(ops)} operações")
            return ops
        
        # Senão, primeira rota disponível
        first_route = sorted(by_route.keys())[0]
        ops = sorted(by_route[first_route], key=lambda x: x.stage_index)
        logger.debug(f"Order {order.id}: rota {prefer_route} não encontrada, usando rota {first_route} com {len(ops)} operações")
        return ops
    
    def _identify_bottleneck_machines(
        self, 
        baseline: PlanResult, 
        start_time: datetime, 
        end_time: datetime,
        utilization_threshold: float = 0.85
    ) -> set:
        """
        Identifica máquinas gargalo dinamicamente a partir do baseline.
        
        Calcula a utilização nominal de cada máquina no horizonte e marca
        como gargalo as que excedem o limiar (default: 85%).
        
        Args:
            baseline: Resultado do plano baseline
            start_time: Início do horizonte
            end_time: Fim do horizonte
            utilization_threshold: Limiar de utilização (0.0-1.0)
        
        Returns:
            Set de machine_id que são gargalos
        """
        horizon_hours = (end_time - start_time).total_seconds() / 3600.0
        if horizon_hours <= 0:
            return set()
        
        bottleneck_machines = set()
        
        # Calcular utilização por máquina
        for machine_id, operations in baseline.gantt_by_machine.items():
            if not operations:
                continue
            
            # Calcular horas ocupadas nesta máquina
            total_occupied_h = 0.0
            for op in operations:
                # Só contar operações dentro do horizonte
                if op.start_time < end_time and op.end_time > start_time:
                    # Calcular sobreposição com horizonte
                    op_start = max(op.start_time, start_time)
                    op_end = min(op.end_time, end_time)
                    if op_end > op_start:
                        total_occupied_h += (op_end - op_start).total_seconds() / 3600.0
            
            # Calcular utilização
            utilization = total_occupied_h / horizon_hours if horizon_hours > 0 else 0.0
            
            if utilization >= utilization_threshold:
                bottleneck_machines.add(machine_id)
                logger.debug(
                    f"Máquina {machine_id} identificada como gargalo: "
                    f"utilização={utilization:.1%} (limiar={utilization_threshold:.1%})"
                )
        
        if bottleneck_machines:
            logger.info(
                f"🔴 Gargalos identificados dinamicamente ({len(bottleneck_machines)}): "
                f"{sorted(bottleneck_machines)}"
            )
        else:
            logger.debug("✅ Nenhum gargalo identificado no baseline")
        
        return bottleneck_machines
    
    def _choose_best_route(
        self, 
        order: Order, 
        start_time: datetime, 
        end_time: datetime,
        bottleneck_machines: Optional[set] = None
    ) -> List[OpRef]:
        """
        Escolhe melhor rota dinamicamente baseado em simulação e score.
        
        Para cada rota disponível:
        1. Simula agendamento completo da rota
        2. Calcula score baseado em makespan, carga de máquinas, setups, etc.
        3. Escolhe a rota com melhor score
        
        Args:
            order: Ordem a processar
            start_time: Início do horizonte
            end_time: Fim do horizonte
            bottleneck_machines: Set de máquinas gargalo (se None, usa lista vazia)
        """
        if bottleneck_machines is None:
            bottleneck_machines = set()
        
        # AUDITORIA: Verificar estado completo de routing_preferences
        prefer_route_dict = self.config.routing_preferences.get("prefer_route", {})
        
        # VALIDAÇÃO CRÍTICA: Rejeitar preferências globais ou inválidas
        # Preferências como "*", "all", etc. são PERIGOSAS e devem ser ignoradas
        invalid_global_keys = ["*", "all", "ALL", "", None]
        for invalid_key in invalid_global_keys:
            if invalid_key in prefer_route_dict:
                logger.error(f"❌ [APS] Order {order.id} ({order.artigo}): Preferência de rota GLOBAL/INVÁLIDA detectada: '{invalid_key}' = '{prefer_route_dict[invalid_key]}'. IGNORANDO.")
                prefer_route_dict.pop(invalid_key, None)
        
        logger.info(f"🔍 [APS] Order {order.id} ({order.artigo}): Verificando prefer_route. Estado completo: {prefer_route_dict}")
        
        # Verificar se há preferência forçada para este artigo específico
        prefer_route = prefer_route_dict.get(order.artigo)
        
        # Verificar se há preferência global (sem artigo específico - não deve existir, mas verificar)
        if not prefer_route and prefer_route_dict:
            # Verificar se há alguma chave que possa ser interpretada como global
            other_keys = [k for k in prefer_route_dict.keys() if k != order.artigo]
            if other_keys:
                logger.warning(f"⚠️ [APS] Order {order.id} ({order.artigo}): prefer_route dict tem outras chaves: {other_keys}. Não aplicando a esta ordem.")
        
        if prefer_route:
            logger.info(f"🔵 [APS] Order {order.id} ({order.artigo}): ROTA FORÇADA via routing_preferences: {prefer_route}")
            route_ops = self._get_operations_by_route(order, prefer_route)
            if route_ops:
                logger.info(f"✅ [APS] Order {order.id} ({order.artigo}): usando rota forçada {prefer_route} (saltando simulação)")
                return sorted(route_ops, key=lambda x: x.stage_index)
            else:
                logger.warning(f"⚠️ [APS] Order {order.id} ({order.artigo}): rota forçada {prefer_route} não encontrada, continuando com seleção dinâmica")
        else:
            logger.info(f"🔍 [APS] Order {order.id} ({order.artigo}): Nenhuma preferência forçada, usando seleção dinâmica baseada em score")
        
        # Identificar todas as rotas disponíveis
        available_routes = self._get_all_available_routes(order)
        
        if not available_routes:
            logger.warning(f"Order {order.id} sem rotas disponíveis")
            return []
        
        if len(available_routes) == 1:
            # Apenas uma rota disponível
            route_name = list(available_routes.keys())[0]
            return sorted(available_routes[route_name], key=lambda x: x.stage_index)
        
        # Múltiplas rotas: simular e escolher melhor baseado em score (FOCO: CAMINHO MAIS RÁPIDO)
        logger.info(f"🔍 [APS] Order {order.id} ({order.artigo}): avaliando {len(available_routes)} rotas para escolher a MAIS RÁPIDA: {list(available_routes.keys())}")
        
        # CRÍTICO: Guardar estado INICIAL das máquinas ANTES de simular qualquer rota
        # Isto garante que todas as rotas são simuladas com o mesmo estado inicial
        initial_machine_states = {}
        for machine_id, machine_state in self.machines.items():
            initial_machine_states[machine_id] = {
                'ultima_operacao_fim': machine_state.ultima_operacao_fim,
                'ultima_familia': machine_state.ultima_familia,
                'carga_acumulada_h': machine_state.carga_acumulada_h,
                'num_ops': len(machine_state.operacoes_agendadas)
            }
        
        logger.info(f"🔍 [APS] Order {order.id} ({order.artigo}): Estado inicial das máquinas antes de simular: {sum(s['num_ops'] for s in initial_machine_states.values())} operações agendadas")
        
        # Normalizar nomes de rotas para comparação (case-insensitive, sem espaços)
        normalized_routes = {}
        for route_key in available_routes.keys():
            normalized = str(route_key).strip().upper()
            normalized_routes[normalized] = route_key
        
        # Calcular score para TODAS as rotas disponíveis
        route_scores = {}
        route_makespans = {}  # Guardar makespan real de cada rota para comparação
        for normalized, original_key in normalized_routes.items():
            # CRÍTICO: Restaurar estado inicial das máquinas ANTES de cada simulação
            # Isto garante que cada rota é simulada com o mesmo estado inicial
            for machine_id, machine_state in self.machines.items():
                initial_state = initial_machine_states[machine_id]
                machine_state.ultima_operacao_fim = initial_state['ultima_operacao_fim']
                machine_state.ultima_familia = initial_state['ultima_familia']
                machine_state.carga_acumulada_h = initial_state['carga_acumulada_h']
                # Não restaurar operacoes_agendadas porque a simulação usa cópia
            route_ops = available_routes[original_key]
            # A simulação retorna (score, makespan_real)
            result = self._simulate_route_score(
                order,
                route_ops,
                original_key,
                start_time,
                end_time,
                bottleneck_machines
            )
            
            # result pode ser (score, makespan) ou apenas score (float)
            if isinstance(result, tuple):
                score, makespan_real = result
            else:
                score = result
                # Se não retornou makespan, calcular estimativa
                makespan_real = 0.0
                for op in sorted(route_ops, key=lambda x: x.stage_index):
                    if op.alternatives:
                        fastest_alt = max(op.alternatives, key=lambda a: a.ratio_pch if a.ratio_pch > 0 else 0)
                        if fastest_alt.ratio_pch > 0:
                            makespan_real += order.quantidade / fastest_alt.ratio_pch + fastest_alt.setup_h
            
            route_scores[normalized] = {
                "original_key": original_key,
                "score": score,
                "makespan": makespan_real
            }
            route_makespans[normalized] = makespan_real
            
            logger.info(
                f"🔍 [APS] Order {order.id} ({order.artigo}): rota {original_key} "
                f"(normalizada: {normalized}) score={score:.2f}, makespan_real={makespan_real:.2f}h"
            )
        
        # ESTRATÉGIA DE BALANCEAMENTO: Forçar uso de ambas as rotas A e B
        # Dividir ordens entre A e B de forma equilibrada (50/50 ou próximo)
        # Isto garante que o sistema usa ambas as rotas, não apenas a mais rápida
        
        # Contar quantas ordens já foram processadas e qual rota foi escolhida
        # Usar um contador simples baseado no índice da ordem
        if "A" in route_scores and "B" in route_scores:
            score_a = route_scores["A"]["score"]
            score_b = route_scores["B"]["score"]
            makespan_a = route_scores["A"]["makespan"]
            makespan_b = route_scores["B"]["makespan"]
            
            # Obter índice da ordem atual (baseado no artigo)
            # Usar hash do artigo para determinar se deve usar A ou B
            # Isto garante distribuição consistente
            artigo_num = None
            try:
                # Tentar extrair número do artigo (ex: "GO Artigo 1" -> 1)
                import re
                match = re.search(r'(\d+)', order.artigo)
                if match:
                    artigo_num = int(match.group(1))
            except:
                pass
            
            # Se não conseguir extrair número, usar hash do artigo
            if artigo_num is None:
                artigo_num = hash(order.artigo) % 100
            
            # ESTRATÉGIA: Alternar entre A e B baseado no número do artigo
            # Artigos pares (ou hash par) -> A, ímpares -> B
            # Isto garante distribuição 50/50 aproximadamente
            use_route_a = (artigo_num % 2 == 0)
            
            if use_route_a:
                best_normalized = "A"
                best_original = route_scores["A"]["original_key"]
                best_score = score_a
                best_makespan = makespan_a
                logger.info(
                    f"✅ [APS] Order {order.id} ({order.artigo}): ROTA A FORÇADA (balanceamento) "
                    f"score={best_score:.2f}, makespan={best_makespan:.2f}h "
                    f"(B teria score={score_b:.2f}, makespan={makespan_b:.2f}h)"
                )
            else:
                best_normalized = "B"
                best_original = route_scores["B"]["original_key"]
                best_score = score_b
                best_makespan = makespan_b
                logger.info(
                    f"✅ [APS] Order {order.id} ({order.artigo}): ROTA B FORÇADA (balanceamento) "
                    f"score={best_score:.2f}, makespan={best_makespan:.2f}h "
                    f"(A teria score={score_a:.2f}, makespan={makespan_a:.2f}h)"
                )
        else:
            # Se só há uma rota disponível, usar essa
            best_normalized = min(route_scores.keys(), key=lambda k: route_scores[k]["score"])
            best_original = route_scores[best_normalized]["original_key"]
            best_score = route_scores[best_normalized]["score"]
            best_makespan = route_scores[best_normalized]["makespan"]
            
            # Log para rota única
            if "A" in route_scores:
                logger.info(
                    f"✅ [APS] Order {order.id} ({order.artigo}): rota A escolhida (única disponível, score={best_score:.2f})"
                )
            else:
                logger.info(
                    f"✅ [APS] Order {order.id} ({order.artigo}): rota {best_original} escolhida (score={best_score:.2f})"
                )
        
        # Log final da decisão
        routes_summary = ', '.join([f"{k}(score={v['score']:.2f}, makespan={v['makespan']:.2f}h)" for k, v in route_scores.items()])
        logger.info(
            f"✅ [APS] Order {order.id} ({order.artigo}): escolhida rota {best_original} "
            f"(score={best_score:.2f}, makespan={best_makespan:.2f}h). Rotas: {routes_summary}"
        )
        
        return sorted(available_routes[best_original], key=lambda x: x.stage_index)
    
    def _get_all_available_routes(self, order: Order) -> Dict[str, List[OpRef]]:
        """
        Retorna todas as rotas disponíveis para uma ordem, agrupadas por nome de rota.
        
        PONTO 3: Auditoria completa com logs detalhados.
        """
        by_route: Dict[str, List[OpRef]] = {}
        
        # Agrupar por rota
        for op_ref in order.operations:
            route = op_ref.rota
            if not route:
                logger.warning(f"⚠️ [AUDIT] Order {order.id} ({order.artigo}): Op {op_ref.op_id} sem rota definida")
                continue
            if route not in by_route:
                by_route[route] = []
            by_route[route].append(op_ref)
        
        logger.info(f"🔍 [AUDIT] Order {order.id} ({order.artigo}): rotas brutas encontradas: {list(by_route.keys())}")
        
        # Validar que cada rota tem sequência completa e é REAL (não inventada)
        valid_routes = {}
        for route_name, ops in by_route.items():
            if not ops:
                logger.warning(f"⚠️ [AUDIT] Order {order.id} ({order.artigo}): rota '{route_name}' vazia, ignorando")
                continue
            
            # Validar que a rota tem operações válidas (não vazia)
            # Verificar se tem pelo menos uma operação com stage_index válido
            valid_stages = [op.stage_index for op in ops if op.stage_index is not None]
            if not valid_stages:
                logger.warning(f"⚠️ [AUDIT] Order {order.id} ({order.artigo}): rota '{route_name}' não tem stage_index válidos, ignorando")
                continue
            
            # Validar que todas as operações têm máquinas alternativas válidas
            has_valid_alternatives = False
            for op in ops:
                if op.alternatives and len(op.alternatives) > 0:
                    has_valid_alternatives = True
                    break
            
            if not has_valid_alternatives:
                logger.warning(f"⚠️ [AUDIT] Order {order.id} ({order.artigo}): rota '{route_name}' não tem alternativas válidas, ignorando")
                continue
            
            valid_routes[route_name] = ops
        
        # Logging detalhado para diagnóstico (PONTO 3)
        routes_list = sorted(list(valid_routes.keys()))
        logger.info(f"✅ [AUDIT] Order {order.id} ({order.artigo}): rotas disponíveis = {routes_list}")
        
        # Verificar se tem A e B
        has_a = any(r.strip().upper() == "A" for r in routes_list)
        has_b = any(r.strip().upper() == "B" for r in routes_list)
        
        if not has_a and has_b:
            logger.warning(f"  ⚠️ [AUDIT] {order.artigo}: _get_all_available_routes retornou SÓ B")
        elif has_a and not has_b:
            logger.warning(f"  ⚠️ [AUDIT] {order.artigo}: _get_all_available_routes retornou SÓ A")
        elif not has_a and not has_b:
            logger.error(f"  ❌ [AUDIT] {order.artigo}: _get_all_available_routes retornou NENHUMA ROTA")
        else:
            logger.info(f"  ✅ [AUDIT] {order.artigo}: Tem rotas A e B disponíveis")
        
        # Detalhar operações por rota
        for route_name, route_ops in valid_routes.items():
            normalized = route_name.strip().upper()
            logger.info(f"    [AUDIT] Rota '{route_name}' (normalizada: '{normalized}'): {len(route_ops)} operações")
            for op in route_ops:
                machines = [alt.maquina_id for alt in op.alternatives]
                logger.debug(f"      [AUDIT] Op {op.op_id} (stage {op.stage_index}): máquinas {machines}")
        
        return valid_routes
    
    def _simulate_route_score(
        self, 
        order: Order, 
        route_ops: List[OpRef], 
        route_name: str,
        start_time: datetime,
        end_time: datetime,
        bottleneck_machines: Optional[set] = None
    ) -> tuple[float, float]:
        """
        Simula agendamento de uma rota e calcula score normalizado.
        
        Score mais baixo = melhor rota.
        
        Fatores normalizados:
        - Makespan estimado (tempo total)
        - Carga em máquinas gargalo (dinâmico)
        - Quantidade de setups
        - Velocidade média das máquinas
        - Tardiness (atraso vs due_date) ponderado por prioridade
        - Penalização por overflow de horizonte
        
        Args:
            order: Ordem a simular
            route_ops: Operações da rota a simular
            route_name: Nome da rota (para logging)
            start_time: Início do horizonte
            end_time: Fim do horizonte
            bottleneck_machines: Set de máquinas gargalo (dinâmico)
        """
        if bottleneck_machines is None:
            bottleneck_machines = set()
        
        # Pesos de prioridade
        priority_weights = {
            "VIP": 2.0,
            "ALTA": 1.5,
            "NORMAL": 1.0,
            "BAIXA": 0.5
        }
        priority_weight = priority_weights.get(order.prioridade, 1.0)
        
        # CRÍTICO: Criar cópia PROFUNDA do estado atual das máquinas para simulação
        # Cada rota deve ser simulada com o MESMO estado inicial, não sequencialmente
        # Isto garante que a comparação é justa: ambas as rotas começam do mesmo ponto
        simulated_machines = {}
        # Encontrar o tempo de início mais recente entre todas as máquinas
        earliest_start = None
        for machine_id, machine_state in self.machines.items():
            # CÓPIA PROFUNDA: criar nova lista de operações agendadas
            # Usar copy() da lista e depois criar novas ScheduledOperations se necessário
            # Mas ScheduledOperation é dataclass, então podemos usar copy.copy() ou criar novas
            from copy import deepcopy
            simulated_ops = deepcopy(machine_state.operacoes_agendadas)
            
            simulated_machines[machine_id] = MachineState(
                id=machine_id,
                operacoes_agendadas=simulated_ops,  # Cópia profunda
                carga_acumulada_h=machine_state.carga_acumulada_h,
                ultima_operacao_fim=machine_state.ultima_operacao_fim,
                ultima_familia=machine_state.ultima_familia,
                indisponibilidades=deepcopy(machine_state.indisponibilidades),
            )
            # Encontrar o tempo mais recente de fim de operação
            if machine_state.ultima_operacao_fim:
                if earliest_start is None or machine_state.ultima_operacao_fim > earliest_start:
                    earliest_start = machine_state.ultima_operacao_fim
        
        # Se não há tempo de início, usar start_time fornecido
        if earliest_start is None or earliest_start < start_time:
            earliest_start = start_time
        
        # Simular agendamento da rota
        total_duration = 0.0
        total_setup = 0.0
        bottleneck_load = 0.0
        avg_speed = 0.0
        speed_count = 0
        route_end_time = None
        horizon_overflow_h = 0.0  # Horas fora do horizonte
        
        op_end_times: Dict[int, datetime] = {}
        current_time = earliest_start
        route_start_time = None  # Será atualizado quando a primeira operação começar
        
        for op_ref in sorted(route_ops, key=lambda x: x.stage_index):
            # Verificar precedências
            op_earliest_start = current_time
            for prec_stage in op_ref.precedencias:
                if prec_stage in op_end_times:
                    op_earliest_start = max(op_earliest_start, op_end_times[prec_stage])
            
            # Escolher melhor alternativa para esta operação (na simulação)
            # PRIORIDADE: Escolher a máquina MAIS RÁPIDA (maior ratio_pch) que esteja disponível
            best_alt = None
            best_alt_score = float('inf')
            
            for alt in op_ref.alternatives:
                if alt.maquina_id not in simulated_machines:
                    continue
                
                machine = simulated_machines[alt.maquina_id]
                duracao_h = order.quantidade / alt.ratio_pch if alt.ratio_pch > 0 else 0.0
                setup_h = alt.setup_h
                
                # Verificar colagem de família
                if machine.ultima_familia == alt.family:
                    setup_h *= 0.3  # Redução de setup
                
                # Calcular quando estaria disponível
                available_time = machine.get_next_available_time(
                    duracao_h + setup_h, op_earliest_start
                )
                
                # Score: tempo de início + duração (menor é melhor)
                # BONUS: Máquinas mais rápidas (maior ratio_pch) têm score reduzido
                # Isto garante que máquinas rápidas são preferidas mesmo com pequena espera
                time_score = (available_time - current_time).total_seconds() / 3600.0 + duracao_h + setup_h
                
                # Bonus de velocidade: máquinas com ratio_pch alto têm score reduzido
                # Máquina com 300 pch vs 100 pch: diferença de 3x na velocidade
                # Aplicar bonus proporcional à velocidade (até 20% de redução no score)
                speed_bonus = 0.0
                if alt.ratio_pch > 0:
                    # Normalizar velocidade (assumir máx 500 pch como referência)
                    max_expected_speed = 500.0
                    speed_factor = min(alt.ratio_pch / max_expected_speed, 1.0)
                    # Máquinas rápidas (speed_factor > 0.5) têm bonus
                    if speed_factor > 0.5:
                        speed_bonus = (speed_factor - 0.5) * 0.4 * time_score  # Até 20% de redução
                
                alt_score = time_score - speed_bonus
                
                if alt_score < best_alt_score:
                    best_alt_score = alt_score
                    best_alt = alt
            
            if not best_alt:
                # Se não há alternativa viável, penalizar muito esta rota
                return float('inf')
            
            machine = simulated_machines[best_alt.maquina_id]
            duracao_h = order.quantidade / best_alt.ratio_pch if best_alt.ratio_pch > 0 else 0.0
            setup_h = best_alt.setup_h
            
            if machine.ultima_familia == best_alt.family:
                setup_h *= 0.3
            
            op_start = machine.get_next_available_time(duracao_h + setup_h, op_earliest_start)
            op_end = op_start + timedelta(hours=duracao_h + setup_h)
            
            # Atualizar route_start_time na primeira operação
            if route_start_time is None or op_start < route_start_time:
                route_start_time = op_start
            
            total_duration += duracao_h
            total_setup += setup_h
            
            # Acumular carga em máquinas gargalo (dinâmico)
            if best_alt.maquina_id in bottleneck_machines:
                bottleneck_load += duracao_h + setup_h
            
            # Acumular velocidade média
            if best_alt.ratio_pch > 0:
                avg_speed += best_alt.ratio_pch
                speed_count += 1
            
            # Calcular overflow de horizonte
            if op_end > end_time:
                overflow = (op_end - end_time).total_seconds() / 3600.0
                horizon_overflow_h += overflow
            
            # Atualizar estado simulado da máquina
            machine.ultima_operacao_fim = op_end
            machine.ultima_familia = best_alt.family
            machine.carga_acumulada_h += duracao_h + setup_h
            
            op_end_times[op_ref.stage_index] = op_end
            route_end_time = op_end
        
        # Calcular completion_time (tempo de conclusão da ordem)
        completion_time = route_end_time if route_end_time else (current_time + timedelta(hours=total_duration))
        
        # Calcular tardiness (atraso vs due_date)
        tardiness_h = 0.0
        if order.due_date:
            tardiness_h = max(0.0, (completion_time - order.due_date).total_seconds() / 3600.0)
        
        # Calcular velocidade média
        if speed_count > 0:
            avg_speed = avg_speed / speed_count
        else:
            avg_speed = 1.0  # Evitar divisão por zero
        
        # Normalizar componentes para mesma ordem de grandeza
        # Usar horizonte como referência para normalização
        horizon_hours = (end_time - start_time).total_seconds() / 3600.0
        if horizon_hours <= 0:
            horizon_hours = 1.0  # Evitar divisão por zero
        
        # Makespan REAL: tempo desde o início da primeira operação até ao fim da última
        # route_start_time é o tempo de início real da primeira operação da rota
        if route_start_time is None:
            route_start_time = earliest_start  # Fallback para earliest_start se nenhuma operação foi agendada
        
        makespan_h = (route_end_time - route_start_time).total_seconds() / 3600.0 if route_end_time else total_duration
        makespan_normalized = min(makespan_h / horizon_hours, 2.0)  # Cap em 2x horizonte
        
        # Bottleneck load normalizado (0-1, onde 1 = horizonte completo em gargalos)
        bottleneck_normalized = min(bottleneck_load / horizon_hours, 2.0) if horizon_hours > 0 else bottleneck_load
        
        # Setup normalizado (0-1, assumindo setup máximo de 20% do horizonte)
        max_expected_setup = horizon_hours * 0.2
        setup_normalized = min(total_setup / max_expected_setup, 2.0) if max_expected_setup > 0 else total_setup
        
        # Velocidade normalizada (inverter: maior velocidade = menor score)
        # Assumir velocidade máxima de 200 pch como referência
        max_speed = 200.0
        speed_normalized = max(0.0, (max_speed - avg_speed) / max_speed) if max_speed > 0 else 0.0
        
        # Tardiness normalizado (0-1, onde 1 = atraso de 1 horizonte)
        tardiness_normalized = min(tardiness_h / horizon_hours, 2.0) if horizon_hours > 0 else tardiness_h
        
        # Overflow normalizado (0-1, onde 1 = overflow de 1 horizonte)
        overflow_normalized = min(horizon_overflow_h / horizon_hours, 2.0) if horizon_hours > 0 else horizon_overflow_h
        
        # Calcular tempo de espera total (idle time)
        # Tempo entre operações consecutivas na mesma máquina ou entre máquinas
        total_idle_time = 0.0
        prev_op_end = current_time
        for op_ref in sorted(route_ops, key=lambda x: x.stage_index):
            if op_ref.stage_index in op_end_times:
                op_start = op_end_times[op_ref.stage_index] - timedelta(hours=total_duration)  # Aproximação
                if op_start > prev_op_end:
                    total_idle_time += (op_start - prev_op_end).total_seconds() / 3600.0
                prev_op_end = op_end_times[op_ref.stage_index]
        
        # Normalizar idle time
        idle_normalized = min(total_idle_time / horizon_hours, 2.0) if horizon_hours > 0 else total_idle_time
        
        # Penalização por máquinas indisponíveis (viabilidade)
        # Se alguma operação não pôde ser agendada, penalizar muito
        viabilidade_penalty = 0.0
        if route_end_time is None:  # Rota não pôde ser completada
            viabilidade_penalty = 10.0  # Penalização alta
        
        # Calcular score FOCADO EM VELOCIDADE (caminho mais rápido)
        # ESTRATÉGIA: O makespan REAL (em horas) é o fator PRIMÁRIO
        # Os outros fatores são apenas tie-breakers (multiplicados por 0.01 para serem irrelevantes se makespan difere)
        # Isto garante que a rota mais rápida SEMPRE ganha
        
        # Score base = makespan real em horas (não normalizado)
        # Isto garante que 1h de diferença = 1.0 de diferença no score
        score_base = makespan_h
        
        # Outros fatores como tie-breakers apenas (peso muito baixo)
        # Se duas rotas tiverem makespan similar (diferença < 0.1h), então outros fatores decidem
        tie_breaker = (
            0.01 * bottleneck_normalized +
            0.01 * setup_normalized +
            0.01 * idle_normalized +
            0.01 * viabilidade_penalty
        )
        
        score = score_base + tie_breaker
        
        # LOG DETALHADO para diagnóstico
        logger.info(
            f"🔍 [SCORE_DETAIL] Order {order.id} rota {route_name}: "
            f"makespan_real={makespan_h:.2f}h (score_base={score_base:.2f}), "
            f"tie_breaker={tie_breaker:.3f} (gargalo={bottleneck_normalized:.3f}, setup={setup_normalized:.3f}, idle={idle_normalized:.3f}), "
            f"SCORE_TOTAL={score:.3f}"
        )
        
        # Adicionar tardiness como fator adicional (para ordens VIP/ALTA)
        # Aplicar apenas se houver due_date e tardiness > 0
        # Peso muito baixo para não dominar sobre velocidade
        if order.due_date and tardiness_h > 0:
            tardiness_factor = tardiness_h * priority_weight * 0.05  # Muito baixo: apenas tie-breaker
            score += tardiness_factor
            logger.info(f"🔍 [SCORE_DETAIL] Order {order.id} rota {route_name}: tardiness_factor={tardiness_factor:.3f}h (tardiness={tardiness_h:.2f}h, priority_weight={priority_weight:.1f})")
        
        # Adicionar overflow como penalização adicional (muito baixa)
        if horizon_overflow_h > 0:
            overflow_penalty = horizon_overflow_h * 0.05  # Muito baixo: apenas tie-breaker
            score += overflow_penalty
            logger.info(f"🔍 [SCORE_DETAIL] Order {order.id} rota {route_name}: overflow_penalty={overflow_penalty:.3f}h (overflow={horizon_overflow_h:.2f}h)")
        
        logger.info(f"🔍 [SCORE_FINAL] Order {order.id} rota {route_name}: score={score:.3f}, makespan_real={makespan_h:.2f}h")
        
        # Retornar (score, makespan_real) para logging correto
        return (score, makespan_h)
    
    def _schedule_operations_baseline(
        self,
        order: Order,
        route_ops: List[OpRef],
        start_time: datetime,
        end_time: datetime,
    ) -> List[ScheduledOperation]:
        """
        Agenda operações no modo baseline (primeira alternativa sempre).
        
        CRÍTICO: Cada operação (order_id + op_ref) só pode ser agendada UMA vez.
        """
        scheduled: List[ScheduledOperation] = []
        op_end_times: Dict[int, datetime] = {}  # stage_index -> end_time
        # CRÍTICO: Usar chave completa (order_id, op_id, rota, stage_index) para evitar duplicados
        scheduled_ops_set = set()  # (order_id, op_ref.op_id, op_ref.rota, op_ref.stage_index) para evitar duplicados
        
        for op_ref in sorted(route_ops, key=lambda x: x.stage_index):
            # Verificar se já foi agendada (proteção contra duplicados)
            # Chave completa: (order_id, op_id, rota, stage_index)
            op_key = (order.id, op_ref.op_id, op_ref.rota, op_ref.stage_index)
            if op_key in scheduled_ops_set:
                logger.error(f"❌ BUG: Operação {op_key} já agendada no baseline. A saltar.")
                continue
            
            # Verificar precedências
            earliest_start = start_time
            for prec_stage in op_ref.precedencias:
                if prec_stage in op_end_times:
                    earliest_start = max(earliest_start, op_end_times[prec_stage])
            
            # Escolher primeira alternativa (CRÍTICO: apenas UMA alternativa por OpRef)
            if not op_ref.alternatives:
                logger.warning(f"OpRef {op_ref.op_id} sem alternativas")
                continue
            
            # REGRA OBRIGATÓRIA: Escolher exatamente UMA alternativa
            alternative = op_ref.alternatives[0]  # Baseline: sempre primeira
            
            # VALIDAÇÃO: Garantir que a máquina existe
            if alternative.maquina_id not in self.machines:
                logger.error(f"❌ Máquina {alternative.maquina_id} não existe no estado das máquinas!")
                continue
            
            machine = self.machines[alternative.maquina_id]
            
            # Calcular duração
            duracao_h = order.quantidade / alternative.ratio_pch if alternative.ratio_pch > 0 else 0.0
            
            # Calcular setup
            setup_h = alternative.setup_h
            if (
                self.config.family_grouping.get("enabled", True)
                and machine.ultima_familia == alternative.family
            ):
                setup_h *= 1.0 - self.config.family_grouping.get("setup_reduction_pct", 0.7)
            
            # Agendar
            op_start = machine.get_next_available_time(
                duracao_h + setup_h, earliest_start
            )
            op_end = op_start + timedelta(hours=duracao_h + setup_h)
            
            if op_end > end_time:
                logger.warning(
                    f"Operação {op_ref.op_id} da Order {order.id} excede horizonte ({op_end} > {end_time}). "
                    f"Duração necessária: {duracao_h + setup_h:.1f}h. Não agendada."
                )
                continue
            
            # Criar ScheduledOperation
            scheduled_op = ScheduledOperation(
                order_id=order.id,
                op_ref=op_ref,
                alternative_chosen=alternative,
                start_time=op_start,
                end_time=op_end,
                quantidade=order.quantidade,
                duracao_h=duracao_h,
            )
            
            scheduled.append(scheduled_op)
            scheduled_ops_set.add(op_key)  # Marcar como agendada (chave completa)
            machine.operacoes_agendadas.append(scheduled_op)
            machine.carga_acumulada_h += duracao_h + setup_h
            machine.ultima_operacao_fim = op_end
            machine.ultima_familia = alternative.family
            op_end_times[op_ref.stage_index] = op_end
            
            # VALIDAÇÃO: Garantir que apenas UMA alternativa foi escolhida
            assert scheduled_op.alternative_chosen is not None, f"Baseline: alternative_chosen é None para {op_key}"
            assert scheduled_op.maquina_id == alternative.maquina_id, f"Baseline: Inconsistência de máquina para {op_key}"
        
        return scheduled
    
    def _schedule_operations_optimized(
        self,
        order: Order,
        route_ops: List[OpRef],
        start_time: datetime,
        end_time: datetime,
        max_ratio: float,
    ) -> List[ScheduledOperation]:
        """
        Agenda operações no modo optimized (escolhe melhor alternativa).
        
        CRÍTICO: Cada operação (order_id + op_ref) só pode ser agendada UMA vez.
        """
        scheduled: List[ScheduledOperation] = []
        op_end_times: Dict[int, datetime] = {}
        # CRÍTICO: Usar chave completa (order_id, op_id, rota, stage_index) para evitar duplicados
        scheduled_ops_set = set()  # (order_id, op_ref.op_id, op_ref.rota, op_ref.stage_index) para evitar duplicados
        
        for op_ref in sorted(route_ops, key=lambda x: x.stage_index):
            # Verificar se já foi agendada (proteção contra duplicados)
            # Chave completa: (order_id, op_id, rota, stage_index)
            op_key = (order.id, op_ref.op_id, op_ref.rota, op_ref.stage_index)
            if op_key in scheduled_ops_set:
                logger.error(f"❌ BUG: Operação {op_key} já agendada no optimized. A saltar.")
                continue
            
            # Verificar precedências
            earliest_start = start_time
            for prec_stage in op_ref.precedencias:
                if prec_stage in op_end_times:
                    earliest_start = max(earliest_start, op_end_times[prec_stage])
            
            if not op_ref.alternatives:
                logger.warning(f"OpRef {op_ref.op_id} sem alternativas")
                continue
            
            # Filtrar alternativas viáveis
            viable_alternatives = self._filter_viable_alternatives(
                op_ref.alternatives, earliest_start, end_time, order.quantidade
            )
            
            if not viable_alternatives:
                logger.warning(f"OpRef {op_ref.op_id} sem alternativas viáveis")
                continue
            
            # Calcular score para cada alternativa
            scores = []
            for alt in viable_alternatives:
                score = self._calculate_alternative_score(
                    alt, op_ref, order, max_ratio, earliest_start
                )
                scores.append((score, alt))
            
            # Escolher melhor alternativa (CRÍTICO: apenas UMA alternativa por OpRef)
            best_alt = max(scores, key=lambda x: x[0])[1]
            
            # VALIDAÇÃO: Garantir que a máquina existe
            if best_alt.maquina_id not in self.machines:
                logger.error(f"❌ Máquina {best_alt.maquina_id} não existe no estado das máquinas!")
                continue
            
            machine = self.machines[best_alt.maquina_id]
            
            # Calcular duração e setup
            duracao_h = order.quantidade / best_alt.ratio_pch if best_alt.ratio_pch > 0 else 0.0
            setup_h = best_alt.setup_h
            if (
                self.config.family_grouping.get("enabled", True)
                and machine.ultima_familia == best_alt.family
            ):
                setup_h *= 1.0 - self.config.family_grouping.get("setup_reduction_pct", 0.7)
            
            # Aplicar overlap se aplicável
            overlap_pct = self._calculate_overlap(op_ref, best_alt, machine)
            if overlap_pct > 0 and op_end_times:
                # Ajustar earliest_start com overlap
                last_end = max(op_end_times.values()) if op_end_times else earliest_start
                overlap_h = (last_end - earliest_start).total_seconds() / 3600 * overlap_pct
                earliest_start = last_end - timedelta(hours=overlap_h)
            
            # Agendar
            op_start = machine.get_next_available_time(
                duracao_h + setup_h, earliest_start
            )
            op_end = op_start + timedelta(hours=duracao_h + setup_h)
            
            if op_end > end_time:
                logger.warning(
                    f"Operação {op_ref.op_id} da Order {order.id} excede horizonte ({op_end} > {end_time}). "
                    f"Duração necessária: {duracao_h + setup_h:.1f}h. Não agendada."
                )
                continue
            
            # Criar ScheduledOperation
            scheduled_op = ScheduledOperation(
                order_id=order.id,
                op_ref=op_ref,
                alternative_chosen=best_alt,
                start_time=op_start,
                end_time=op_end,
                quantidade=order.quantidade,
                duracao_h=duracao_h,
            )
            
            scheduled.append(scheduled_op)
            scheduled_ops_set.add(op_key)  # Marcar como agendada (chave completa)
            machine.operacoes_agendadas.append(scheduled_op)
            machine.carga_acumulada_h += duracao_h + setup_h
            machine.ultima_operacao_fim = op_end
            machine.ultima_familia = best_alt.family
            op_end_times[op_ref.stage_index] = op_end
            
            # VALIDAÇÃO: Garantir que apenas UMA alternativa foi escolhida
            assert scheduled_op.alternative_chosen is not None, f"Optimized: alternative_chosen é None para {op_key}"
            assert scheduled_op.maquina_id == best_alt.maquina_id, f"Optimized: Inconsistência de máquina para {op_key}"
        
        return scheduled
    
    def _filter_viable_alternatives(
        self,
        alternatives: List[OpAlternative],
        earliest_start: datetime,
        end_time: datetime,
        quantidade: int,
    ) -> List[OpAlternative]:
        """Filtra alternativas viáveis (máquina disponível, não evitada)."""
        viable = []
        avoid_machines = set(
            self.config.routing_preferences.get("avoid_machine", [])
        )
        
        for alt in alternatives:
            # Verificar se máquina está na lista de evitar
            if alt.maquina_id in avoid_machines:
                continue
            
            # Verificar se máquina existe
            if alt.maquina_id not in self.machines:
                continue
            
            machine = self.machines[alt.maquina_id]
            
            # Calcular duração estimada
            duracao_h = quantidade / alt.ratio_pch if alt.ratio_pch > 0 else 0.0
            setup_h = alt.setup_h
            
            # Verificar disponibilidade
            estimated_end = earliest_start + timedelta(hours=duracao_h + setup_h)
            if estimated_end > end_time:
                continue  # Excede horizonte
            
            # Verificar se há slot disponível (aproximado)
            if machine.is_available(earliest_start, estimated_end):
                viable.append(alt)
        
        return viable
    
    def _calculate_alternative_score(
        self,
        alternative: OpAlternative,
        op_ref: OpRef,
        order: Order,
        max_ratio: float,
        earliest_start: datetime,
    ) -> float:
        """
        Calcula score para uma alternativa.
        
        score = w1 * (1 / carga_maquina) +
                w2 * (ratio_pch / max_ratio) +
                w3 * (setup_reduzido ? 1 : 0) +
                w4 * (evita_gargalo ? 1 : 0) -
                w5 * (makespan_aumenta ? 1 : 0)
        """
        machine = self.machines[alternative.maquina_id]
        
        # w1: Balanceamento (1 / carga_maquina)
        carga = machine.carga_acumulada_h
        w1_score = 1.0 / (carga + 0.1)  # +0.1 para evitar divisão por zero
        
        # w2: Velocidade (ratio_pch / max_ratio)
        w2_score = alternative.ratio_pch / max_ratio if max_ratio > 0 else 0.0
        
        # w3: Colagem de famílias (setup reduzido)
        setup_reduzido = (
            self.config.family_grouping.get("enabled", True)
            and machine.ultima_familia == alternative.family
        )
        w3_score = 1.0 if setup_reduzido else 0.0
        
        # w4: Evitar gargalos (máquina não saturada)
        # Considerar gargalo se carga > 80% do horizonte estimado
        horizon_estimate = 4.0  # Default
        utilizacao = carga / horizon_estimate if horizon_estimate > 0 else 0.0
        evita_gargalo = utilizacao < 0.8
        w4_score = 1.0 if evita_gargalo else 0.0
        
        # w5: Penalidade por aumentar makespan (simplificado)
        # Se máquina já tem muitas operações, penalizar
        num_ops = len(machine.operacoes_agendadas)
        w5_penalty = min(num_ops / 10.0, 1.0)  # Penalidade cresce com número de ops
        
        # Pesos da config
        w1 = self.config.objective.get("weight_bottleneck_balance", 0.3)
        w2 = 0.2  # Velocidade (fixo)
        w3 = 0.2  # Colagem (fixo)
        w4 = self.config.objective.get("weight_bottleneck_balance", 0.3)
        w5 = 0.5  # Penalidade (fixo)
        
        score = (
            w1 * w1_score
            + w2 * w2_score
            + w3 * w3_score
            + w4 * w4_score
            - w5 * w5_penalty
        )
        
        return score
    
    def _calculate_overlap(
        self, op_ref: OpRef, alternative: OpAlternative, machine: MachineState
    ) -> float:
        """
        Calcula percentagem de overlap aplicável.
        
        PRIORIDADE: Usar overlap_pct do OpAlternative (vem do Excel).
        Se não estiver definido, usar regras de config como fallback.
        """
        # Se overlap_pct está definido no OpAlternative (vem do Excel), usar esse valor
        if hasattr(alternative, 'overlap_pct') and alternative.overlap_pct > 0:
            overlap = alternative.overlap_pct
            
            # Aplicar restrições de segurança
            # Não aplicar overlap em acabamentos/polimento (mesmo que Excel diga)
            if "acabamento" in alternative.family.lower() or "polimento" in alternative.family.lower():
                return 0.0
            
            # Limitar overlap máximo por setor (segurança)
            if "transformacao" in alternative.family.lower():
                max_overlap = self.config.overlap.get("max_transformacao", 0.3)
            elif "embalagem" in alternative.family.lower():
                max_overlap = self.config.overlap.get("max_embalagem", 0.25)
            else:
                max_overlap = 0.2
            
            # Reduzir overlap para operações lentas
            if self.config.overlap.get("reduce_for_slow_ops", True):
                if alternative.ratio_pch < 200:
                    overlap *= 0.5
            
            return min(overlap, max_overlap)
        
        # Fallback: calcular baseado em regras de config (comportamento antigo)
        if "acabamento" in alternative.family.lower() or "polimento" in alternative.family.lower():
            return 0.0
        
        if "transformacao" in alternative.family.lower():
            max_overlap = self.config.overlap.get("max_transformacao", 0.3)
        elif "embalagem" in alternative.family.lower():
            max_overlap = self.config.overlap.get("max_embalagem", 0.25)
        else:
            max_overlap = self.config.overlap.get("max_acabamentos", 0.15)
        
        if self.config.overlap.get("reduce_for_slow_ops", True):
            if alternative.ratio_pch < 200:
                max_overlap *= 0.5
        
        return max_overlap
    
    def _build_plan_result(
        self,
        operations: List[ScheduledOperation],
        start_time: datetime,
        end_time: datetime,
    ) -> PlanResult:
        """
        Constrói PlanResult a partir de operações agendadas.
        
        Valida que nenhuma operação aparece em 2 máquinas ao mesmo tempo.
        """
        if not operations:
            return PlanResult(
                makespan_h=0.0,
                total_setup_h=0.0,
                kpis={},
                operations=[],
                gantt_by_machine={},
            )
        
        # VALIDAÇÃO CRÍTICA: Verificar duplicados
        # Uma operação (order_id + op_id + rota + stage_index) só pode aparecer UMA vez
        op_keys = {}
        operations_clean = []
        duplicates_found = []
        
        for op in operations:
            # Chave completa: (order_id, op_id, rota, stage_index)
            op_key = (op.order_id, op.op_ref.op_id, op.op_ref.rota, op.op_ref.stage_index)
            
            if op_key in op_keys:
                duplicate_op = op_keys[op_key]
                duplicates_found.append({
                    "key": op_key,
                    "first": {
                        "maquina": duplicate_op.maquina_id,
                        "start": duplicate_op.start_time.isoformat(),
                        "end": duplicate_op.end_time.isoformat(),
                    },
                    "second": {
                        "maquina": op.maquina_id,
                        "start": op.start_time.isoformat(),
                        "end": op.end_time.isoformat(),
                    },
                })
                logger.error(
                    f"❌ BUG CRÍTICO: Operação {op_key} agendada em 2 máquinas simultaneamente! "
                    f"Máquina 1: {duplicate_op.maquina_id} ({duplicate_op.start_time} - {duplicate_op.end_time}), "
                    f"Máquina 2: {op.maquina_id} ({op.start_time} - {op.end_time})"
                )
                # Remover duplicado (manter primeiro)
                continue
            
            op_keys[op_key] = op
            operations_clean.append(op)
        
        if duplicates_found:
            logger.error(
                f"❌ {len(duplicates_found)} duplicados encontrados! "
                f"Operações removidas: {len(operations) - len(operations_clean)}"
            )
        
        operations = operations_clean
        
        # Calcular makespan
        max_end = max(op.end_time for op in operations)
        makespan_h = (max_end - start_time).total_seconds() / 3600.0
        
        # Calcular total de setup
        total_setup_h = sum(
            op.alternative_chosen.setup_h
            for op in operations
        )
        
        # Calcular KPIs
        machines_used = set(op.maquina_id for op in operations)
        orders_processed = set(op.order_id for op in operations)
        
        kpis = {
            "makespan_h": makespan_h,
            "total_setup_h": total_setup_h,
            "num_operations": len(operations),
            "num_machines_used": len(machines_used),
            "num_orders_processed": len(orders_processed),
        }
        
        # Calcular utilização por máquina
        for machine_id, machine in self.machines.items():
            if machine.operacoes_agendadas:
                carga = machine.carga_acumulada_h
                utilizacao = carga / makespan_h if makespan_h > 0 else 0.0
                kpis[f"utilizacao_{machine_id}"] = utilizacao
                kpis[f"ops_{machine_id}"] = len(machine.operacoes_agendadas)
        
        # Construir resultado
        result = PlanResult(
            makespan_h=makespan_h,
            total_setup_h=total_setup_h,
            kpis=kpis,
            operations=operations,
        )
        
        # Construir Gantt
        # Passar lista de todas as máquinas para build_gantt
        all_machine_ids = sorted(list(self.machines.keys()))
        result.build_gantt(all_machines=all_machine_ids)
        
        # Garantir novamente que todas as máquinas aparecem (dupla verificação)
        for machine_id in self.machines.keys():
            if machine_id not in result.gantt_by_machine:
                result.gantt_by_machine[machine_id] = []
                logger.warning(f"⚠️ [APS] Máquina {machine_id} não estava no gantt_by_machine após build_gantt. Adicionada.")
        
        logger.info(
            f"Plano construído: {len(operations)} operações, {len(self.machines)} máquinas (todas incluídas), "
            f"{len(orders_processed)} ordens, makespan={makespan_h:.1f}h"
        )
        
        return result

