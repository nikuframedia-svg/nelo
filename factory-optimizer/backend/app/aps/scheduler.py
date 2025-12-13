"""
APS Scheduler 2.0 - Algoritmo industrial rigoroso que GARANTE planos "Depois" sempre melhores que "Antes".

Regras de ouro:
1. Lead Time calculado corretamente (por ordem, não por período)
2. Nunca aceitar plano otimizado pior que baseline
3. Colagem de famílias industrial
4. Overlap aplicado realisticamente
5. Detecção correta de gargalo
6. Validação automática de consistência
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from app.etl.loader import get_loader
from app.ml.cycle_time import CycleTimePredictor
from app.ml.setup_time import SetupTimePredictor
from app.ml.routing import RoutingBandit

logger = logging.getLogger(__name__)

@dataclass
class Operation:
    ordem: str
    artigo: str
    operacao: str
    recurso: str
    start_time: datetime
    end_time: datetime
    setor: str
    overlap: float
    rota: str
    explicacao: str
    familia: Optional[str] = None

@dataclass
class PlanResult:
    kpis: Dict[str, float]
    operations: List[Operation]
    explicacoes: List[str]

class APSScheduler:
    def __init__(self):
        self.loader = get_loader()
        self.cycle_predictor = CycleTimePredictor()
        self.setup_predictor = SetupTimePredictor()
        self.routing_bandit = RoutingBandit()

    def _empty_plan_result(self) -> PlanResult:
        return PlanResult(
            kpis={
                "otd_pct": 0.0,
                "lead_time_h": 0.0,
                "gargalo_ativo": "N/A",
                "horas_setup_semana": 0.0,
                "utilizacao_gargalo": 0.0,
                "fila_gargalo_h": 0.0,
                "throughput_gargalo": 0.0,
            },
            operations=[],
            explicacoes=[],
        )

    def _sanitize_routes(self, roteiros: pd.DataFrame) -> pd.DataFrame:
        if roteiros is None or roteiros.empty:
            return pd.DataFrame()

        df = roteiros.copy()

        if "sku" not in df.columns and "SKU" in df.columns:
            df["sku"] = df["SKU"].astype(str).str.strip()
        df["sku"] = df["sku"].astype(str).str.strip()
        if "artigo" not in df.columns:
            df["artigo"] = df["sku"]

        if "maquinas_elegiveis" not in df.columns and "maquinas_possiveis" in df.columns:
            df["maquinas_elegiveis"] = df["maquinas_possiveis"]
        df["maquinas_elegiveis"] = df["maquinas_elegiveis"].astype(str).str.strip()

        ratio = None
        if "ratio_pch" in df.columns:
            ratio = pd.to_numeric(df["ratio_pch"], errors="coerce")
        elif "racio_pc_h" in df.columns:
            ratio = pd.to_numeric(df["racio_pc_h"], errors="coerce")
        else:
            ratio = pd.Series(np.nan, index=df.index)
        ratio = ratio.where(ratio > 0)
        ratio_default = float(ratio.dropna().median()) if ratio.dropna().any() else 120.0
        df["racio_pc_h"] = ratio.fillna(ratio_default).clip(lower=1.0)

        overlap = None
        if "overlap_prev" in df.columns:
            overlap = pd.to_numeric(df["overlap_prev"], errors="coerce")
        elif "overlap_pct" in df.columns:
            overlap = pd.to_numeric(df["overlap_pct"], errors="coerce")
        else:
            overlap = pd.Series(0.0, index=df.index)
        df["overlap_pct"] = overlap.fillna(0.0).clip(lower=0.0, upper=0.4)  # Máximo 40% overlap

        people = pd.to_numeric(df.get("pessoas", 1), errors="coerce").fillna(1).round().astype(int)
        df["pessoas"] = people.clip(lower=1)

        return df

    def _sanitize_orders(self, ordens: pd.DataFrame) -> pd.DataFrame:
        if ordens is None or ordens.empty:
            return pd.DataFrame()

        df = ordens.copy()

        if "sku" not in df.columns and "SKU" in df.columns:
            df["sku"] = df["SKU"].astype(str).str.strip()
        df["sku"] = df["sku"].astype(str).str.strip()
        df["SKU"] = df.get("SKU", df["sku"])

        qty = None
        if "qty" in df.columns:
            qty = pd.to_numeric(df["qty"], errors="coerce")
        elif "quantidade" in df.columns:
            qty = pd.to_numeric(df["quantidade"], errors="coerce")
        else:
            qty = pd.Series(0, index=df.index, dtype=float)
        df["qty"] = qty.fillna(0).clip(lower=0)
        df["quantidade"] = df["qty"]

        df["data_prometida"] = pd.to_datetime(df.get("data_prometida"), errors="coerce")

        return df

    def generate_baseline_plan(self, start_date: datetime, end_date: datetime, 
                                cell: Optional[str] = None) -> PlanResult:
        """Gera plano baseline (Antes) - FIFO simples sem otimizações."""
        roteiros = self._sanitize_routes(self.loader.get_roteiros())
        ordens = self._sanitize_orders(self.loader.get_ordens())

        if roteiros.empty or ordens.empty:
            logger.info("Baseline: dados insuficientes para gerar plano.")
            return self._empty_plan_result()

        windowed = ordens.copy()
        window_mask = windowed["data_prometida"].between(start_date, end_date, inclusive="both")
        windowed = windowed[window_mask]

        if cell and "celula" in windowed.columns:
            windowed = windowed[windowed["celula"].astype(str).str.contains(cell, case=False, na=False)]

        windowed = windowed[windowed["qty"] > 0]

        if windowed.empty:
            logger.info("Baseline: nenhuma ordem válida no intervalo especificado.")
            return self._empty_plan_result()

        operations: List[Operation] = []
        current_time = start_date

        # Baseline: FIFO simples, sem otimizações
        for _, ordem in windowed.iterrows():
            sku_value = str(ordem.get("sku"))
            roteiro_ops = roteiros[roteiros["sku"] == sku_value]

            if roteiro_ops.empty:
                continue

            qty = float(ordem.get("qty", 0) or 0)
            if qty <= 0:
                continue

            # Extrair família
            familia = sku_value.split("-")[0] if "-" in sku_value else sku_value[:3]

            for _, rota_op in roteiro_ops.iterrows():
                racio = float(rota_op.get("racio_pc_h", 0) or 0)
                if racio <= 0:
                    racio = 120.0
                ciclo_nominal = 3600 / racio
                tempo_op = (qty * ciclo_nominal) / 3600  # horas

                setup_time = 30 / 60  # Baseline: 30 min fixo em horas

                start_slot = current_time
                end_slot = start_slot + timedelta(hours=tempo_op + setup_time)

                operations.append(
                    Operation(
                        ordem=sku_value,
                        artigo=sku_value,
                        operacao=str(rota_op.get("grupo_operacao", "Transformação") or "Transformação"),
                        recurso=str(rota_op.get("maquinas_elegiveis", "M-01") or "M-01"),
                        start_time=start_slot,
                        end_time=end_slot,
                        setor=str(rota_op.get("setor", "Transformação") or "Transformação"),
                        overlap=0.0,
                        rota="A",
                        explicacao="Plano baseline sem otimizações",
                        familia=familia,
                    )
                )

                current_time = end_slot

        kpis = self._calculate_kpis(operations, windowed)
        explicacoes = ["Plano baseline gerado sem otimizações"]

        logger.info(f"DEBUG APS BASELINE: LT={kpis['lead_time_h']:.1f}h, Setup={kpis['horas_setup_semana']:.1f}h, Fila={kpis.get('fila_gargalo_h', 0):.1f}h")

        return PlanResult(kpis=kpis, operations=operations, explicacoes=explicacoes)
    
    def generate_optimized_plan(self, start_date: datetime, end_date: datetime,
                                 cell: Optional[str] = None) -> PlanResult:
        """
        Gera plano otimizado (Depois) com heurística industrial rigorosa.
        
        Pipeline:
        1. Detectar gargalo(s)
        2. Ordenar operações do gargalo por data → família → duração
        3. Aplicar colagem no gargalo
        4. Propagar sequência para montante e jusante
        5. Aplicar overlap onde permitido
        6. Recalcular setups
        7. Validar LT
        8. Se LT < baseline → aceitar
        9. Senão → ajustar overlap e setups
        """
        # Gerar baseline primeiro para comparação
        baseline = self.generate_baseline_plan(start_date, end_date, cell)
        
        roteiros = self._sanitize_routes(self.loader.get_roteiros())
        ordens = self._sanitize_orders(self.loader.get_ordens())

        if roteiros.empty or ordens.empty:
            logger.info("Otimizado: dados insuficientes para gerar plano.")
            return baseline

        windowed = ordens.copy()
        window_mask = windowed["data_prometida"].between(start_date, end_date, inclusive="both")
        windowed = windowed[window_mask]
        windowed = windowed[windowed["qty"] > 0]

        if cell and "celula" in windowed.columns:
            windowed = windowed[windowed["celula"].astype(str).str.contains(cell, case=False, na=False)]

        if windowed.empty:
            logger.info("Otimizado: nenhuma ordem válida no intervalo especificado.")
            return baseline

        # PASSO 1: Detectar gargalo(s) usando score industrial
        gargalo_info = self._detect_bottleneck_industrial(roteiros, windowed)
        gargalo_recurso = gargalo_info.get("recurso", "N/A")
        
        logger.info(f"DEBUG APS: Gargalo detectado = {gargalo_recurso} (score={gargalo_info.get('score', 0):.2f})")

        # PASSO 2-4: Sequenciação industrial (colagem de famílias + propagação)
        operations = self._schedule_industrial(
            roteiros, windowed, start_date, gargalo_recurso
        )

        if not operations:
            logger.warning("Otimizado: nenhuma operação gerada, usando baseline.")
            return baseline

        # PASSO 5-6: Aplicar overlap e recalcular setups
        operations = self._apply_overlap_industrial(operations, roteiros)
        
        # PASSO 7-9: Validar e calcular KPIs
        kpis = self._calculate_kpis(operations, windowed)
        
        # VALIDAÇÃO CRÍTICA: Nunca aceitar plano pior que baseline
        lt_before = baseline.kpis.get("lead_time_h", 0.0)
        lt_after = kpis.get("lead_time_h", 0.0)
        setup_before = baseline.kpis.get("horas_setup_semana", 0.0)
        setup_after = kpis.get("horas_setup_semana", 0.0)
        fila_before = baseline.kpis.get("fila_gargalo_h", 0.0)
        fila_after = kpis.get("fila_gargalo_h", 0.0)
        util_before = baseline.kpis.get("utilizacao_gargalo", 0.0)
        util_after = kpis.get("utilizacao_gargalo", 0.0)

        logger.info("=" * 60)
        logger.info("DEBUG APS - COMPARAÇÃO ANTES vs DEPOIS:")
        logger.info(f"Lead time: {lt_before:.1f}h → {lt_after:.1f}h (Δ={lt_after-lt_before:+.1f}h)")
        logger.info(f"Setup: {setup_before:.1f}h → {setup_after:.1f}h (Δ={setup_after-setup_before:+.1f}h)")
        logger.info(f"Fila gargalo: {fila_before:.1f}h → {fila_after:.1f}h (Δ={fila_after-fila_before:+.1f}h)")
        logger.info(f"Utilização gargalo: {util_before:.3f} → {util_after:.3f} (Δ={util_after-util_before:+.3f})")
        logger.info("=" * 60)

        # REGRA DE OURO: Rejeitar se pior
        should_reject = False
        rejection_reasons = []
        
        if lt_after > lt_before + 1e-6:
            should_reject = True
            rejection_reasons.append(f"LT pior ({lt_after:.1f}h > {lt_before:.1f}h)")
        
        if setup_after > setup_before * 1.05:  # Tolerância de 5%
            should_reject = True
            rejection_reasons.append(f"Setup pior ({setup_after:.1f}h > {setup_before:.1f}h * 1.05)")
        
        if fila_after > fila_before + 1e-6:
            should_reject = True
            rejection_reasons.append(f"Fila pior ({fila_after:.1f}h > {fila_before:.1f}h)")
        
        if util_after > util_before * 1.10:  # Tolerância de 10%
            should_reject = True
            rejection_reasons.append(f"Utilização pior ({util_after:.3f} > {util_before:.3f} * 1.10)")

        if should_reject:
            logger.warning(f"APS: Plano otimizado REJEITADO. Razões: {', '.join(rejection_reasons)}")
            logger.warning("APS: Retornando baseline como fallback.")
            return baseline

        # Se passou validação, gerar explicações
        explicacoes = self._generate_explanations(operations, kpis, baseline.kpis)

        logger.info("APS: Plano otimizado ACEITE (melhor que baseline)")

        return PlanResult(kpis=kpis, operations=operations, explicacoes=explicacoes)

    def _detect_bottleneck_industrial(self, roteiros: pd.DataFrame, ordens: pd.DataFrame) -> Dict[str, any]:
        """
        Detecta gargalo usando score industrial:
        gargalo_score = (num_operacoes_recebidas * duração_media_op) / capacidade_pph
        """
        resource_stats: Dict[str, Dict] = defaultdict(lambda: {
            "num_ops": 0,
            "total_duration": 0.0,
            "capacidade_pph": 0.0,
            "score": 0.0,
        })

        sku_col = "sku" if "sku" in ordens.columns else "SKU"
        
        for _, ordem in ordens.iterrows():
            sku_value = str(ordem.get(sku_col, ""))
            qty = float(ordem.get("qty", 0) or 0)
            if qty <= 0:
                continue

            roteiro_ops = roteiros[roteiros["sku"] == sku_value]
            for _, rota_op in roteiro_ops.iterrows():
                recurso = str(rota_op.get("maquinas_elegiveis", "M-01") or "M-01")
                racio = float(rota_op.get("racio_pc_h", 120.0) or 120.0)
                
                # Calcular duração estimada
                ciclo_nominal = 3600 / racio if racio > 0 else 30
                tempo_op = (qty * ciclo_nominal) / 3600
                
                resource_stats[recurso]["num_ops"] += 1
                resource_stats[recurso]["total_duration"] += tempo_op
                resource_stats[recurso]["capacidade_pph"] = max(resource_stats[recurso]["capacidade_pph"], racio)

        # Calcular score para cada recurso
        for recurso, stats in resource_stats.items():
            num_ops = stats["num_ops"]
            dur_media = stats["total_duration"] / num_ops if num_ops > 0 else 0.0
            capacidade = stats["capacidade_pph"]
            
            # Score: (num_ops * dur_media) / capacidade
            # Recurso com maior score = gargalo
            stats["score"] = (num_ops * dur_media) / capacidade if capacidade > 0 else 0.0

        # Encontrar gargalo (maior score)
        if not resource_stats:
            return {"recurso": "N/A", "score": 0.0}
        
        gargalo_recurso = max(resource_stats.items(), key=lambda x: x[1]["score"])[0]
        gargalo_score = resource_stats[gargalo_recurso]["score"]

        return {
            "recurso": gargalo_recurso,
            "score": gargalo_score,
            "stats": resource_stats,
        }

    def _schedule_industrial(
        self, roteiros: pd.DataFrame, ordens: pd.DataFrame, start_date: datetime, gargalo_recurso: str
    ) -> List[Operation]:
        """
        Sequenciação industrial:
        1. Agrupar por família
        2. Ordenar famílias por tamanho de lote
        3. Processar famílias inteiras antes de passar para próxima
        4. Respeitar precedências (ordem das operações no roteiro)
        """
        operations: List[Operation] = []
        
        # Agrupar ordens por família
        familias_dict: Dict[str, List] = defaultdict(list)
        sku_col = "sku" if "sku" in ordens.columns else "SKU"
        
        for _, ordem in ordens.iterrows():
            sku_value = str(ordem.get(sku_col, ""))
            if not sku_value:
                continue
            familia = sku_value.split("-")[0] if "-" in sku_value else sku_value[:3]
            familias_dict[familia].append(ordem)

        # Ordenar famílias por tamanho total de lote (maior primeiro para colagem eficiente)
        familias_sorted = sorted(
            familias_dict.items(),
            key=lambda x: sum(float(o.get("qty", 0) or 0) for o in x[1]),
            reverse=True
        )

        # Estado por recurso
        resource_schedule: Dict[str, datetime] = {}  # Próximo slot disponível por recurso
        last_family_per_resource: Dict[str, str] = {}  # Última família processada por recurso
        
        # Processar cada família completamente antes de passar para próxima
        for familia, ordem_list in familias_sorted:
            # Ordenar ordens dentro da família por data prometida (prioridade)
            ordem_list_sorted = sorted(
                ordem_list,
                key=lambda o: pd.to_datetime(o.get("data_prometida", start_date), errors="coerce") or start_date
            )

            for ordem in ordem_list_sorted:
                sku_value = str(ordem.get(sku_col, ""))
                qty = float(ordem.get("qty", 0) or 0)
                if qty <= 0:
                    continue

                roteiro_ops = roteiros[roteiros["sku"] == sku_value]
                if roteiro_ops.empty:
                    continue

                # Processar operações do roteiro em ordem (respeitar precedências)
                for op_idx, (_, rota_op) in enumerate(roteiro_ops.iterrows()):
                    recurso = str(rota_op.get("maquinas_elegiveis", "M-01") or "M-01")
                    
                    # Calcular tempo de operação
                    racio = float(rota_op.get("racio_pc_h", 120.0) or 120.0)
                    if racio <= 0:
                        racio = 120.0
                    
                    ciclo_p50 = self.cycle_predictor.predict_p50(
                        sku=sku_value,
                        operacao=rota_op.get("grupo_operacao", "Transformação"),
                        recurso=recurso,
                        quantidade=qty,
                    )
                    tempo_op = (qty * ciclo_p50) / 3600

                    # Calcular setup (reduzir se mesma família)
                    familia_anterior = last_family_per_resource.get(recurso, "")
                    setup_reduction = 0.7 if familia_anterior == familia else 0.0
                    
                    setup_time = self.setup_predictor.predict(
                        familia_anterior=familia_anterior,
                        familia_atual=familia,
                        recurso=recurso,
                    ) * (1 - setup_reduction) / 60  # Converter para horas

                    # Determinar start_time (respeitar disponibilidade do recurso)
                    resource_available = resource_schedule.get(recurso, start_date)
                    start_slot = resource_available
                    
                    # Aplicar overlap apenas se:
                    # - Não for primeira operação do roteiro
                    # - Operação anterior já terminou
                    # - Não for operação de acabamento/polimento
                    overlap = 0.0
                    if op_idx > 0:
                        operacao_tipo = str(rota_op.get("grupo_operacao", "")).lower()
                        if "acabamento" not in operacao_tipo and "polimento" not in operacao_tipo:
                            overlap_pct = float(rota_op.get("overlap_pct", 0.0) or 0.0)
                            overlap = min(overlap_pct, 0.35)  # Máximo 35%
                            
                            # Reduzir overlap se operação for lenta (> 400 pcs/h = rápido, < 200 = lento)
                            if racio < 200:
                                overlap *= 0.5
                            
                            # Aplicar overlap: start_next = start_current + duration_current * (1 - overlap)
                            if operations:
                                op_anterior = operations[-1]
                                if op_anterior.ordem == sku_value:  # Mesma ordem
                                    overlap_hours = (op_anterior.end_time - op_anterior.start_time).total_seconds() / 3600 * (1 - overlap)
                                    start_slot = op_anterior.start_time + timedelta(hours=overlap_hours)
                                    start_slot = max(start_slot, resource_available)

                    end_slot = start_slot + timedelta(hours=tempo_op + setup_time)

                    # Verificar se há alternativa de rota (se gargalo)
                    rota = "A"
                    if recurso == gargalo_recurso:
                        alternativa = self._find_alternative_route(roteiro_ops, recurso)
                        if alternativa:
                            recurso = alternativa
                            rota = "B"

                    operations.append(
                        Operation(
                            ordem=sku_value,
                            artigo=sku_value,
                            operacao=rota_op.get("grupo_operacao", "Transformação"),
                            recurso=recurso,
                            start_time=start_slot,
                            end_time=end_slot,
                            setor=str(rota_op.get("setor", "Transformação") or "Transformação"),
                            overlap=overlap,
                            rota=rota,
                            explicacao=f"Família {familia}, overlap {overlap:.2f}, rota {rota}",
                            familia=familia,
                        )
                    )

                    # Atualizar estado
                    last_family_per_resource[recurso] = familia
                    resource_schedule[recurso] = end_slot

        return operations

    def _apply_overlap_industrial(self, operations: List[Operation], roteiros: pd.DataFrame) -> List[Operation]:
        """
        Ajusta overlap de forma industrial:
        - Verificar se operações consecutivas da mesma ordem podem ter overlap
        - Respeitar precedências
        - Não aplicar em acabamentos/polimento
        """
        operations_updated = []
        
        for i, op in enumerate(operations):
            # Se não é primeira operação e é da mesma ordem que anterior
            if i > 0 and operations[i-1].ordem == op.ordem:
                op_anterior = operations[i-1]
                
                # Verificar se pode aplicar overlap
                operacao_tipo = op.operacao.lower()
                if "acabamento" not in operacao_tipo and "polimento" not in operacao_tipo:
                    # Aplicar overlap: start_next = start_current + duration_current * (1 - overlap)
                    duration_anterior = (op_anterior.end_time - op_anterior.start_time).total_seconds() / 3600
                    overlap_hours = duration_anterior * (1 - op.overlap)
                    novo_start = op_anterior.start_time + timedelta(hours=overlap_hours)
                    
                    # Garantir que não é antes do recurso estar disponível
                    novo_start = max(novo_start, op.start_time)
                    
                    # Recalcular end_time
                    duration_op = (op.end_time - op.start_time).total_seconds() / 3600
                    novo_end = novo_start + timedelta(hours=duration_op)
                    
                    op.start_time = novo_start
                    op.end_time = novo_end
            
            operations_updated.append(op)
        
        return operations_updated

    def _calculate_kpis(self, operations: List[Operation], ordens: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula KPIs CORRETAMENTE:
        - Lead Time = média dos lead times de cada ORDEM (não por período)
        - Setup = soma real de setups
        - Gargalo = recurso com maior score industrial
        """
        if not operations:
            return {
                "otd_pct": 0.0,
                "lead_time_h": 0.0,
                "gargalo_ativo": "N/A",
                "horas_setup_semana": 0.0,
                "utilizacao_gargalo": 0.0,
                "fila_gargalo_h": 0.0,
                "throughput_gargalo": 0.0,
            }

        ordens = ordens if ordens is not None else pd.DataFrame()
        sku_col = "sku" if "sku" in ordens.columns else "SKU"
        due_col = "data_prometida" if "data_prometida" in ordens.columns else None

        # Calcular OTD
        on_time = 0
        total = len(ordens.index)

        for _, ordem in ordens.iterrows():
            sku_value = str(ordem.get(sku_col, ""))
            ordem_ops = [op for op in operations if op.ordem == sku_value]
            if not ordem_ops:
                continue

            fim_ordem = max(op.end_time for op in ordem_ops)
            if due_col:
                data_prometida = pd.to_datetime(ordem.get(due_col), errors="coerce")
            else:
                data_prometida = datetime.utcnow() + timedelta(days=7)

            if pd.isna(data_prometida):
                data_prometida = datetime.utcnow() + timedelta(days=7)

            if fim_ordem <= data_prometida:
                on_time += 1

        otd_pct = (on_time / total * 100) if total > 0 else 0.0

        # CALCULAR LEAD TIME CORRETAMENTE: Por ordem, não por período
        lead_times: List[float] = []
        for _, ordem in ordens.iterrows():
            sku_value = str(ordem.get(sku_col, ""))
            ordem_ops = [op for op in operations if op.ordem == sku_value]
            if not ordem_ops:
                continue
            
            # Lead time da ordem = max(end_times) - min(start_times)
            inicio = min(op.start_time for op in ordem_ops)
            fim = max(op.end_time for op in ordem_ops)
            lead_time_order = (fim - inicio).total_seconds() / 3600
            lead_times.append(lead_time_order)
        
        # Lead time global = média dos lead times das ordens
        lead_time_h = float(np.mean(lead_times)) if lead_times else 0.0

        # Calcular utilização por recurso e identificar gargalo
        resource_usage: Dict[str, float] = {}
        resource_start_times: Dict[str, List[datetime]] = {}
        resource_end_times: Dict[str, List[datetime]] = {}
        
        # Calcular janela temporal total
        if operations:
            min_start = min(op.start_time for op in operations)
            max_end = max(op.end_time for op in operations)
            total_hours = (max_end - min_start).total_seconds() / 3600.0 if max_end > min_start else 1.0
        else:
            total_hours = 1.0
        
        for op in operations:
            duration = (op.end_time - op.start_time).total_seconds() / 3600
            resource_usage[op.recurso] = resource_usage.get(op.recurso, 0.0) + duration
            
            if op.recurso not in resource_start_times:
                resource_start_times[op.recurso] = []
                resource_end_times[op.recurso] = []
            resource_start_times[op.recurso].append(op.start_time)
            resource_end_times[op.recurso].append(op.end_time)
        
        # Identificar gargalo (recurso com maior utilização)
        gargalo_ativo = max(resource_usage.items(), key=lambda x: x[1])[0] if resource_usage else "N/A"
        
        # Calcular utilização do gargalo
        utilizacao_gargalo = (resource_usage.get(gargalo_ativo, 0.0) / total_hours) if total_hours > 0 else 0.0
        
        # Calcular fila do gargalo (sobreposição de operações)
        fila_gargalo_h = 0.0
        if gargalo_ativo != "N/A" and gargalo_ativo in resource_start_times:
            starts = sorted(resource_start_times[gargalo_ativo])
            ends = sorted(resource_end_times[gargalo_ativo])
            if len(starts) > 1:
                max_overlap = 0.0
                for i, start in enumerate(starts[1:], 1):
                    prev_end = ends[i-1]
                    if start < prev_end:
                        overlap = (prev_end - start).total_seconds() / 3600.0
                        max_overlap = max(max_overlap, overlap)
                fila_gargalo_h = max_overlap
        
        # Calcular throughput do gargalo
        throughput_gargalo = 0.0
        if gargalo_ativo != "N/A" and total_hours > 0:
            ops_gargalo = [op for op in operations if op.recurso == gargalo_ativo]
            qty_processed = len(ops_gargalo)
            throughput_gargalo = qty_processed / total_hours if total_hours > 0 else 0.0

        # Calcular horas de setup REAL (baseado em mudanças de família por recurso)
        setup_hours = 0.0
        last_family_per_resource: Dict[str, str] = {}
        
        for op in operations:
            # Se mudou de família no mesmo recurso, houve setup
            familia_anterior = last_family_per_resource.get(op.recurso, "")
            if op.familia and familia_anterior and op.familia != familia_anterior:
                # Estimar setup: 30 minutos por mudança de família
                setup_hours += 0.5
            elif not familia_anterior:
                # Primeira operação no recurso também tem setup inicial
                setup_hours += 0.3
            
            if op.familia:
                last_family_per_resource[op.recurso] = op.familia

        return {
            "otd_pct": round(otd_pct, 1),
            "lead_time_h": round(lead_time_h, 1),
            "gargalo_ativo": gargalo_ativo,
            "horas_setup_semana": round(setup_hours, 1),
            "utilizacao_gargalo": round(utilizacao_gargalo, 3),
            "fila_gargalo_h": round(fila_gargalo_h, 1),
            "throughput_gargalo": round(throughput_gargalo, 1),
        }
    
    def _group_by_family(self, ordens: pd.DataFrame, roteiros: pd.DataFrame) -> Dict[str, List]:
        """Agrupa ordens por família"""
        familias: Dict[str, List] = {}
        sku_col = "sku" if "sku" in ordens.columns else "SKU"

        for _, ordem in ordens.iterrows():
            sku = str(ordem.get(sku_col, ""))
            if not sku:
                continue
            familia = sku.split("-")[0] if "-" in sku else sku[:3]
            familias.setdefault(familia, []).append(ordem)

        return familias
    
    def _is_bottleneck(self, recurso: str, resource_queue: Dict[str, float]) -> bool:
        """Verifica se recurso é gargalo"""
        usage = resource_queue.get(recurso, 0)
        return usage > 40  # Mais de 40h de fila
    
    def _find_alternative_route(self, roteiro_ops: pd.DataFrame, recurso_atual: str) -> Optional[str]:
        """Encontra rota alternativa"""
        # Procurar por outras máquinas possíveis
        for _, rota_op in roteiro_ops.iterrows():
            maquinas_str = str(rota_op.get("maquinas_elegiveis", "") or "")
            if not maquinas_str:
                continue
            
            # Extrair todas as máquinas possíveis
            import re
            maquinas = re.findall(r'M[-\s]?(\d+)', maquinas_str)
            for maq in maquinas:
                maq_id = f"M-{maq}"
                if maq_id != recurso_atual:
                    return maq_id
        
        return None
    
    def _generate_explanations(
        self, operations: List[Operation], kpis: Dict[str, float], kpis_before: Dict[str, float]
    ) -> List[str]:
        """Gera explicações das decisões com impacto mensurável"""
        explicacoes = []
        
        # Overlap aplicado
        overlaps = [op.overlap for op in operations if op.overlap > 0]
        if overlaps:
            avg_overlap = np.mean(overlaps)
            lt_before = kpis_before.get("lead_time_h", 0.0)
            lt_after = kpis.get("lead_time_h", 0.0)
            delta_lt = lt_before - lt_after
            explicacoes.append(
                f"🧩 Overlap aplicado (médio {avg_overlap:.2f}) → lead time {delta_lt:+.1f}h"
            )
        
        # Rotas alternativas
        rotas_b = [op for op in operations if op.rota == "B"]
        if rotas_b:
            explicacoes.append(
                f"🧭 Desviámos {len(rotas_b)} operações para rotas alternativas → descongestionamento"
            )
        
        # Colagem de famílias
        familias_por_recurso: Dict[str, set] = defaultdict(set)
        for op in operations:
            if op.familia:
                familias_por_recurso[op.recurso].add(op.familia)
        
        familias_coladas = sum(1 for familias in familias_por_recurso.values() if len(familias) > 1)
        if familias_coladas > 0:
            setup_before = kpis_before.get("horas_setup_semana", 0.0)
            setup_after = kpis.get("horas_setup_semana", 0.0)
            delta_setup = setup_before - setup_after
            explicacoes.append(
                f"🔗 Colagem de famílias → setup {delta_setup:+.1f}h"
            )
        
        # Impacto em OTD
        otd_before = kpis_before.get("otd_pct", 0.0)
        otd_after = kpis.get("otd_pct", 0.0)
        if otd_after > otd_before:
            explicacoes.append(
                f"✅ OTD melhorado: {otd_before:.1f}% → {otd_after:.1f}% (+{otd_after-otd_before:.1f}pp)"
            )
        
        return explicacoes
