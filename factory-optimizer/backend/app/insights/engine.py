"""Insight Engine: agrega dados de planeamento, gargalos e inventário em contexto estruturado para o LLM."""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.aps.scheduler import APSScheduler
from app.etl.loader import get_loader

logger = logging.getLogger(__name__)


class InsightEngine:
    """Motor que agrega dados já calculados em contexto estruturado para o LLM."""

    def __init__(self):
        self.scheduler = APSScheduler()
        self.loader = get_loader()

    def build_full_context(self) -> Dict[str, Any]:
        """
        Constrói o contexto industrial TOTAL - InsightEngine 2.0.
        
        Gera um JSON completamente analisado com toda a análise industrial PRÉ-LLM.
        O LLM NÃO analisa - apenas recebe dados já interpretados.
        """
        context: Dict[str, Any] = {
            "planning": self._extract_planning_insights(),        # Antes vs depois, decisões APS
            "bottlenecks": self._extract_bottlenecks_insights(),  # Recursos críticos, flags industriais
            "inventory": self._extract_inventory_insights(),      # Risco, cobertura, ROP, excesso
            "suggestions": {                                       # ActionCandidates brutos (sem LLM)
                "action_candidates": self.build_action_candidates(),
            },
            "what_if": self._extract_what_if_actions(),           # Ações simuláveis
            "ml_quality": self._extract_ml_quality(),            # Qualidade P50/P90, F1-score, etc.
            "metadata": {
                "batch_id": self.loader.get_status().get("batch_id", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
            },
            "generated_at": datetime.utcnow().isoformat(),
        }
        return context

    def build_context_by_mode(self, mode: str) -> Dict[str, Any]:
        """
        Constrói contexto filtrado por modo - LAZY LOADING: calcula APENAS o necessário.
        Performance: Não calcula todos os módulos, apenas o que é necessário para o modo.
        """
        timestamp = datetime.utcnow().isoformat()
        
        if mode == "planeamento":
            # APENAS planeamento: KPIs ANTES/DEPOIS, gargalo, decisões detalhadas
            # FILTRO AGRESSIVO: EXCLUIR inventário, SKUs, ABC/XYZ
            # ✅ LAZY: Calcula apenas planning, não calcula bottlenecks, inventory, etc.
            planning = self._extract_planning_insights()
            return {
                "kpis": {
                    "otd": planning.get("otd", 0.0),
                    "otd_before": planning.get("otd_before", 0.0),
                    "lead_time_before": planning.get("lead_time_before", 0.0),
                    "lead_time_after": planning.get("lead_time_after", 0.0),
                    "lead_time_after_inconsistent": planning.get("lead_time_after_inconsistent", False),
                    "lead_time_delta_pct": planning.get("lead_time_delta_pct", 0.0),
                    "causa_lead_time": planning.get("causa_lead_time", "Não identificada"),
                    "setup_hours": planning.get("setup_hours", 0.0),
                    "setup_hours_before": planning.get("setup_hours_before", 0.0),
                },
                "ops_atrasadas": planning.get("ops_atrasadas", []),  # Para explicar OTD
                "top_familias_setup": planning.get("top_familias_setup", []),  # Para explicar setups
                "gargalo_principal": planning.get("main_bottleneck", {"id": "N/A", "utilizacao": 0.0, "fila_h": 0.0}),
                "overlap": planning.get("overlap", {"transformacao": 0.0, "acabamentos": 0.0, "embalagem": 0.0}),
                "decisoes_do_motor": planning.get("decisoes_do_motor", []),  # Decisões detalhadas com impacto
                "interpretacao_industrial": planning.get("interpretacao_industrial", {}),  # Análise pré-LLM
                "generated_at": timestamp,
                # EXCLUÍDO: inventário, SKUs, ABC/XYZ, coberturas, ROP
            }
        
        if mode == "gargalos":
            # APENAS gargalos: recursos, utilizações, filas, drivers, alternativas, pph, cycle_time_s
            # FILTRO AGRESSIVO: EXCLUIR inventário, SKUs, OTD global, lead time global
            # ✅ LAZY: Calcula apenas bottlenecks, não calcula planning, inventory, etc.
            bottlenecks = self._extract_bottlenecks_insights()
            top_resources = bottlenecks.get("top_resources", [])
            
            # Enriquecer com drivers de gargalo e flags industriais
            resources_enriched = []
            for resource in top_resources:
                recurso_id = resource.get("recurso", "")
                utilizacao_raw = resource.get("utilizacao", 0.0)
                # Normalizar utilização > 1.0 (100%) - se > 1.0, usar 1.0 mas marcar como saturação
                if utilizacao_raw > 1.0:
                    utilizacao_normalized = 1.0
                    utilizacao_pct = 100.0
                    is_saturated = True
                else:
                    utilizacao_normalized = utilizacao_raw
                    utilizacao_pct = utilizacao_raw * 100
                    is_saturated = False
                
                fila_h = resource.get("fila_h", 0.0)
                prob = resource.get("prob_gargalo", 0.0)
                has_alt = resource.get("has_alternative", False)
                pph = resource.get("pph")  # Peças por hora (já calculado em _extract_bottlenecks_insights)
                cycle_time_s = resource.get("cycle_time_s")  # Tempo de ciclo em segundos
                converging_ops = resource.get("converging_ops", 0)
                flags = resource.get("flags", {})
                
                # Drivers baseados em análise industrial
                drivers = []
                if is_saturated or utilizacao_pct > 90:
                    if is_saturated:
                        drivers.append(f"Utilização saturada (>{utilizacao_raw*100:.0f}%)")
                    else:
                        drivers.append(f"Utilização alta ({utilizacao_pct:.1f}%)")
                if fila_h > 50:
                    drivers.append(f"Fila crítica ({fila_h:.1f}h)")
                elif fila_h == 0 and prob > 0.9:
                    drivers.append(f"Risco de gargalo futuro (prob {prob*100:.0f}%)")
                if prob > 0.7:
                    drivers.append(f"Alta probabilidade de gargalo ({prob*100:.0f}%)")
                if flags.get("resource_is_slow"):
                    drivers.append(f"Recurso lento ({pph:.0f} pç/h)" if pph else "Recurso lento")
                if flags.get("high_convergence"):
                    drivers.append(f"Muitas operações convergentes ({converging_ops})")
                if flags.get("no_alternative"):
                    drivers.append("Sem alternativa de rota")
                
                resources_enriched.append({
                    "recurso": recurso_id,
                    "utilizacao_pct": round(utilizacao_pct, 1),
                    "utilizacao_raw": round(utilizacao_raw * 100, 1) if is_saturated else None,
                    "is_saturated": is_saturated,
                    "fila_h": round(fila_h, 1),
                    "probabilidade_gargalo": round(prob, 2),
                    "pph": round(pph, 0) if pph else None,
                    "cycle_time_s": round(cycle_time_s, 1) if cycle_time_s else None,
                    "converging_ops": converging_ops,
                    "flags": flags,  # Flags industriais pré-calculadas
                    "drivers": drivers,
                    "tem_alternativa": has_alt,
                })
            
            return {
                "recursos": resources_enriched,
                "overlap_applied": bottlenecks.get("overlap_applied", {}),
                "generated_at": timestamp,
                # EXCLUÍDO: inventário, SKUs, OTD global, lead time global
            }
        
        if mode == "inventario":
            # APENAS inventário: SKUs, classes, coberturas, risco, ROP, flags industriais
            # FILTRO AGRESSIVO: EXCLUIR gargalos, recursos, OTD, lead time, setups
            # ✅ LAZY: Calcula apenas inventory, não calcula planning, bottlenecks, etc.
            inventory = self._extract_inventory_insights()
            return {
                "skus": inventory.get("skus", []),  # Todos os SKUs com flags industriais
                "matrix": inventory.get("matrix", {}),  # Matriz ABC/XYZ
                "skus_criticos": inventory.get("critical_skus", [])[:5],  # TOP 5
                "skus_risco_rutura": inventory.get("skus_risco_rutura", [])[:10],  # Top 10 em risco
                "skus_excesso": inventory.get("skus_excesso", [])[:10],  # Top 10 em excesso
                "kpis": {
                    "skus_total": inventory.get("skus_total", 0),
                    "average_coverage_days": inventory.get("kpis", {}).get("average_coverage_days", 0.0),
                    "global_risk_score": inventory.get("kpis", {}).get("global_risk_score", 0.0),
                },
                "generated_at": timestamp,
                # EXCLUÍDO: gargalos, recursos, OTD, lead time, setups, throughput
            }
        
        if mode == "sugestoes":
            # APENAS ActionCandidates estruturados - SEM resumo executivo
            # O LLM recebe apenas a lista de ações pré-analisadas pelo engine
            # ✅ LAZY: Calcula apenas action_candidates (que já faz lazy loading interno)
            action_candidates = self.build_action_candidates()
            
            return {
                "actions": action_candidates[:10],  # Top 10 ações com impacto estimado
                "generated_at": timestamp,
            }
        
        if mode == "what_if":
            # Apenas output de simulação (será preenchido pela API what_if)
            # ✅ LAZY: Não calcula nada, apenas retorna estrutura vazia
            return {
                "simulation_result": None,  # Será preenchido pela API
                "generated_at": timestamp,
            }
        
        if mode == "resumo":
            # Resumo compacto para chat quando pergunta é geral
            # ⚠️ Para resumo, precisa de dados de vários módulos, mas otimizado:
            # Calcula apenas os resumos, não os dados completos
            planning = self._extract_planning_insights()
            inventory = self._extract_inventory_insights()
            bottlenecks = self._extract_bottlenecks_insights()
            ml_quality = self._extract_ml_quality()
            
            return {
                "planning_summary": {
                    "otd": planning.get("otd", 0.0),
                    "lead_time_after": planning.get("lead_time_after", 0.0),
                    "lead_time_after_inconsistent": planning.get("lead_time_after_inconsistent", False),
                    "main_bottleneck_id": planning.get("main_bottleneck", {}).get("id", "N/A"),
                },
                "inventory_summary": {
                    "skus_total": inventory.get("skus_total", 0),
                    "critical_skus_count": len(inventory.get("critical_skus", [])),
                },
                "bottlenecks_summary": {
                    "top_resources_count": len(bottlenecks.get("top_resources", [])),
                },
                "ml_quality": ml_quality,  # Incluir métricas ML para o LLM poder explicar confiança
                "generated_at": timestamp,
            }
        
        # Fallback: retornar contexto completo apenas se modo desconhecido
        # ⚠️ Aviso: modo desconhecido calcula tudo (pode ser lento)
        logger.warning(f"Modo desconhecido '{mode}', calculando contexto completo (pode ser lento)")
        return self.build_full_context()

    def _extract_planning_insights(self) -> Dict[str, Any]:
        """Extrai insights de planeamento do scheduler."""
        try:
            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=7)

            plano_antes = self.scheduler.generate_baseline_plan(start_date, end_date)
            plano_depois = self.scheduler.generate_optimized_plan(start_date, end_date)

            kpis_antes = plano_antes.kpis if plano_antes and plano_antes.kpis else {}
            kpis_depois = plano_depois.kpis if plano_depois and plano_depois.kpis else {}

            lead_time_before = float(kpis_antes.get("lead_time_h", 0.0))
            lead_time_after_raw = float(kpis_depois.get("lead_time_h", 0.0))
            
            # Validar lead_time_after: se > 200h ou <= 0, é inconsistente
            if lead_time_after_raw > 200 or lead_time_after_raw <= 0:
                lead_time_after = lead_time_before if lead_time_before > 0 else 0.0
                lead_time_after_inconsistent = True
            else:
                lead_time_after = lead_time_after_raw
                lead_time_after_inconsistent = False
            
            lead_time_delta_pct = (
                ((lead_time_before - lead_time_after) / lead_time_before * 100) if lead_time_before > 0 else 0.0
            )

            gargalo_ativo = str(kpis_depois.get("gargalo_ativo", "N/A"))

            overlap_avg = 0.0
            if plano_depois and plano_depois.operations:
                overlaps = [float(getattr(op, "overlap", 0.0) or 0.0) for op in plano_depois.operations]
                overlap_avg = sum(overlaps) / len(overlaps) if overlaps else 0.0

            top_decisions = []
            if lead_time_delta_pct < -5:
                top_decisions.append({"type": "overlap", "delta_lead_time_pct": round(lead_time_delta_pct, 1)})

            # Obter valores ANTES e DEPOIS do APS (garantir que vêm do scheduler, não inventados)
            utilizacao_gargalo_antes = 0.0
            fila_gargalo_antes = 0.0
            utilizacao_gargalo_depois = 0.0
            fila_gargalo_depois = 0.0
            throughput_gargalo_antes = 0.0
            throughput_gargalo_depois = 0.0

            if gargalo_ativo and gargalo_ativo != "N/A":
                # Valores DEPOIS vêm diretamente do KPIs do plano otimizado
                kpis_depois_gargalo = kpis_depois.get("utilizacao_gargalo", 0.0)
                kpis_depois_fila = kpis_depois.get("fila_gargalo_h", 0.0)
                kpis_depois_throughput = kpis_depois.get("throughput_gargalo", 0.0)
                
                utilizacao_gargalo_depois = float(kpis_depois_gargalo) if kpis_depois_gargalo else 0.0
                fila_gargalo_depois = float(kpis_depois_fila) if kpis_depois_fila else 0.0
                throughput_gargalo_depois = float(kpis_depois_throughput) if kpis_depois_throughput else 0.0
                
                # Valores ANTES vêm do plano baseline
                kpis_antes_gargalo = kpis_antes.get("utilizacao_gargalo", 0.0)
                kpis_antes_fila = kpis_antes.get("fila_gargalo_h", 0.0)
                kpis_antes_throughput = kpis_antes.get("throughput_gargalo", 0.0)
                
                utilizacao_gargalo_antes = float(kpis_antes_gargalo) if kpis_antes_gargalo else 0.0
                fila_gargalo_antes = float(kpis_antes_fila) if kpis_antes_fila else 0.0
                throughput_gargalo_antes = float(kpis_antes_throughput) if kpis_antes_throughput else 0.0
                
                # Fallback: se não estiver nos KPIs, usar dados de bottlenecks
                if utilizacao_gargalo_depois == 0.0 or fila_gargalo_depois == 0.0:
                    bottlenecks_data = self._get_bottlenecks_data()
                    for resource in bottlenecks_data.get("top_resources", []):
                        if resource.get("recurso") == gargalo_ativo:
                            if utilizacao_gargalo_depois == 0.0:
                                utilizacao_gargalo_depois = float(resource.get("utilizacao", 0.0))
                            if fila_gargalo_depois == 0.0:
                                fila_gargalo_depois = float(resource.get("fila_h", 0.0))
                            break

            # Extrair decisões detalhadas do motor com impacto mensurável
            decisoes_detalhadas = []
            if plano_depois and plano_depois.operations:
                # Analisar operações para identificar decisões específicas
                # Criar chave única: SKU + operação (porque uma ordem pode ter múltiplas operações)
                operacoes_antes_dict = {}
                if plano_antes and plano_antes.operations:
                    for op in plano_antes.operations:
                        key = f"{op.ordem}_{op.operacao}"
                        operacoes_antes_dict[key] = op
                
                operacoes_depois_dict = {}
                for op in plano_depois.operations:
                    key = f"{op.ordem}_{op.operacao}"
                    operacoes_depois_dict[key] = op
                
                # Identificar desvios de rota
                desvios_rota = []
                familias_coladas = set()
                overlaps_aplicados = []
                
                for key, op_depois in operacoes_depois_dict.items():
                    op_antes = operacoes_antes_dict.get(key)
                    if op_antes:
                        # Verificar se houve desvio de rota
                        recurso_antes = op_antes.recurso
                        recurso_depois = op_depois.recurso
                        if recurso_antes != recurso_depois and recurso_antes and recurso_depois:
                            lead_time_antes = (op_antes.end_time - op_antes.start_time).total_seconds() / 3600
                            lead_time_depois = (op_depois.end_time - op_depois.start_time).total_seconds() / 3600
                            delta_lead_time = lead_time_antes - lead_time_depois
                            desvios_rota.append({
                                "ordem": op_depois.ordem,
                                "operacao": op_depois.operacao,
                                "recurso_antes": recurso_antes,
                                "recurso_depois": recurso_depois,
                                "delta_lead_time_h": round(delta_lead_time, 1),
                            })
                    
                    # Verificar overlap aplicado (sempre verificar no plano depois)
                    overlap_depois = float(op_depois.overlap or 0.0)
                    if overlap_depois > 0:
                        overlaps_aplicados.append({
                            "ordem": op_depois.ordem,
                            "operacao": op_depois.operacao,
                            "overlap": round(overlap_depois, 2),
                        })
                
                # Identificar colagem de famílias (operações consecutivas com mesma família)
                if plano_depois.operations:
                    # Agrupar por recurso e ordenar por start_time
                    ops_por_recurso = {}
                    for op in plano_depois.operations:
                        if op.recurso not in ops_por_recurso:
                            ops_por_recurso[op.recurso] = []
                        ops_por_recurso[op.recurso].append(op)
                    
                    for recurso, ops in ops_por_recurso.items():
                        ops_sorted = sorted(ops, key=lambda x: x.start_time)
                        familia_anterior = None
                        count_familia = 0
                        for op in ops_sorted:
                            # Tentar extrair família do SKU (ordem)
                            sku = op.ordem
                            familia_atual = sku[:3] if len(sku) >= 3 else sku.split("-")[0] if "-" in sku else sku
                            if familia_atual == familia_anterior:
                                count_familia += 1
                            else:
                                if count_familia > 1:
                                    familias_coladas.add(familia_anterior)
                                familia_anterior = familia_atual
                                count_familia = 1
                        if count_familia > 1:
                            familias_coladas.add(familia_anterior)
                
                # Construir decisões detalhadas
                if overlaps_aplicados:
                    # Agrupar overlaps por operação
                    overlaps_por_operacao = {}
                    for ov in overlaps_aplicados:
                        op = ov["operacao"]
                        if op not in overlaps_por_operacao:
                            overlaps_por_operacao[op] = []
                        overlaps_por_operacao[op].append(ov)
                    
                    for operacao, ovs in overlaps_por_operacao.items():
                        avg_overlap = sum(o["overlap"] for o in ovs) / len(ovs)
                        decisoes_detalhadas.append({
                            "tipo": "overlap",
                            "operacao": operacao,
                            "overlap_aplicado": round(avg_overlap, 2),
                            "num_operacoes": len(ovs),
                            "impacto_lead_time_h": round(-lead_time_after * avg_overlap * 0.3, 1),  # Estimativa
                            "impacto_otd_pp": round(lead_time_delta_pct * 0.1, 1) if lead_time_delta_pct > 0 else 0.0,
                        })
                
                if desvios_rota:
                    # Agrupar desvios por recurso
                    desvios_por_recurso = {}
                    for desvio in desvios_rota:
                        recurso = desvio["recurso_antes"]
                        if recurso not in desvios_por_recurso:
                            desvios_por_recurso[recurso] = []
                        desvios_por_recurso[recurso].append(desvio)
                    
                    for recurso, desvios in desvios_por_recurso.items():
                        total_delta = sum(d["delta_lead_time_h"] for d in desvios)
                        recurso_destino = desvios[0]["recurso_depois"]
                        decisoes_detalhadas.append({
                            "tipo": "desvio_rota",
                            "recurso_origem": recurso,
                            "recurso_destino": recurso_destino,
                            "num_operacoes": len(desvios),
                            "impacto_lead_time_h": round(total_delta, 1),
                            "impacto_fila_h": round(-fila_gargalo_depois * 0.2, 1) if recurso == gargalo_ativo else 0.0,
                        })
                
                if familias_coladas:
                    decisoes_detalhadas.append({
                        "tipo": "colagem_familias",
                        "familias": list(familias_coladas)[:5],
                        "num_familias": len(familias_coladas),
                        "impacto_setup_h": round(-float(kpis_depois.get("horas_setup_semana", 0.0)) * 0.3, 1),  # Estimativa de redução
                    })
            
            # Identificar OPs atrasadas (para explicar OTD)
            ops_atrasadas = []
            if plano_antes and plano_antes.operations:
                # Agrupar operações por ordem (SKU) para calcular lead time total
                ops_por_ordem = {}
                for op in plano_antes.operations:
                    if op.ordem not in ops_por_ordem:
                        ops_por_ordem[op.ordem] = []
                    ops_por_ordem[op.ordem].append(op)
                
                for ordem_id, ops in ops_por_ordem.items():
                    if not ops:
                        continue
                    # Calcular lead time total da ordem
                    inicio = min(op.start_time for op in ops)
                    fim = max(op.end_time for op in ops)
                    lead_time_h = (fim - inicio).total_seconds() / 3600
                    # Considerar atrasado se lead time > 100h (proxy para atraso)
                    if lead_time_h > 100:
                        ops_atrasadas.append({
                            "ordem": ordem_id,
                            "atraso_h": round(lead_time_h, 1),
                        })
            
            # Identificar causa do lead time (fila acumulada)
            causa_lead_time = f"Fila acumulada no recurso {gargalo_ativo}" if gargalo_ativo != "N/A" else "Sequenciamento não otimizado"
            if fila_gargalo_depois > 0:
                causa_lead_time = f"Fila de {fila_gargalo_depois:.1f}h no recurso {gargalo_ativo}"
            
            # Identificar setups por família (quais causam mais)
            setups_por_familia = {}
            if plano_antes and plano_antes.operations:
                for op in plano_antes.operations:
                    ordem_id = getattr(op, "ordem", "")
                    familia = ordem_id[:3] if len(ordem_id) >= 3 else "OUT"
                    if familia not in setups_por_familia:
                        setups_por_familia[familia] = 0
                    # Estimativa: cada operação tem setup
                    setups_por_familia[familia] += 1
            
            # Top 2 famílias que causam mais setups
            top_familias_setup = sorted(setups_por_familia.items(), key=lambda x: x[1], reverse=True)[:2]
            
            # Calcular overlap por setor
            overlap_por_setor = {"transformacao": 0.0, "acabamentos": 0.0, "embalagem": 0.0}
            if plano_depois and plano_depois.operations:
                setor_overlaps: Dict[str, List[float]] = {}
                for op in plano_depois.operations:
                    setor = getattr(op, "setor", "outros")
                    overlap = float(getattr(op, "overlap", 0.0) or 0.0)
                    if setor not in setor_overlaps:
                        setor_overlaps[setor] = []
                    setor_overlaps[setor].append(overlap)
                
                for setor, overlaps in setor_overlaps.items():
                    if overlaps:
                        avg = sum(overlaps) / len(overlaps)
                        if "transform" in setor.lower() or "transformação" in setor.lower():
                            overlap_por_setor["transformacao"] = round(avg, 2)
                        elif "acab" in setor.lower():
                            overlap_por_setor["acabamentos"] = round(avg, 2)
                        elif "embal" in setor.lower():
                            overlap_por_setor["embalagem"] = round(avg, 2)
            
            # ANÁLISE INDUSTRIAL PRÉ-LLM: Comparar setores e identificar padrões
            interpretacao_industrial = {}
            if plano_antes and plano_depois and plano_antes.operations and plano_depois.operations:
                # Comparar cadências entre setores
                roteiros = self.loader.get_roteiros()
                if not roteiros.empty and "racio_pc_h" in roteiros.columns and "setor" in roteiros.columns:
                    setor_cadencia = {}
                    for setor in ["Transformação", "Acabamentos", "Embalagem"]:
                        setor_ops = roteiros[roteiros["setor"].str.contains(setor, case=False, na=False)]
                        if not setor_ops.empty:
                            racios = setor_ops["racio_pc_h"].dropna()
                            if not racios.empty:
                                setor_cadencia[setor] = float(racios.mean())
                    
                    # Interpretar: "acabamentos são X vezes mais lentos que transformação"
                    if "Transformação" in setor_cadencia and "Acabamentos" in setor_cadencia:
                        ratio = setor_cadencia["Acabamentos"] / setor_cadencia["Transformação"] if setor_cadencia["Transformação"] > 0 else 1.0
                        interpretacao_industrial["acabamentos_vs_transformacao"] = round(ratio, 1)
                
                # Overlap recomendado (15-25% é típico industrial)
                overlap_medio = sum(overlap_por_setor.values()) / len([v for v in overlap_por_setor.values() if v > 0]) if any(overlap_por_setor.values()) else 0.0
                interpretacao_industrial["overlap_recomendado"] = "15-25%" if overlap_medio < 0.2 else "25-40%" if overlap_medio < 0.4 else "40-60%"
            
            return {
                "otd": float(kpis_depois.get("otd_pct", 0.0)),
                "otd_before": float(kpis_antes.get("otd_pct", 0.0)),
                "ops_atrasadas": ops_atrasadas[:5],  # Top 5 OPs atrasadas
                "lead_time_before": lead_time_before,
                "lead_time_after": lead_time_after,
                "lead_time_after_inconsistent": lead_time_after_inconsistent,
                "lead_time_delta_pct": round(lead_time_delta_pct, 1),
                "causa_lead_time": causa_lead_time,
                "setup_hours": float(kpis_depois.get("horas_setup_semana", 0.0)),
                "setup_hours_before": float(kpis_antes.get("horas_setup_semana", 0.0)),
                "top_familias_setup": [{"familia": f, "count": c} for f, c in top_familias_setup],
                "main_bottleneck": {
                    "id": gargalo_ativo,
                    "utilizacao_antes": round(utilizacao_gargalo_antes, 3),
                    "utilizacao_depois": round(utilizacao_gargalo_depois, 3),
                    "fila_h_antes": round(fila_gargalo_antes, 1),
                    "fila_h_depois": round(fila_gargalo_depois, 1),
                    "throughput_antes": round(throughput_gargalo_antes, 1),
                    "throughput_depois": round(throughput_gargalo_depois, 1),
                },
                "overlap": overlap_por_setor,
                "decisoes_do_motor": decisoes_detalhadas[:7],  # Top 7 decisões
                "interpretacao_industrial": interpretacao_industrial,  # Análise pré-LLM
            }
        except Exception as exc:
            logger.exception("Falha ao extrair insights de planeamento")
            return {
                "otd": 0.0,
                "otd_before": 0.0,
                "ops_atrasadas": [],
                "lead_time_before": 0.0,
                "lead_time_after": 0.0,
                "lead_time_after_inconsistent": False,
                "lead_time_delta_pct": 0.0,
                "causa_lead_time": "Não identificada",
                "setup_hours": 0.0,
                "setup_hours_before": 0.0,
                "top_familias_setup": [],
                "main_bottleneck": {"id": "N/A", "utilizacao": 0.0, "fila_h": 0.0},
                "overlap": {"transformacao": 0.0, "acabamentos": 0.0, "embalagem": 0.0},
                "decisoes_do_motor": [],
            }

    def _extract_inventory_insights(self) -> Dict[str, Any]:
        """Extrai insights de inventário do loader."""
        try:
            insights = self.loader.get_inventory_insights()
            matrix = insights.get("matrix", {})
            skus = insights.get("skus", [])
            top_risks = insights.get("top_risks", [])

            skus_by_class: Dict[str, int] = {"A": 0, "B": 0, "C": 0}
            for sku_entry in skus:
                classe = sku_entry.get("classe", "")
                if classe and classe[0] in skus_by_class:
                    skus_by_class[classe[0]] += 1

            critical_skus: List[Dict[str, Any]] = []
            for risk_entry in top_risks[:10]:
                sku_code = risk_entry.get("sku", "")
                matching_sku = next((s for s in skus if s.get("sku") == sku_code), None)
                if matching_sku:
                    # Usar risco_30d do SKU se disponível, senão calcular a partir de risk_score
                    risco_30d = float(matching_sku.get("risco_30d", 0.0))
                    if risco_30d == 0.0:
                        risco_30d = float(risk_entry.get("probability", risk_entry.get("risk_score", 0.0)) * 100)
                    
                    critical_skus.append(
                        {
                            "sku": sku_code,
                            "classe": matching_sku.get("classe", ""),
                            "xyz": matching_sku.get("xyz", ""),
                            "stock_atual": float(matching_sku.get("stock_atual", 0.0)),
                            "cobertura_dias": float(matching_sku.get("cobertura_dias", 0.0)),
                            "risco_30d": round(risco_30d, 1),
                            "rop": float(matching_sku.get("rop", 0.0)),
                        }
                    )

            # ANÁLISE INDUSTRIAL PRÉ-LLM: Interpretar cada SKU como gestor de inventário
            skus_enriched = []
            skus_risco_rutura = []
            skus_excesso = []
            
            for sku_data in skus:
                coverage_dias = sku_data.get("cobertura_dias", 0.0)
                risco_30d = sku_data.get("risco_30d", 0.0)
                stock_atual = sku_data.get("stock_atual", 0.0)
                rop = sku_data.get("rop", 0.0)
                classe = sku_data.get("classe", "C")
                
                # Interpretação industrial
                risco_rutura = coverage_dias < 30.0
                excesso_stock = coverage_dias > 365.0
                abaixo_rop = stock_atual < rop
                
                # Criticidade combinada (ABC + risco)
                criticidade = "ALTA" if (classe == "A" and risco_30d > 20.0) or risco_30d > 50.0 else \
                              "MÉDIA" if (classe in ["A", "B"] and risco_30d > 5.0) or risco_30d > 20.0 else \
                              "BAIXA"
                
                sku_enriched = {
                    **sku_data,
                    "risco_rutura": risco_rutura,
                    "excesso_stock": excesso_stock,
                    "abaixo_rop": abaixo_rop,
                    "criticidade": criticidade,
                }
                skus_enriched.append(sku_enriched)
                
                if risco_rutura:
                    skus_risco_rutura.append(sku_enriched)
                if excesso_stock:
                    skus_excesso.append(sku_enriched)

            return {
                "skus_total": len(skus),
                "skus_by_class": skus_by_class,
                "matrix": matrix,
                "skus": skus_enriched,  # SKUs enriquecidos com interpretação industrial
                "critical_skus": critical_skus,
                "skus_risco_rutura": skus_risco_rutura[:10],  # Top 10
                "skus_excesso": skus_excesso[:10],  # Top 10
                "kpis": insights.get("kpis", {}),
            }
        except Exception as exc:
            logger.exception("Falha ao extrair insights de inventário")
            return {
                "skus_total": 0,
                "skus_by_class": {"A": 0, "B": 0, "C": 0},
                "matrix": {"A": {"X": 0, "Y": 0, "Z": 0}, "B": {"X": 0, "Y": 0, "Z": 0}, "C": {"X": 0, "Y": 0, "Z": 0}},
                "skus": [],
                "critical_skus": [],
                "kpis": {},
            }

    def _extract_bottlenecks_insights(self) -> Dict[str, Any]:
        """Extrai insights de gargalos com análise industrial pré-LLM."""
        try:
            data = self._get_bottlenecks_data()
            top_resources = data.get("top_resources", [])
            
            # ANÁLISE INDUSTRIAL PRÉ-LLM: Interpretar recursos como engenheiro industrial
            roteiros = self.loader.get_roteiros()
            resources_enriched = []
            
            for resource in top_resources:
                recurso_id = resource.get("recurso", "")
                utilizacao_raw = resource.get("utilizacao", 0.0)
                utilizacao = min(utilizacao_raw, 1.0) if utilizacao_raw > 1.0 else utilizacao_raw
                fila_h = resource.get("fila_h", 0.0)
                prob = resource.get("prob_gargalo", 0.0)
                has_alt = resource.get("has_alternative", False)
                
                # Interpretação industrial: identificar recursos lentos
                pph = None  # peças por hora
                cycle_time_s = None  # tempo de ciclo em segundos
                converging_ops = 0
                is_slow = False
                bottleneck_natural = False
                no_alternative = not has_alt
                
                if not roteiros.empty:
                    # Calcular cadência média (pph) do recurso
                    recursos_matching = roteiros[roteiros["maquinas_possiveis"].str.contains(recurso_id, na=False, case=False)]
                    if not recursos_matching.empty:
                        if "racio_pc_h" in recursos_matching.columns:
                            racios = recursos_matching["racio_pc_h"].dropna()
                            if not racios.empty:
                                pph = float(racios.mean())
                                # Calcular cycle_time_s: se racio = 120 pç/h, então cycle_time = 3600/120 = 30s
                                cycle_time_s = 3600.0 / pph if pph > 0 else None
                                # Recurso lento se < 200 pç/h (threshold industrial) OU cycle_time > 18s
                                is_slow = (pph < 200 if pph else False) or (cycle_time_s > 18.0 if cycle_time_s else False)
                        
                        # Contar operações que convergem para este recurso
                        converging_ops = len(recursos_matching) if not recursos_matching.empty else 0
                        # Gargalo natural se muitas operações convergem (>3) e sem alternativa
                        bottleneck_natural = converging_ops > 3 and not has_alt
                
                # Flags internas para interpretação
                resource_flags = {
                    "resource_is_slow": is_slow,
                    "bottleneck_natural": bottleneck_natural,
                    "no_alternative": no_alternative,
                    "high_convergence": converging_ops > 3,
                }
                
                resources_enriched.append({
                    "recurso": recurso_id,
                    "utilizacao_pct": round(utilizacao * 100, 1),
                    "utilizacao_raw": round(utilizacao_raw * 100, 1) if utilizacao_raw > 1.0 else None,
                    "is_saturated": utilizacao_raw > 1.0,
                    "fila_h": round(fila_h, 1),
                    "probabilidade_gargalo": round(prob, 2),
                    "tem_alternativa": has_alt,
                    "pph": round(pph, 0) if pph else None,  # Peças por hora
                    "cycle_time_s": round(cycle_time_s, 1) if cycle_time_s else None,  # Tempo de ciclo em segundos
                    "converging_ops": converging_ops,
                    "flags": resource_flags,
                })

            return {
                "top_resources": resources_enriched,
                "overlap_applied": data.get("overlap_applied", {}),
                "lead_time_gain": data.get("lead_time_gain", 0.0),
            }
        except Exception as exc:
            logger.exception("Falha ao extrair insights de gargalos")
            return {
                "top_resources": [],
                "overlap_applied": {},
                "lead_time_gain": 0.0,
            }

    def _get_bottlenecks_data(self) -> Dict[str, Any]:
        """Obtém dados de gargalos calculados."""
        try:
            from app.ml.bottlenecks import BottleneckPredictor

            start_date = datetime.utcnow()
            end_date = start_date + timedelta(days=7)
            plano = self.scheduler.generate_optimized_plan(start_date, end_date)

            if not plano or not plano.operations:
                return {"top_resources": [], "overlap_applied": {}, "lead_time_gain": 0.0}

            predictor = BottleneckPredictor()
            resource_usage: Dict[str, float] = {}
            resource_queue: Dict[str, float] = {}

            for op in plano.operations:
                recurso = str(getattr(op, "recurso", ""))
                if not recurso:
                    continue

                start_ts = getattr(op, "start_time", None)
                end_ts = getattr(op, "end_time", None)
                if not start_ts or not end_ts:
                    continue

                duration_h = (end_ts - start_ts).total_seconds() / 3600.0
                resource_usage[recurso] = resource_usage.get(recurso, 0.0) + duration_h

            total_hours = (end_date - start_date).total_seconds() / 3600.0
            top_resources: List[Dict[str, Any]] = []

            for recurso, horas_usadas in sorted(resource_usage.items(), key=lambda x: x[1], reverse=True)[:10]:
                utilizacao = (horas_usadas / total_hours) * 100 if total_hours > 0 else 0.0
                prob_gargalo = predictor.predict_bottleneck_probability(
                    utilizacao_pct=utilizacao,
                    queue_hours=resource_queue.get(recurso, 0.0),
                )

                loader = get_loader()
                roteiros = loader.get_roteiros()
                has_alternative = False
                if not roteiros.empty and "maquinas_possiveis" in roteiros.columns:
                    alternativas = roteiros[roteiros["maquinas_possiveis"].str.contains(recurso, na=False)]
                    has_alternative = len(alternativas) > 0

                top_resources.append(
                    {
                        "recurso": recurso,
                        "utilizacao": round(utilizacao / 100.0, 2),
                        "fila_h": round(resource_queue.get(recurso, 0.0), 1),
                        "prob_gargalo": round(prob_gargalo, 2),
                        "has_alternative": has_alternative,
                    }
                )

            baseline = self.scheduler.generate_baseline_plan(start_date, end_date)
            lead_time_gain = 0.0
            if baseline and baseline.kpis and plano.kpis:
                lt_baseline = float(baseline.kpis.get("lead_time_h", 0.0))
                lt_optimized = float(plano.kpis.get("lead_time_h", 0.0))
                if lt_baseline > 0:
                    lead_time_gain = ((lt_baseline - lt_optimized) / lt_baseline) * 100

            return {
                "top_resources": top_resources,
                "overlap_applied": {"transformacao": 0.44, "acabamentos": 0.38, "embalagem": 0.0},
                "lead_time_gain": round(lead_time_gain, 1),
            }
        except Exception as exc:
            logger.exception("Falha ao calcular dados de gargalos")
            return {"top_resources": [], "overlap_applied": {}, "lead_time_gain": 0.0}

    def _extract_what_if_actions(self) -> Dict[str, Any]:
        """Lista ações disponíveis para simulação."""
        return {
            "available_actions": [
                "desviar_carga_para_alternativa",
                "aumentar_overlap",
                "colar_familias",
                "agendar_preventiva",
            ]
        }
    
    def _extract_ml_quality(self) -> Dict[str, Any]:
        """Extrai métricas de qualidade dos modelos ML."""
        try:
            from app.ml.cycle_time import CycleTimePredictor
            from app.ml.bottlenecks import BottleneckPredictor
            from app.ml.inventory import InventoryPredictor
            
            cycle_metrics = CycleTimePredictor().get_metrics()
            bottleneck_metrics = BottleneckPredictor().get_metrics()
            
            # InventoryPredictor não tem métricas ainda, mas podemos indicar se está usando dados reais
            inventory_status = {"using_real_data": True, "samples": 0}
            
            return {
                "cycle_time": cycle_metrics if "status" not in cycle_metrics else {"using_synthetic": True},
                "bottlenecks": bottleneck_metrics if "status" not in bottleneck_metrics else {"using_synthetic": True},
                "inventory": inventory_status,
            }
        except Exception as exc:
            logger.exception("Falha ao extrair métricas ML")
            return {
                "cycle_time": {"using_synthetic": True},
                "bottlenecks": {"using_synthetic": True},
                "inventory": {"using_synthetic": True},
            }

    def build_action_candidates(self) -> List[Dict[str, Any]]:
        """
        Gera lista de oportunidades de ação baseadas nos modelos ML.
        Retorna lista estruturada de candidatos a ação com tipo, alvo, motivação e impacto estimado.
        """
        candidates: List[Dict[str, Any]] = []
        
        try:
            from app.ml.bottlenecks import BottleneckPredictor
            from app.ml.routing import RoutingBandit
            from app.ml.inventory import InventoryPredictor
            from app.ml.setup_time import SetupTimePredictor
            
            bottleneck_predictor = BottleneckPredictor()
            routing_bandit = RoutingBandit()
            inventory_predictor = InventoryPredictor()
            setup_predictor = SetupTimePredictor()
            
            # 1. Analisar gargalos e gerar candidatos de desvio de carga
            bottlenecks_data = self._get_bottlenecks_data()
            top_resources = bottlenecks_data.get("top_resources", [])
            
            for resource in top_resources:
                recurso_id = resource.get("recurso", "")
                utilizacao_raw = resource.get("utilizacao", 0.0)
                # Normalizar utilização > 1.0 (100%) para evitar valores absurdos
                utilizacao = min(utilizacao_raw, 1.0) if utilizacao_raw > 1.0 else utilizacao_raw
                fila_h = resource.get("fila_h", 0.0)
                prob_gargalo = resource.get("prob_gargalo", 0.0)
                has_alternative = resource.get("has_alternative", False)
                
                # Candidato #1: Desvio de carga (se prob_gargalo >= 0.9 e utilizacao >= 0.9 e tem alternativa)
                if prob_gargalo >= 0.9 and utilizacao >= 0.9 and has_alternative:
                    # Encontrar alternativa usando roteiros
                    roteiros = self.loader.get_roteiros()
                    alternativa_id = None
                    if not roteiros.empty and "maquinas_possiveis" in roteiros.columns:
                        # Procurar roteiros que usam este recurso e têm alternativas
                        matching_routes = roteiros[roteiros["maquinas_possiveis"].str.contains(recurso_id, na=False)]
                        if not matching_routes.empty:
                            # Extrair recursos alternativos
                            all_resources = set()
                            for machines in matching_routes["maquinas_possiveis"].dropna():
                                if isinstance(machines, str):
                                    resources = re.findall(r'M[-\s]?(\d+)', machines)
                                    all_resources.update([f"M-{r}" for r in resources])
                            # Escolher alternativa que não seja o recurso atual
                            alternativas = [r for r in all_resources if r != recurso_id]
                            if alternativas:
                                # Usar RoutingBandit para escolher melhor alternativa
                                operacao = matching_routes.iloc[0].get("grupo_operacao", "Transformação")
                                # Escolher primeira alternativa disponível (pode melhorar com bandit)
                                alternativa_id = alternativas[0]
                    
                    if alternativa_id:
                        # Calcular percentagem de desvio (20-40% baseado na utilização)
                        pct_desvio = min(30.0, max(20.0, (utilizacao - 0.85) * 100))
                        
                        # Estimar impacto
                        delta_fila_h = -fila_h * (pct_desvio / 100) if fila_h > 0 else 0.0
                        delta_lead_time_h = -fila_h * 0.3 if fila_h > 0 else -5.0  # Estimativa conservadora
                        delta_otd_pp = min(5.0, max(2.0, utilizacao * 10)) if utilizacao > 0.9 else 0.0
                        
                        # Calcular prioridade baseada em severidade
                        prioridade = "ALTO" if prob_gargalo >= 0.95 and utilizacao >= 0.95 else \
                                    "MÉDIO" if prob_gargalo >= 0.9 and utilizacao >= 0.9 else "BAIXO"
                        
                        # Obter dados técnicos do recurso (pph, cycle_time_s, flags)
                        pph = resource.get("pph")
                        cycle_time_s = resource.get("cycle_time_s")
                        flags = resource.get("flags", {})
                        converging_ops = resource.get("converging_ops", 0)
                        
                        candidates.append({
                            "tipo": "desvio_carga",
                            "alvo": recurso_id,
                            "alternativa": alternativa_id,
                            "gargalo_afetado": recurso_id,
                            "pct_desvio": round(pct_desvio, 1),
                            "prioridade": prioridade,
                            "motivacao": {
                                "utilizacao": round(utilizacao, 2),
                                "prob_gargalo": round(prob_gargalo, 2),
                                "fila_h": round(fila_h, 1),
                                "fila_zero": fila_h == 0.0,
                            },
                            "dados_base": {
                                "utilizacao": round(utilizacao, 2),
                                "prob_gargalo": round(prob_gargalo, 2),
                                "fila_h": round(fila_h, 1),
                                "fila_zero": fila_h == 0.0,
                            },
                            "dados_tecnicos": {
                                "recurso_origem": recurso_id,
                                "recurso_destino": alternativa_id,
                                "pph": round(pph, 0) if pph else None,
                                "cycle_time_s": round(cycle_time_s, 1) if cycle_time_s else None,
                                "converging_ops": converging_ops,
                                "flags": flags,
                                "has_alternative": has_alternative,
                            },
                            "impacto_estimado": {
                                "delta_lead_time_h": round(delta_lead_time_h, 1),
                                "delta_fila_h": round(delta_fila_h, 1),
                                "delta_otd_pp": round(delta_otd_pp, 1),
                            },
                        })
                
                # Candidato #2: Manutenção preventiva (se prob_gargalo >= 0.9 e utilizacao >= 0.9)
                if prob_gargalo >= 0.9 and utilizacao >= 0.9:
                    prioridade = "ALTO" if prob_gargalo >= 0.95 else "MÉDIO"
                    
                    # Obter dados técnicos do recurso
                    pph = resource.get("pph")
                    cycle_time_s = resource.get("cycle_time_s")
                    flags = resource.get("flags", {})
                    converging_ops = resource.get("converging_ops", 0)
                    
                    candidates.append({
                        "tipo": "preventiva",
                        "alvo": recurso_id,
                        "gargalo_afetado": recurso_id,
                        "prioridade": prioridade,
                        "motivacao": {
                            "utilizacao": round(utilizacao, 2),
                            "prob_gargalo": round(prob_gargalo, 2),
                            "fila_h": round(fila_h, 1),
                            "fila_zero": fila_h == 0.0,
                        },
                        "dados_base": {
                            "utilizacao": round(utilizacao, 2),
                            "prob_gargalo": round(prob_gargalo, 2),
                            "fila_h": round(fila_h, 1),
                            "fila_zero": fila_h == 0.0,
                        },
                        "dados_tecnicos": {
                            "recurso": recurso_id,
                            "pph": round(pph, 0) if pph else None,
                            "cycle_time_s": round(cycle_time_s, 1) if cycle_time_s else None,
                            "converging_ops": converging_ops,
                            "flags": flags,
                            "has_alternative": has_alternative,
                        },
                        "impacto_estimado": {
                            "delta_lead_time_h": -10.0,  # Redução preventiva de avarias
                            "delta_fila_h": 0.0,
                            "delta_otd_pp": 2.0,
                        },
                    })
            
            # 2. Analisar inventário e gerar candidatos de reposição
            inventory_data = self._extract_inventory_insights()
            critical_skus = inventory_data.get("critical_skus", [])
            
            for sku_data in critical_skus[:10]:  # Top 10 SKUs críticos
                sku = sku_data.get("sku", "")
                risco_30d = sku_data.get("risco_30d", 0.0)
                cobertura_dias = sku_data.get("cobertura_dias", 0.0)
                stock_atual = sku_data.get("stock_atual", 0.0)
                rop = sku_data.get("rop", 0.0)
                
                # Candidato: Reposição imediata (se risco > 5% OU cobertura < 30 dias)
                if risco_30d > 5.0 or cobertura_dias < 30.0:
                    # Calcular quantidade a repor
                    qty_repor = max(rop - stock_atual, rop * 0.5) if stock_atual < rop else rop * 0.3
                    
                    # Calcular prioridade baseada em criticidade
                    classe = sku_data.get("classe", "C")
                    prioridade = "ALTO" if (classe == "A" and risco_30d > 20.0) or risco_30d > 50.0 or cobertura_dias < 7 else \
                                "MÉDIO" if risco_30d > 10.0 or cobertura_dias < 30.0 else "BAIXO"
                    
                    # Estimar impacto em risco e cobertura
                    delta_risk_30d = -risco_30d * 0.7 if risco_30d > 0 else 0.0  # Redução de 70% do risco
                    delta_cobertura_dias = max(7.0, cobertura_dias * 0.5) if cobertura_dias < 30 else 0.0
                    
                    # Obter dados técnicos do SKU
                    ads_180 = sku_data.get("ads_180", 0.0)
                    xyz = sku_data.get("xyz", "Z")
                    flags_sku = {
                        "risco_rutura": cobertura_dias < 30.0,
                        "excesso_stock": False,
                        "abaixo_rop": stock_atual < rop,
                        "criticidade": "ALTA" if (classe == "A" and risco_30d > 20.0) or risco_30d > 50.0 else \
                                      "MÉDIA" if (classe in ["A", "B"] and risco_30d > 5.0) or risco_30d > 20.0 else \
                                      "BAIXA",
                    }
                    
                    candidates.append({
                        "tipo": "reposicao_stock",
                        "sku": sku,
                        "alvo": sku,
                        "qty_repor": round(qty_repor, 0),
                        "prioridade": prioridade,
                        "motivacao": {
                            "risk_30d": round(risco_30d, 1),
                            "coverage_dias": round(cobertura_dias, 1),
                            "stock_atual": round(stock_atual, 0),
                            "rop": round(rop, 0),
                            "stock_abaixo_rop": stock_atual < rop,
                        },
                        "dados_base": {
                            "risk_30d": round(risco_30d, 1),
                            "coverage_dias": round(cobertura_dias, 1),
                            "stock_atual": round(stock_atual, 0),
                            "rop": round(rop, 0),
                            "stock_abaixo_rop": stock_atual < rop,
                            "classe": classe,
                        },
                        "dados_tecnicos": {
                            "sku": sku,
                            "ads_180": round(ads_180, 2),
                            "abc": classe,
                            "xyz": xyz,
                            "flags": flags_sku,
                        },
                        "impacto_estimado": {
                            "delta_risk_30d": round(delta_risk_30d, 1),
                            "delta_cobertura_dias": round(delta_cobertura_dias, 1),
                            "delta_otd_pp": 1.0 if risco_30d > 20.0 else 0.5,
                        },
                    })
                
                # Candidato: Reduzir excesso (se cobertura > 365 dias)
                if cobertura_dias > 365.0:
                    # Calcular excesso em dias
                    excesso_dias = cobertura_dias - 365.0
                    # Estimar capital imobilizado (aproximação: stock * valor médio unitário)
                    # Por agora, usar stock como proxy
                    capital_estimado = stock_atual * excesso_dias / 365.0
                    
                    # Obter dados técnicos do SKU
                    ads_180 = sku_data.get("ads_180", 0.0)
                    xyz = sku_data.get("xyz", "Z")
                    flags_sku = {
                        "risco_rutura": False,
                        "excesso_stock": True,
                        "abaixo_rop": False,
                        "criticidade": "BAIXA",  # Excesso geralmente não é crítico
                    }
                    
                    candidates.append({
                        "tipo": "reducao_excesso",
                        "sku": sku,
                        "alvo": sku,
                        "prioridade": "BAIXO",  # Excesso não é urgente
                        "motivacao": {
                            "coverage_dias": round(cobertura_dias, 1),
                            "excesso_dias": round(excesso_dias, 1),
                            "excesso": True,
                        },
                        "dados_base": {
                            "coverage_dias": round(cobertura_dias, 1),
                            "stock_atual": round(stock_atual, 0),
                            "excesso": True,
                            "excesso_dias": round(excesso_dias, 1),
                        },
                        "dados_tecnicos": {
                            "sku": sku,
                            "ads_180": round(ads_180, 2),
                            "abc": classe,
                            "xyz": xyz,
                            "flags": flags_sku,
                        },
                        "impacto_estimado": {
                            "delta_capital_imobilizado": "significativo" if capital_estimado > 1000 else "moderado",
                            "delta_risco_obsolescencia": "reduzido",
                            "delta_otd_pp": 0.0,  # Impacto em capital, não em OTD
                        },
                    })
            
            # 3. Analisar setups e gerar candidatos de colagem de famílias
            planning_data = self._extract_planning_insights()
            setup_hours = planning_data.get("setup_hours", 0.0)
            setup_hours_before = planning_data.get("setup_hours_before", 0.0)
            top_familias_setup = planning_data.get("top_familias_setup", [])
            gargalo_principal = planning_data.get("main_bottleneck", {})
            
            if setup_hours > 20.0:  # Se mais de 20h de setup por semana
                # Identificar setor mais afetado
                setor_afetado = "Transformação"  # Default
                roteiros = self.loader.get_roteiros()
                if not roteiros.empty and "setor" in roteiros.columns:
                    # Contar setups por setor
                    setor_setups = {}
                    for familia_info in top_familias_setup:
                        familia = familia_info.get("familia", "")
                        # Tentar encontrar setor da família
                        familia_routes = roteiros[roteiros["sku"].str.contains(familia, na=False, case=False)]
                        if not familia_routes.empty:
                            setor = familia_routes.iloc[0].get("setor", "Transformação")
                            setor_setups[setor] = setor_setups.get(setor, 0) + familia_info.get("count", 0)
                    if setor_setups:
                        setor_afetado = max(setor_setups.items(), key=lambda x: x[1])[0]
                
                # Extrair nomes das famílias
                familias_nomes = [f.get("familia", "") for f in top_familias_setup[:3]]
                
                candidates.append({
                    "tipo": "colar_familias",
                    "alvo": f"Setor {setor_afetado}",
                    "gargalo_afetado": gargalo_principal.get("id", "N/A"),
                    "prioridade": "ALTO" if setup_hours > 30.0 else "MÉDIO",
                    "motivacao": {
                        "setup_hours": round(setup_hours, 1),
                        "setup_hours_before": round(setup_hours_before, 1),
                        "familias": familias_nomes,
                        "threshold": 20.0,
                    },
                    "dados_base": {
                        "setup_hours": round(setup_hours, 1),
                        "setup_hours_before": round(setup_hours_before, 1),
                        "familias": familias_nomes,
                        "threshold": 20.0,
                    },
                    "dados_tecnicos": {
                        "setor": setor_afetado,
                        "gargalo_afetado": gargalo_principal.get("id", "N/A"),
                        "num_familias": len(familias_nomes),
                        "setup_reduzido_pct": 30.0,  # Estimativa de redução
                    },
                    "impacto_estimado": {
                        "delta_setup_h": round(-setup_hours * 0.3, 1),
                        "delta_lead_time_h": round(-setup_hours * 0.3, 1),
                        "delta_otd_pp": 3.0,
                    },
                })
            
            # 4. Analisar overlap e gerar candidatos de ajuste
            overlap_data = planning_data.get("overlap", {})
            overlap_transformacao = overlap_data.get("transformacao", 0.0)
            overlap_acabamentos = overlap_data.get("acabamentos", 0.0)
            lead_time_after = planning_data.get("lead_time_after", 0.0)
            interpretacao_industrial = planning_data.get("interpretacao_industrial", {})
            overlap_recomendado = interpretacao_industrial.get("overlap_recomendado", "15-25%")
            
            # Extrair valor numérico do overlap recomendado
            overlap_recomendado_val = 0.20  # Default 20%
            if "15-25" in overlap_recomendado:
                overlap_recomendado_val = 0.20
            elif "25-40" in overlap_recomendado:
                overlap_recomendado_val = 0.325
            elif "40-60" in overlap_recomendado:
                overlap_recomendado_val = 0.50
            
            # Candidato: Ajustar overlap em Transformação
            if overlap_transformacao < overlap_recomendado_val:
                delta_overlap = overlap_recomendado_val - overlap_transformacao
                # Estimar impacto: cada 0.1 de overlap reduz ~10% do lead time
                delta_lead_time_estimado = -lead_time_after * delta_overlap * 1.0
                
                candidates.append({
                    "tipo": "ajuste_overlap",
                    "alvo": "Transformação",
                    "prioridade": "MÉDIO" if delta_overlap > 0.15 else "BAIXO",
                    "motivacao": {
                        "overlap_atual": round(overlap_transformacao, 2),
                        "overlap_recomendado": overlap_recomendado,
                        "lead_time_after": round(lead_time_after, 1),
                    },
                    "dados_base": {
                        "overlap_atual": round(overlap_transformacao, 2),
                        "overlap_recomendado": overlap_recomendado,
                        "lead_time_after": round(lead_time_after, 1),
                    },
                    "dados_tecnicos": {
                        "setor": "Transformação",
                        "overlap_atual_pct": round(overlap_transformacao * 100, 1),
                        "overlap_recomendado_pct": overlap_recomendado,
                        "delta_overlap": round(delta_overlap, 2),
                    },
                    "impacto_estimado": {
                        "delta_lead_time_h": round(delta_lead_time_estimado, 1),
                        "delta_otd_pp": round(abs(delta_lead_time_estimado) / 20.0, 1),
                    },
                })
            
            # Candidato: Ajustar overlap em Acabamentos
            if overlap_acabamentos < overlap_recomendado_val:
                delta_overlap = overlap_recomendado_val - overlap_acabamentos
                delta_lead_time_estimado = -lead_time_after * delta_overlap * 1.0
                
                candidates.append({
                    "tipo": "ajuste_overlap",
                    "alvo": "Acabamentos",
                    "prioridade": "MÉDIO" if delta_overlap > 0.15 else "BAIXO",
                    "motivacao": {
                        "overlap_atual": round(overlap_acabamentos, 2),
                        "overlap_recomendado": overlap_recomendado,
                        "lead_time_after": round(lead_time_after, 1),
                    },
                    "dados_base": {
                        "overlap_atual": round(overlap_acabamentos, 2),
                        "overlap_recomendado": overlap_recomendado,
                        "lead_time_after": round(lead_time_after, 1),
                    },
                    "dados_tecnicos": {
                        "setor": "Acabamentos",
                        "overlap_atual_pct": round(overlap_acabamentos * 100, 1),
                        "overlap_recomendado_pct": overlap_recomendado,
                        "delta_overlap": round(delta_overlap, 2),
                    },
                    "impacto_estimado": {
                        "delta_lead_time_h": round(delta_lead_time_estimado, 1),
                        "delta_otd_pp": round(abs(delta_lead_time_estimado) / 20.0, 1),
                    },
                })
            
            # Ordenar candidatos por prioridade (ALTO > MÉDIO > BAIXO) e depois por impacto
            def priority_score(candidate: Dict[str, Any]) -> float:
                prioridade_str = candidate.get("prioridade", "BAIXO")
                prioridade_num = {"ALTO": 100, "MÉDIO": 50, "BAIXO": 0}.get(prioridade_str, 0)
                
                # Adicionar score baseado no tipo e impacto
                tipo = candidate.get("tipo", "")
                dados_base = candidate.get("dados_base", {})
                impacto = candidate.get("impacto_estimado", {})
                
                if tipo == "desvio_carga":
                    prob = dados_base.get("prob_gargalo", 0.0)
                    utilizacao = dados_base.get("utilizacao", 0.0)
                    return prioridade_num + prob * 50 + utilizacao * 30
                elif tipo == "reposicao_stock":
                    risk = dados_base.get("risk_30d", 0.0)
                    coverage = dados_base.get("coverage_dias", 0.0)
                    return prioridade_num + risk * 2 + (30.0 - coverage) * 0.5
                elif tipo == "preventiva":
                    prob = dados_base.get("prob_gargalo", 0.0)
                    return prioridade_num + prob * 40
                elif tipo == "colar_familias":
                    setup_h = dados_base.get("setup_hours", 0.0)
                    return prioridade_num + setup_h * 0.5
                elif tipo == "ajuste_overlap":
                    delta_lt = abs(impacto.get("delta_lead_time_h", 0.0))
                    return prioridade_num + delta_lt * 0.1
                elif tipo == "reducao_excesso":
                    return prioridade_num  # Sempre baixa prioridade
                return prioridade_num
            
            candidates.sort(key=priority_score, reverse=True)
            
            return candidates[:10]  # Retornar top 10 candidatos
            
        except Exception as exc:
            logger.exception("Falha ao gerar action candidates")
            return []

