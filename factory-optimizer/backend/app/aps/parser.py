"""
Parser específico para Nikufra DadosProducao (2).xlsx - ProdPlan 4.0

REGRAS OBRIGATÓRIAS:
1. Cada folha = 1 artigo GO → 1 Order
2. Linhas consecutivas com mesmo (Rota, Operação) = alternativas OR (mutuamente exclusivas)
3. Deve ler TODAS as folhas do Excel
4. A mesma operação NÃO pode aparecer em duas máquinas simultaneamente
"""

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app.aps.models import OpAlternative, OpRef, Order

logger = logging.getLogger(__name__)


class ProductionDataParser:
    """Parser específico para Nikufra DadosProducao (2).xlsx com regras industriais rigorosas."""
    
    def __init__(self):
        # Mapeamento de sinónimos de colunas (incluindo variações com acentos)
        self.column_synonyms = {
            "rota": ["rota", "route", "alternativa", "alternativa_rota"],
            "operacao": [
                "operacao", "op", "codigo_operacao", "cod_op", "operação",
                "operação"  # Com acento
            ],
            "maquinas_possiveis": [
                "maquinas_possiveis", "maquinas", "machines", "recursos", 
                "maquina", "máquinas possíveis", "máquinas_possíveis",
                "máquinas possiveis", "máquinas_possiveis",  # Sem acento no "possiveis"
                "máquinas possiveis"  # Variação encontrada no Excel
            ],
            "ratio_pch": [
                "ratio_pch", "racio_pc_h", "racio_pch", "pcs_por_h", 
                "pecas_por_hora", "pc_h", "throughput", "racio peças/hora",
                "racio_pecas_hora", "ratio_pecas_hora",
                "racio peças/hora", "racio peças/hora"  # Com acento
            ],
            "pessoas": [
                "pessoas", "num_pessoas", "qtd_pessoas", "colaboradores", 
                "people", "pessoas necessárias", "pessoas_necessarias",
                "pessoas necessárias"  # Com acento
            ],
            "grupo_operacao": [
                "grupo_operacao", "grupo_op", "family", "familia", "grupo",
                "grupo operação", "grupo_operação",  # Com acento
                "grupo operação"  # Variação encontrada no Excel
            ],
        }
    
    def parse_dadosnikufra2(self, file_path: str) -> List[Order]:
        """
        Parse Nikufra DadosProducao (2).xlsx - lê TODAS as folhas e cria uma Order por folha.
        
        Args:
            file_path: Caminho para o ficheiro Excel (Nikufra DadosProducao (2).xlsx)
        
        Returns:
            List[Order] - uma Order por folha (artigo GO)
        """
        try:
            excel_file = pd.ExcelFile(file_path)
            all_orders: List[Order] = []
            
            logger.info(f"Abrindo {file_path} com {len(excel_file.sheet_names)} folhas")
            
            # Iterar por TODAS as folhas
            for sheet_name in excel_file.sheet_names:
                try:
                    # Ignorar folhas que não são artigos (ex: WIKI, índices, etc.)
                    if sheet_name.upper() in ['WIKI', 'ÍNDICE', 'INDEX', 'LEIA-ME', 'README', 'INFO']:
                        logger.info(f"Folha '{sheet_name}' ignorada (folha informativa)")
                        continue
                    
                    logger.info(f"Processando folha: '{sheet_name}'")
                    
                    # Ler folha inteira
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    if df.empty:
                        logger.warning(f"Folha '{sheet_name}' está vazia. A saltar.")
                        continue
                    
                    # Normalizar colunas
                    df = self._normalize_columns(df)
                    
                    # Determinar artigo da folha
                    artigo = self._extract_artigo_from_sheet(df, sheet_name)
                    
                    # Construir Order para esta folha
                    order = self._build_order_from_sheet(df, artigo, sheet_name)
                    
                    if order and order.operations:
                        all_orders.append(order)
                        logger.info(
                            f"Folha '{sheet_name}': Order {order.id} criada com "
                            f"{len(order.operations)} operações"
                        )
                    else:
                        logger.warning(f"Folha '{sheet_name}': nenhuma operação válida encontrada")
                        
                except Exception as exc:
                    logger.exception(f"Erro ao processar folha '{sheet_name}': {exc}")
                    continue
            
            logger.info(f"Parse completo: {len(all_orders)} Orders criadas de {len(excel_file.sheet_names)} folhas")
            
            # VALIDAÇÃO: Garantir que temos pelo menos 6 Orders (GO Artigo 1-6)
            if len(all_orders) < 6:
                logger.warning(
                    f"⚠️ AVISO: Esperava pelo menos 6 Orders (GO Artigo 1-6), mas apenas {len(all_orders)} foram criadas. "
                    f"Folhas processadas: {[o.artigo for o in all_orders]}"
                )
            
            # VALIDAÇÃO: Verificar invariantes de cada Order
            for order in all_orders:
                # Verificar que cada Order tem operações
                if not order.operations:
                    logger.error(f"❌ Order {order.id} ({order.artigo}) não tem operações!")
                    continue
                
                # Verificar que não há duplicados de (stage_index, rota, op_id)
                op_keys = {}
                for op_ref in order.operations:
                    key = (op_ref.stage_index, op_ref.rota, op_ref.op_id)
                    if key in op_keys:
                        logger.error(
                            f"❌ DUPLICADO no parser: Order {order.id}, "
                            f"OpRef duplicado: stage={op_ref.stage_index}, rota={op_ref.rota}, op_id={op_ref.op_id}"
                        )
                    op_keys[key] = op_ref
                
                # Verificar que cada OpRef tem alternativas
                ops_sem_alt = [op for op in order.operations if len(op.alternatives) == 0]
                if ops_sem_alt:
                    logger.warning(
                        f"⚠️ Order {order.id}: {len(ops_sem_alt)} operações sem alternativas: "
                        f"{[(op.op_id, op.rota, op.stage_index) for op in ops_sem_alt[:3]]}"
                    )
            
            return all_orders
            
        except Exception as exc:
            logger.exception(f"Erro ao fazer parse do Excel: {exc}")
            raise
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nomes de colunas para formato canónico."""
        df_renamed = df.copy()
        
        # Criar mapping reverso
        reverse_mapping = {}
        for canonical, synonyms in self.column_synonyms.items():
            for synonym in synonyms:
                reverse_mapping[synonym.lower().strip()] = canonical
        
        # Mapeamentos adicionais para colunas específicas do Excel
        additional_mappings = {
            "ordem grupo": "ordem_grupo",
            "ordem_grupo": "ordem_grupo",
            "% da anterior": "overlap_pct",
            "overlap_pct": "overlap_pct",
            "tempo de setup (minutos)": "setup_minutos",
            "tempo de setup": "setup_minutos",
            "setup_minutos": "setup_minutos",
        }
        reverse_mapping.update(additional_mappings)
        
        # Renomear colunas
        new_columns = []
        for col in df.columns:
            col_lower = str(col).lower().strip()
            canonical = reverse_mapping.get(col_lower, col_lower)
            new_columns.append(canonical)
        
        df_renamed.columns = new_columns
        
        return df_renamed
    
    def _extract_artigo_from_sheet(self, df: pd.DataFrame, sheet_name: str) -> str:
        """
        Extrai identificador do artigo da folha.
        
        Prioridade:
        1. Nome da folha (ex: "GO Artigo 6" → "GO Artigo 6" ou "GO6")
        2. Coluna "artigo" se existir
        3. Usar sheet_name como fallback
        """
        import re
        
        # Tentar extrair do nome da folha
        artigo = sheet_name.strip()
        
        # Se houver coluna "artigo", usar primeiro valor único
        if "artigo" in df.columns:
            artigos_unicos = df["artigo"].dropna().unique()
            if len(artigos_unicos) > 0:
                artigo = str(artigos_unicos[0]).strip()
                logger.debug(f"Artigo extraído da coluna: {artigo}")
                return artigo
        
        # Limpar nome da folha - manter formato "GO Artigo X" ou extrair "GOX"
        if "GO" in artigo.upper():
            # Tentar extrair "GO Artigo 6" ou "GO6" de "GO Artigo 6"
            match = re.search(r'GO\s+Artigo\s+(\d+)', artigo, re.IGNORECASE)
            if match:
                # Manter formato "GO Artigo X"
                artigo = f"GO Artigo {match.group(1)}"
            else:
                # Tentar extrair apenas número após GO
                match = re.search(r'GO\s*(\d+)', artigo, re.IGNORECASE)
                if match:
                    artigo = f"GO{match.group(1)}"
        
        return artigo
    
    def _build_order_from_sheet(
        self, df: pd.DataFrame, artigo: str, sheet_name: str
    ) -> Optional[Order]:
        """
        Constrói uma Order a partir de uma folha do Excel.
        
        Regra CRÍTICA: Linhas consecutivas com mesmo (Rota, Operação) = alternativas OR.
        """
        # Remover linhas completamente vazias
        df = df.dropna(how="all")
        
        if df.empty:
            return None
        
        # Validar colunas obrigatórias
        required = ["rota", "operacao", "maquinas_possiveis", "ratio_pch"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.warning(f"Folha '{sheet_name}' sem colunas obrigatórias: {missing}")
            return None
        
        # Verificar se tem Ordem Grupo (obrigatório para sequência)
        if "ordem_grupo" not in df.columns:
            logger.warning(f"Folha '{sheet_name}' sem coluna 'Ordem Grupo'. Tentando inferir sequência...")
            # Tentar inferir sequência pela ordem das linhas (fallback)
            df["ordem_grupo"] = range(1, len(df) + 1)
        
        # Construir OpRefs agrupando por (Rota, Operação)
        op_refs = self._build_op_refs_from_dataframe(df, artigo)
        
        if not op_refs:
            logger.warning(f"Folha '{sheet_name}': nenhum OpRef válido criado")
            return None
        
        # Criar Order
        order = Order(
            id=f"ORD-{artigo}",
            artigo=artigo,
            quantidade=1000,  # Default para demo
            prioridade="NORMAL",
            operations=op_refs,
        )
        
        return order
    
    def _build_op_refs_from_dataframe(
        self, df: pd.DataFrame, artigo: str
    ) -> List[OpRef]:
        """
        Constrói OpRefs a partir do DataFrame.
        
        REGRA CRÍTICA: Linhas com mesmo (Ordem Grupo, Rota, Operação) = alternativas OR.
        O stage_index vem diretamente da coluna "Ordem Grupo".
        """
        op_refs: List[OpRef] = []
        
        # Remover linhas sem dados essenciais
        df = df.dropna(subset=["rota", "operacao", "maquinas_possiveis"])
        
        if df.empty:
            logger.warning(f"DataFrame vazio após limpeza para artigo {artigo}")
            return []
        
        # Ordenar por Ordem Grupo, depois por Rota, depois por Operação
        if "ordem_grupo" in df.columns:
            df = df.sort_values(by=["ordem_grupo", "rota", "operacao"])
        else:
            df = df.sort_index()
        
        # Agrupar por (Ordem Grupo, Rota, Operação) - estas são as alternativas OR
        grouped = df.groupby(["ordem_grupo", "rota", "operacao"], sort=False)
        
        # Construir OpRefs para cada grupo
        for (ordem_grupo, rota, operacao), group_df in grouped:
            # Converter ordem_grupo para int
            try:
                stage_index = int(float(ordem_grupo))
            except (ValueError, TypeError):
                logger.warning(f"Ordem Grupo inválida: {ordem_grupo}. A saltar.")
                continue
            
            # Normalizar rota e operação
            rota = str(rota).strip().upper()
            operacao = str(operacao).strip()
            
            if not rota or not operacao:
                continue
            
            # Construir alternativas (cada linha do grupo = uma alternativa)
            alternatives: List[OpAlternative] = []
            
            for idx, row in group_df.iterrows():
                # Extrair máquinas
                maquinas_str = str(row.get("maquinas_possiveis", "")).strip() if pd.notna(row.get("maquinas_possiveis")) else ""
                maquinas = self._parse_machines(maquinas_str)
                if not maquinas:
                    logger.warning(f"Linha {idx}: sem máquinas válidas em '{maquinas_str}'")
                    continue
                
                # Extrair ratio_pch
                ratio_pch = self._safe_float(row.get("ratio_pch"), 0.0)
                if ratio_pch <= 0:
                    logger.warning(f"Linha {idx}: ratio_pch inválido ({ratio_pch})")
                    continue
                
                # Extrair pessoas (default 1.0)
                pessoas = self._safe_float(row.get("pessoas"), 1.0)
                
                # Extrair grupo operação (family)
                grupo_op = str(row.get("grupo_operacao", "")).strip() if pd.notna(row.get("grupo_operacao")) else ""
                if not grupo_op:
                    grupo_op = operacao  # Fallback para código da operação
                
                # Extrair overlap_pct (% da anterior)
                overlap_pct = self._safe_float(row.get("overlap_pct"), 0.0)
                # Garantir que está entre 0 e 1
                overlap_pct = max(0.0, min(1.0, overlap_pct))
                
                # Extrair setup em minutos e converter para horas
                setup_minutos = self._safe_float(row.get("setup_minutos"), 0.0)
                setup_h = setup_minutos / 60.0
                
                # Criar uma alternativa por máquina
                for maquina_id in maquinas:
                    alternative = OpAlternative(
                        maquina_id=maquina_id,
                        ratio_pch=ratio_pch,
                        pessoas=pessoas,
                        family=grupo_op,
                        setup_h=setup_h,
                        overlap_pct=overlap_pct,
                    )
                    alternatives.append(alternative)
                    logger.debug(
                        f"Alternativa: {maquina_id} para {rota}/{operacao} (Ordem Grupo {stage_index}), "
                        f"overlap={overlap_pct:.1%}, setup={setup_h:.2f}h"
                    )
            
            if not alternatives:
                logger.warning(f"Grupo (Ordem Grupo {stage_index}, {rota}, {operacao}) sem alternativas válidas")
                continue
            
            # Calcular precedências: todas as operações com stage_index menor
            precedencias = [i for i in range(1, stage_index) if i < stage_index]
            
            # Criar OpRef
            op_ref = OpRef(
                op_id=operacao,
                rota=rota,
                stage_index=stage_index,
                precedencias=precedencias,
                operacao_logica=grupo_op if group_df["grupo_operacao"].notna().any() else operacao,
                alternatives=alternatives,
            )
            op_refs.append(op_ref)
            logger.debug(
                f"Criado OpRef: Ordem Grupo {stage_index}, {rota}/{operacao}, "
                f"{len(alternatives)} alternativas"
            )
        
        # Ordenar por stage_index e rota
        op_refs.sort(key=lambda x: (x.stage_index, x.rota))
        
        logger.info(f"Artigo {artigo}: {len(op_refs)} operações criadas de {len(grouped)} grupos")
        return op_refs
    
    def _parse_machines(self, maquinas_str: str) -> List[str]:
        """
        Extrai lista de IDs de máquinas de string.
        
        Aceita formatos:
        - "244"
        - "M-244"
        - "244, 020"
        - "244; 020"
        - "244 020"
        """
        if not maquinas_str or pd.isna(maquinas_str):
            return []
        
        # Remover prefixos M- e espaços
        maquinas_str = str(maquinas_str).replace("M-", "").replace("m-", "").strip()
        
        # Separar por vírgula, ponto e vírgula, ou espaço
        machines = []
        for part in maquinas_str.replace(",", " ").replace(";", " ").split():
            machine_id = part.strip()
            # Aceitar apenas números (IDs de máquina)
            if machine_id and (machine_id.isdigit() or machine_id.replace(".", "").isdigit()):
                # Remover decimais se houver
                machine_id = machine_id.split(".")[0]
                machines.append(machine_id)
        
        return machines
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Converte valor para float com fallback."""
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


def parse_dadosnikufra2(file_path: str) -> List[Order]:
    """
    Função de conveniência para parse de Nikufra DadosProducao (2).xlsx.
    
    Args:
        file_path: Caminho para o ficheiro Excel (Nikufra DadosProducao (2).xlsx)
    
    Returns:
        List[Order] - uma Order por folha (artigo GO)
    """
    parser = ProductionDataParser()
    return parser.parse_dadosnikufra2(file_path)
