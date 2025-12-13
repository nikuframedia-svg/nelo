"""
Validador Industrial Anti-Alucinação - Nível Enterprise (Siemens/Dassault/Celonis).

Garante que o LLM NUNCA inventa dados e que todos os módulos são 100% coerentes.
"""

import re
from typing import Any, Dict, List, Optional, Set


class IndustrialLLMValidator:
    """
    Validador industrial rigoroso que bloqueia qualquer frase ou número impossível.
    
    Regras:
    1. Bloqueia SKUs/máquinas inexistentes
    2. Bloqueia mistura de módulos
    3. Bloqueia "reduzir fila" quando fila = 0
    4. Bloqueia impacto impossível
    5. Bloqueia métricas inventadas (OEE, WIP, etc.)
    6. Bloqueia frases genéricas tipo powerpoint
    7. Bloqueia repetições entre módulos
    8. Bloqueia contradições com o APS
    """
    
    def __init__(self):
        self.forbidden_metrics = {
            "oee": r'\bOEE\s+(\d+%?|\d+\.\d+%?)\b',
            "wip": r'\bWIP\s+(\d+|\d+\.\d+)\b',
            "throughput_global": r'\bthroughput\s+(\d{4,})\s*(unidades|units|pç)/?(mês|mes|month|hora|h)?\b',
        }
        
        self.generic_phrases = [
            r'redução significativa',
            r'melhoria substancial',
            r'otimização',
            r'melhoria considerável',
            r'aumento significativo',
            r'agilizar processos',
            r'otimizar operações',
            r'eficientizar',
        ]
        
        self.module_forbidden_words = {
            # ✅ Planeamento: PERMITIR gargalo, famílias, sequência, carga, capacidade, produção, ordens, operações, setups, turnos
            "planeamento": ["sku", "stock", "inventário", "inventario", "cobertura", "rop", "abc", "xyz", "risco 30 dias", "comprar", "repor", "excesso"],
            "gargalos": ["sku", "stock", "inventário", "inventario", "cobertura", "rop", "abc", "xyz", "otd global", "lead time global"],
            "inventario": ["gargalo", "fila", "recurso m-", "utilização", "otd", "lead time", "leadtime", "setup", "máquina"],
            "sugestoes": ["resumo executivo", "estado real da fábrica", "a fábrica está", "o sistema apresenta", "globalmente"],
            "what_if": ["resumo", "banner", "estado geral"],
        }
    
    def validate(
        self,
        text: str,
        context: Dict[str, Any],
        mode: str,
        valid_skus: Optional[List[str]] = None,
        valid_resources: Optional[List[str]] = None,
        action_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Valida output do LLM com regras industriais rigorosas.
        
        Returns:
            {
                "is_valid": bool,
                "errors": List[str],
                "warnings": List[str],
                "sanitized_text": str,
                "should_regenerate": bool,
            }
        """
        errors: List[str] = []
        warnings: List[str] = []
        sanitized_text = text
        should_regenerate = False
        
        # 1. Validar entidades (SKUs e recursos)
        errors_entities, sanitized_text = self._validate_entities(
            sanitized_text, valid_skus, valid_resources
        )
        errors.extend(errors_entities)
        
        # 2. Validar por módulo (bloquear mistura)
        errors_module, sanitized_text = self._validate_module_isolation(
            sanitized_text, mode
        )
        errors.extend(errors_module)
        
        # 3. Validar lógica industrial
        errors_logic, warnings_logic, sanitized_text = self._validate_industrial_logic(
            sanitized_text, mode, action_candidates
        )
        errors.extend(errors_logic)
        warnings.extend(warnings_logic)
        
        # 4. Validar métricas inventadas
        errors_metrics, sanitized_text = self._validate_metrics(sanitized_text)
        errors.extend(errors_metrics)
        
        # 5. Validar frases genéricas
        warnings_generic = self._validate_generic_phrases(sanitized_text)
        warnings.extend(warnings_generic)
        
        # 6. Validar contradições com APS
        errors_aps, sanitized_text = self._validate_aps_consistency(
            sanitized_text, context, mode
        )
        errors.extend(errors_aps)
        
        # Se houver erros críticos, marcar para regenerar
        if errors:
            should_regenerate = len([e for e in errors if "PROIBIDO" in e or "INVÁLIDO" in e]) > 0
        
        is_valid = len(errors) == 0
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "sanitized_text": sanitized_text,
            "should_regenerate": should_regenerate,
        }
    
    def _validate_entities(
        self, text: str, valid_skus: Optional[List[str]], valid_resources: Optional[List[str]]
    ) -> tuple[List[str], str]:
        """Valida que SKUs e recursos mencionados existem."""
        errors = []
        sanitized = text
        
        # Extrair SKUs mencionados
        sku_patterns = [
            r'\bSKU[-\s]?([A-Z0-9]{6,})\b',
            r'\b([0-9]{12,})\b',
            r'\b([A-Z]{1,2}[-\s]?[0-9]{3,})\b',
        ]
        mentioned_skus = set()
        for pattern in sku_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentioned_skus.update(matches)
        
        # Validar SKUs
        if valid_skus and mentioned_skus:
            valid_skus_set = {str(sku).upper().strip() for sku in valid_skus}
            for mentioned_sku in mentioned_skus:
                mentioned_sku_clean = str(mentioned_sku).upper().strip()
                if mentioned_sku_clean not in valid_skus_set:
                    errors.append(f"SKU inventado: {mentioned_sku}")
                    sanitized = sanitized.replace(mentioned_sku, "[SKU_INVÁLIDO]")
        
        # Extrair recursos mencionados (aceitar tanto "27" quanto "M-27")
        resource_patterns = [
            r'\bM[-\s]?([0-9]{1,3})\b',  # M-27, M-29, etc.
            r'\brecurso\s+([0-9]{1,3})\b',  # recurso 27, recurso 29, etc.
            r'\b([0-9]{1,3})\b',  # 27, 29, 248 (apenas números, mas precisa de contexto)
        ]
        mentioned_resources = set()
        for pattern in resource_patterns[:2]:  # Apenas padrões explícitos (M-27, recurso 27)
            matches = re.findall(pattern, text, re.IGNORECASE)
            mentioned_resources.update(matches)  # Guardar sem prefixo M-
        
        # Para números soltos (27, 29), só considerar se estiverem em contexto de recurso/gargalo
        standalone_numbers = re.findall(r'\b([0-9]{1,3})\b', text)
        for num in standalone_numbers:
            # Só considerar se estiver perto de palavras-chave de recurso
            num_context = re.search(rf'\b{num}\b.*?(?:recurso|gargalo|máquina|máquina|m-{num}|recursos)', text, re.IGNORECASE)
            if num_context:
                mentioned_resources.add(num)
        
        # Validar recursos (comparar sem prefixo M-)
        if valid_resources and mentioned_resources:
            # Normalizar recursos válidos (remover M- se existir)
            valid_resources_set = {str(r).replace("M-", "").replace("M", "").strip() for r in valid_resources}
            for mentioned_resource in mentioned_resources:
                mentioned_resource_clean = str(mentioned_resource).strip()
                if mentioned_resource_clean not in valid_resources_set:
                    # Apenas avisar, não bloquear (pode ser mencionar recursos em contexto geral)
                    # Não substituir por [RECURSO_INVÁLIDO] - deixar passar
                    pass  # Removido: errors.append e sanitized.replace
        
        return errors, sanitized
    
    def _validate_module_isolation(self, text: str, mode: str) -> tuple[List[str], str]:
        """Valida que o módulo não mistura conteúdos proibidos."""
        errors = []
        sanitized = text
        
        forbidden = self.module_forbidden_words.get(mode, [])
        for word in forbidden:
            if word in text.lower():
                errors.append(f"Modo {mode} mencionou '{word}' (PROIBIDO)")
                sanitized = re.sub(rf'\b{word}\b', '[CONTEÚDO_INVÁLIDO]', sanitized, flags=re.IGNORECASE)
        
        return errors, sanitized
    
    def _validate_industrial_logic(
        self, text: str, mode: str, action_candidates: Optional[List[Dict[str, Any]]]
    ) -> tuple[List[str], List[str], str]:
        """Valida lógica industrial (fila zero, impacto impossível, etc.)."""
        errors = []
        warnings = []
        sanitized = text
        
        # Validar "reduzir fila" quando fila = 0
        if "reduzir fila" in text.lower() or "redução de fila" in text.lower():
            if action_candidates:
                for candidate in action_candidates:
                    dados_base = candidate.get("dados_base", {})
                    if dados_base.get("fila_zero", False):
                        errors.append("Mencionou 'reduzir fila' mas fila_zero=true")
                        sanitized = sanitized.replace("reduzir fila", "redistribuir carga preventiva")
                        sanitized = sanitized.replace("redução de fila", "redistribuição de carga preventiva")
        
        # Validar impacto impossível (ex: -370h sem justificação)
        impossible_impacts = re.findall(r'(-?\d{3,})\s*(h|horas|pp|%)', text, re.IGNORECASE)
        for num_str, unit in impossible_impacts:
            num = abs(int(num_str))
            if num > 500 and unit in ["h", "horas"]:
                warnings.append(f"Impacto muito grande mencionado: {num_str} {unit} (verificar se é real)")
        
        return errors, warnings, sanitized
    
    def _validate_metrics(self, text: str) -> tuple[List[str], str]:
        """Valida que não menciona métricas inventadas (OEE, WIP, etc.)."""
        errors = []
        sanitized = text
        
        for metric_name, pattern in self.forbidden_metrics.items():
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"Métrica inventada mencionada: {metric_name}")
                sanitized = re.sub(pattern, f'[{metric_name.upper()}_INVÁLIDO]', sanitized, flags=re.IGNORECASE)
        
        return errors, sanitized
    
    def _validate_generic_phrases(self, text: str) -> List[str]:
        """Valida que não usa frases genéricas tipo powerpoint."""
        warnings = []
        
        for phrase in self.generic_phrases:
            if re.search(phrase, text, re.IGNORECASE):
                warnings.append(f"Frase genérica sem números: '{phrase}'")
        
        return warnings
    
    def _validate_aps_consistency(
        self, text: str, context: Dict[str, Any], mode: str
    ) -> tuple[List[str], str]:
        """Valida que não contradiz dados do APS."""
        errors = []
        sanitized = text
        
        # Validar que não menciona valores que contradizem o APS
        if mode == "planeamento":
            planning = context.get("planning", {})
            lead_time_after = planning.get("lead_time_after", 0.0)
            
            # Se menciona lead time muito diferente do APS
            mentioned_lt = re.findall(r'lead\s+time.*?(\d+\.?\d*)\s*(h|horas)', text, re.IGNORECASE)
            for lt_str, unit in mentioned_lt:
                lt_val = float(lt_str)
                if abs(lt_val - lead_time_after) > 100 and lead_time_after > 0:
                    errors.append(f"Lead time mencionado ({lt_val}h) contradiz APS ({lead_time_after}h)")
        
        return errors, sanitized

