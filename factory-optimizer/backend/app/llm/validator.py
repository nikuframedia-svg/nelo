"""Validador de output do LLM para prevenir alucinações e invenções."""

import re
from typing import Any, Dict, List, Optional, Set


def extract_skus_from_text(text: str) -> Set[str]:
    """Extrai possíveis SKUs mencionados no texto."""
    # Padrões comuns: SKU-123, 082100100160000000, P-12, etc.
    patterns = [
        r'\bSKU[-\s]?([A-Z0-9]{6,})\b',
        r'\b([0-9]{12,})\b',  # SKUs numéricos longos
        r'\b([A-Z]{1,2}[-\s]?[0-9]{3,})\b',  # P-12, SKU-123
    ]
    skus = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        skus.update(matches)
    return skus


def extract_resources_from_text(text: str) -> Set[str]:
    """Extrai possíveis recursos mencionados no texto."""
    # Padrões: M-16, M-12, recurso 16, etc.
    patterns = [
        r'\bM[-\s]?([0-9]{1,3})\b',
        r'\brecurso\s+([0-9]{1,3})\b',
        r'\b([A-Z][-\s]?[0-9]{1,3})\b',
    ]
    resources = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        resources.update([f"M-{m}" if not m.startswith('M') else m for m in matches])
    return resources


def validate_llm_output(
    text: str,
    context: Dict,
    mode: str,
    valid_skus: Optional[List[str]] = None,
    valid_resources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Valida output do LLM para prevenir alucinações.
    
    Returns:
        {
            "is_valid": bool,
            "errors": List[str],
            "warnings": List[str],
            "sanitized_text": str
        }
    """
    errors: List[str] = []
    warnings: List[str] = []
    sanitized_text = text
    should_regenerate = False

    # Extrair SKUs e recursos mencionados
    mentioned_skus = extract_skus_from_text(text)
    mentioned_resources = extract_resources_from_text(text)

    # Validar SKUs
    if valid_skus and mentioned_skus:
        valid_skus_set = {str(sku).upper().strip() for sku in valid_skus}
        for mentioned_sku in mentioned_skus:
            mentioned_sku_clean = str(mentioned_sku).upper().strip()
            if mentioned_sku_clean not in valid_skus_set:
                errors.append(f"SKU inventado mencionado: {mentioned_sku}")
                # Remover do texto
                sanitized_text = sanitized_text.replace(mentioned_sku, f"[SKU_INVÁLIDO]")

    # Validar recursos
    if valid_resources and mentioned_resources:
        valid_resources_set = {str(r).upper().strip() for r in valid_resources}
        for mentioned_resource in mentioned_resources:
            mentioned_resource_clean = str(mentioned_resource).upper().strip()
            if mentioned_resource_clean not in valid_resources_set:
                errors.append(f"Recurso inventado mencionado: {mentioned_resource}")
                sanitized_text = sanitized_text.replace(mentioned_resource, f"[RECURSO_INVÁLIDO]")

    # VALIDAÇÃO AGRESSIVA POR MODO - Anti-alucinação industrial
    if mode == "gargalos":
        # PROIBIDO: SKUs, inventário, OTD global, lead time global
        forbidden_words = ["sku", "stock", "inventário", "inventario", "cobertura", "rop", "abc", "xyz"]
        for word in forbidden_words:
            if word in text.lower():
                errors.append(f"Modo gargalos mencionou '{word}' (PROIBIDO)")
                sanitized_text = re.sub(rf'\b{word}\b', '[CONTEÚDO_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)
        
        # Não pode mencionar OTD/lead time global (só se for para explicar impacto do gargalo)
        if re.search(r'\botd\s+[0-9]+\s*%', text, re.IGNORECASE) and "planeamento" not in text.lower():
            errors.append("Modo gargalos mencionou OTD global (PROIBIDO)")
            sanitized_text = re.sub(r'\bOTD\s+[0-9]+\s*%', '[OTD_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)

    if mode == "inventario":
        # PROIBIDO: gargalos, filas, recursos, OTD, lead time, setups
        forbidden_words = ["gargalo", "fila", "recurso m-", "utilização", "otd", "lead time", "leadtime", "setup"]
        for word in forbidden_words:
            if word in text.lower():
                errors.append(f"Modo inventario mencionou '{word}' (PROIBIDO)")
                sanitized_text = re.sub(rf'\b{word}\b', '[CONTEÚDO_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)

    if mode == "sugestoes":
        # PROIBIDO: Resumo executivo, estado geral da fábrica
        forbidden_phrases = [
            "resumo executivo",
            "estado real da fábrica",
            "a fábrica está",
            "o sistema apresenta",
            "globalmente",
            "resumo rápido",
        ]
        for phrase in forbidden_phrases:
            if phrase in text.lower():
                errors.append(f"Modo sugestoes incluiu '{phrase}' (PROIBIDO - não é resumo)")
                sanitized_text = re.sub(rf'\b{phrase}\b', '[RESUMO_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)
        
        # Deve ter pelo menos 2 ações concretas
        action_indicators = ["desviar", "repor", "colar", "aplicar", "reduzir", "aumentar", "agendar"]
        action_count = sum(1 for indicator in action_indicators if indicator in text.lower())
        if action_count < 2:
            warnings.append("Modo sugestoes deve ter pelo menos 2 ações concretas")
        
        # Validar que menciona dados_base dos ActionCandidates (não inventa números)
        action_candidates = context.get("actions", [])
        if action_candidates:
            # Verificar se menciona números que não estão nos impactos estimados
            for candidate in action_candidates:
                impacto = candidate.get("impacto_estimado", {})
                dados_base = candidate.get("dados_base", {})
                alvo = candidate.get("alvo", "")
                
                # Se menciona o alvo, verificar se os números mencionados são coerentes
                if alvo and alvo.lower() in text.lower():
                    # Extrair números mencionados perto do alvo
                    # Por agora, apenas verificar se menciona "reduzir fila" quando fila_zero=True
                    if dados_base.get("fila_zero", False):
                        if "reduzir fila" in text.lower() or "redução de fila" in text.lower():
                            errors.append(f"Mencionou 'reduzir fila' mas {alvo} tem fila_zero=true")
                            sanitized_text = sanitized_text.replace("reduzir fila", "redistribuir carga preventiva")
                            sanitized_text = sanitized_text.replace("redução de fila", "redistribuição de carga preventiva")

    if mode == "planeamento":
        # PROIBIDO: inventário, SKUs, ABC/XYZ, coberturas, stocks, ROP, compras, risco
        # ✅ PERMITIDO: gargalo, famílias, sequência, carga, capacidade, produção, ordens, operações, setups, turnos
        forbidden_words = [
            "sku", "stock", "inventário", "inventario", "cobertura", "rop", "abc", "xyz", 
            "reduzir stock", "reposicionar família", "risco 30 dias", "risco de rutura",
            "comprar", "repor", "excesso de stock", "capital imobilizado"
        ]
        for word in forbidden_words:
            if word in text.lower():
                errors.append(f"Modo planeamento mencionou '{word}' (PROIBIDO no resumo)")
                sanitized_text = re.sub(rf'\b{word}\b', '[CONTEÚDO_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)
        
        # PROIBIDO: estrutura antiga do resumo
        old_structure_patterns = [
            r'resumo\s+executivo',
            r'1️⃣\s*plano\s+antes',
            r'2️⃣\s*plano\s+depois',
            r'3️⃣\s*impacto',
            r'💰\s*Δ\s*OTD',
            r'📈\s*overlap\s+aplicado',
        ]
        for pattern in old_structure_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                errors.append(f"Modo planeamento usa estrutura antiga do resumo (PROIBIDO)")
                should_regenerate = True
        
        # PROIBIDO: placeholders genéricos (X, Y, Z, W, V, U, etc.)
        placeholder_patterns = [
            r'\b(X|Y|Z|W|V|U)\s*(h|%|horas|unidades|pp)\b',
            r'\btempo\s+de\s+ciclo\s+de\s+[X-Z]\s*h\b',
            r'\bOTD\s+de\s+[X-Z]\s*%\b',
            r'\bsetup\s+de\s+[X-Z]\s*h\b',
            r'\b[X-Z]h\b',
            r'\b[X-Z]\s*%\b',
        ]
        for pattern in placeholder_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                errors.append(f"Modo planeamento usa placeholders genéricos (PROIBIDO)")
                sanitized_text = re.sub(pattern, '[VALOR_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)
                should_regenerate = True
        
        # OBRIGATÓRIO: deve começar com "Nesta demonstração"
        if not text.strip().lower().startswith("nesta demonstração"):
            errors.append("Modo planeamento não começa com 'Nesta demonstração' (OBRIGATÓRIO)")
            should_regenerate = True
        
        # PROIBIDO: "tempo de ciclo em horas" se não vier do contexto
        if re.search(r'tempo\s+de\s+ciclo\s+em\s+horas', text, re.IGNORECASE):
            # Verificar se cycle_time está no contexto
            has_cycle_time = any(
                "cycle_time" in str(v).lower() or "tempo_ciclo" in str(v).lower()
                for v in context.values()
            )
            if not has_cycle_time:
                errors.append("Modo planeamento mencionou 'tempo de ciclo em horas' sem estar no contexto")
                sanitized_text = re.sub(r'tempo\s+de\s+ciclo\s+em\s+horas', '[TEMPO_CICLO_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)
        
        # Validar recursos mencionados (permitir recursos numéricos reais)
        # Extrair todos os recursos mencionados (formato: "27", "29", "248", "M-27", "recurso 27", etc.)
        resource_mentions = re.findall(r'\b(?:recurso\s+)?([0-9]{1,3})\b', text, re.IGNORECASE)
        resource_mentions_m = re.findall(r'\bM[-\s]?([0-9]{1,3})\b', text, re.IGNORECASE)
        all_mentioned_resources = set(resource_mentions + resource_mentions_m)
        
        # Validar contra lista de recursos válidos
        if valid_resources and all_mentioned_resources:
            valid_resources_set = {str(r).replace("M-", "").replace("M", "").strip() for r in valid_resources}
            for mentioned_resource in all_mentioned_resources:
                mentioned_resource_clean = str(mentioned_resource).strip()
                if mentioned_resource_clean not in valid_resources_set:
                    # Apenas avisar, não bloquear (pode ser mencionar recursos em contexto geral)
                    warnings.append(f"Recurso mencionado ({mentioned_resource}) não encontrado na lista de recursos válidos")
        
        # Validar que o gargalo mencionado bate com gargalo_ativo do contexto (apenas aviso)
        gargalo_mentions = re.findall(r'\bgargalo.*?(?:recurso\s+)?([0-9]+|M[-\s]?[0-9]+)', text, re.IGNORECASE)
        gargalo_ativo = context.get("gargalo_principal", {}).get("id", "")
        if gargalo_mentions and gargalo_ativo:
            # Normalizar gargalo_ativo (pode ser "M-27", "27", etc.)
            gargalo_ativo_clean = str(gargalo_ativo).replace("M-", "").replace("M", "").strip()
            for mention in gargalo_mentions:
                mention_clean = str(mention).replace("M-", "").replace("M", "").strip()
                # Se mencionar um gargalo diferente do ativo, avisar (mas não bloquear)
                if mention_clean != gargalo_ativo_clean:
                    warnings.append(f"Gargalo mencionado ({mention}) não bate com gargalo_ativo ({gargalo_ativo})")
    
    # Verificar se menciona "reduzir fila" quando fila é zero (apenas para sugestões)
    if mode == "sugestoes" and ("reduzir fila" in text.lower() or "redução de fila" in text.lower()):
        # Verificar nos action_candidates do contexto
        action_candidates = context.get("actions", [])
        for candidate in action_candidates:
            if candidate.get("tipo") == "desvio_carga":
                motivacao = candidate.get("motivacao", {})
                fila_zero = motivacao.get("fila_zero", False)
                fila_h = motivacao.get("fila_h", 0.0)
                alvo = candidate.get("alvo", "")
                
                # Se fila é zero e o recurso é mencionado no texto
                if (fila_zero or fila_h == 0.0) and alvo and alvo.lower() in text.lower():
                    errors.append(f"Mencionou 'reduzir fila' mas recurso {alvo} tem fila zero (fila_h=0)")
                    # Remover do texto
                    sanitized_text = sanitized_text.replace("reduzir fila", "redistribuir carga preventiva")
                    sanitized_text = sanitized_text.replace("redução de fila", "redistribuição de carga preventiva")
                    sanitized_text = sanitized_text.replace("reduz a fila", "redistribui carga preventiva")
    
    # Verificar se sugere desviar carga sem alternativa (apenas para sugestões)
    if mode == "sugestoes" and ("desviar" in text.lower() or "desvio" in text.lower()):
        action_candidates = context.get("actions", [])
        for candidate in action_candidates:
            if candidate.get("tipo") == "desvio_carga":
                alvo = candidate.get("alvo", "")
                alternativa = candidate.get("alternativa")
                # Se recurso é mencionado mas não tem alternativa no candidato
                if alvo and alvo.lower() in text.lower() and not alternativa:
                    warnings.append(f"Sugeriu desviar carga de {alvo} mas não há alternativa disponível no candidato")

    # Validação adicional para planeamento (lead_time_after_inconsistent)
    if mode == "planeamento":
        if context.get("kpis", {}).get("lead_time_after_inconsistent", False):
            if "783" in text or "inconsistente" not in text.lower():
                warnings.append("Planeamento deve mencionar inconsistência do lead_time_after")

    # Validar que não menciona "fila zero" quando há fila real
    if mode in ["gargalos", "planeamento"]:
        fila_real = 0.0
        if mode == "gargalos":
            recursos = context.get("recursos", [])
            if recursos:
                fila_real = max((r.get("fila_h", 0.0) for r in recursos), default=0.0)
        elif mode == "planeamento":
            fila_real = context.get("gargalo_principal", {}).get("fila_h", 0.0)
        
        if fila_real > 0 and "fila zero" in text.lower():
            errors.append(f"Mencionou 'fila zero' mas fila real é {fila_real}h")

    # Validar que não inventa números absurdos
    # Procurar por padrões como "783h", "2000 unidades", "12.500 unidades/mês", "OEE 92%", etc.
    absurd_numbers = re.findall(r'\b(7[0-9]{2,}|[8-9][0-9]{2,}|[0-9]{4,})\s*(h|horas|unidades|units|unidade)\b', text, re.IGNORECASE)
    if absurd_numbers and mode != "planeamento":  # Planeamento pode mencionar se for inconsistente
        for num, unit in absurd_numbers:
            if int(num) > 500:  # Números muito altos são suspeitos
                errors.append(f"Número inventado mencionado: {num} {unit} (não existe no contexto)")
                # Remover do texto
                sanitized_text = sanitized_text.replace(f"{num} {unit}", f"[NÚMERO_INVÁLIDO] {unit}")
    
    # Validar OEE mencionado (não existe no contexto)
    if re.search(r'\bOEE\s+(\d+%?|\d+\.\d+%?)\b', text, re.IGNORECASE):
        errors.append("OEE mencionado mas não existe no contexto")
        sanitized_text = re.sub(r'\bOEE\s+(\d+%?|\d+\.\d+%?)\b', '[OEE_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)
    
    # Validar WIP mencionado (não existe no contexto)
    if re.search(r'\bWIP\s+(\d+|\d+\.\d+)\b', text, re.IGNORECASE):
        errors.append("WIP mencionado mas não existe no contexto")
        sanitized_text = re.sub(r'\bWIP\s+(\d+|\d+\.\d+)\b', '[WIP_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)
    
    # Validar números com vírgula/ponto (ex: 12.500 unidades/mês)
    large_numbers = re.findall(r'\b(\d{1,3}[.,]\d{3,}|\d{4,})\s*(unidades|units|unidade)/?(mês|mes|month)?\b', text, re.IGNORECASE)
    if large_numbers:
        for num_str, unit, period in large_numbers:
            # Verificar se este número existe no contexto
            num_clean = num_str.replace(',', '').replace('.', '')
            if len(num_clean) >= 4:  # Números grandes são suspeitos
                errors.append(f"Número inventado mencionado: {num_str} {unit} (não existe no contexto)")
                sanitized_text = sanitized_text.replace(f"{num_str} {unit}", f"[NÚMERO_INVÁLIDO] {unit}")
    
    # Validar mistura de idiomas (inglês/PT)
    english_phrases = [
        r'\bThe factory\b',
        r'\bis operating\b',
        r'\bhas been\b',
        r'\bshould be\b',
        r'\bcan be\b',
    ]
    for pattern in english_phrases:
        if re.search(pattern, text, re.IGNORECASE):
            warnings.append("Texto em inglês misturado com português")
            # Substituir por português
            sanitized_text = re.sub(r'\bThe factory\b', 'A fábrica', sanitized_text, flags=re.IGNORECASE)
            sanitized_text = re.sub(r'\bis operating\b', 'está a operar', sanitized_text, flags=re.IGNORECASE)
            sanitized_text = re.sub(r'\bhas been\b', 'tem sido', sanitized_text, flags=re.IGNORECASE)
            sanitized_text = re.sub(r'\bshould be\b', 'deve ser', sanitized_text, flags=re.IGNORECASE)
            sanitized_text = re.sub(r'\bcan be\b', 'pode ser', sanitized_text, flags=re.IGNORECASE)
    
    # Validar frases genéricas sem números
    generic_phrases = [
        r'redução significativa',
        r'melhoria substancial',
        r'otimização',
        r'melhoria considerável',
        r'aumento significativo',
    ]
    for phrase in generic_phrases:
        if re.search(phrase, text, re.IGNORECASE):
            warnings.append(f"Frase genérica sem números: '{phrase}'")
            # Tentar encontrar número próximo para substituir
            # Se não encontrar, marcar como problema
    
    # Validar utilização > 100% mencionada (deve ser normalizada)
    if re.search(r'utiliza[çc][ãa]o.*[2-9]\d{2,}%', text, re.IGNORECASE):
        errors.append("Utilização > 100% mencionada sem normalização")
        # Normalizar para "saturação acima de X%"
        sanitized_text = re.sub(
            r'utiliza[çc][ãa]o.*?(\d{3,})%',
            r'utilização saturada (>\1%)',
            sanitized_text,
            flags=re.IGNORECASE
        )
    
    # Validar repetição de texto entre módulos
    # Se o texto contém "OTD baixo" e estamos em modo inventario/gargalos, é erro
    if mode == "inventario" and ("otd" in text.lower() or "lead time" in text.lower()):
        errors.append("Modo inventario mencionou OTD/lead time (proibido)")
        sanitized_text = re.sub(r'OTD|lead time|leadtime', '[KPI_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)
    
    if mode == "gargalos" and ("sku" in text.lower() or "stock" in text.lower() or "inventário" in text.lower()):
        errors.append("Modo gargalos mencionou inventário (proibido)")
        sanitized_text = re.sub(r'SKU|stock|inventário|inventario|cobertura|ROP', '[INVENTÁRIO_INVÁLIDO]', sanitized_text, flags=re.IGNORECASE)

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "sanitized_text": sanitized_text if errors else text,
    }

