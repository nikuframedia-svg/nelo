from typing import Dict, List, Optional
import os
from openai import OpenAI

class ExplanationGenerator:
    def __init__(self):
        self.client = None
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
            except:
                pass
    
    def generate_explanations(self, kpis_antes: Dict[str, float], 
                            kpis_depois: Dict[str, float],
                            decisoes: List[str],
                            contexto: Optional[Dict] = None) -> List[str]:
        """
        Gera explicações usando LLM ou fallback heurístico
        """
        if self.client:
            return self._generate_llm_explanations(kpis_antes, kpis_depois, decisoes, contexto)
        else:
            return self._generate_heuristic_explanations(kpis_antes, kpis_depois, decisoes)
    
    def _generate_llm_explanations(self, kpis_antes: Dict, kpis_depois: Dict,
                                  decisoes: List[str], contexto: Optional[Dict]) -> List[str]:
        """Gera explicações usando OpenAI"""
        prompt = f"""
Analisa as seguintes métricas de uma fábrica e gera 3-6 frases curtas explicando as decisões tomadas.

KPIs Antes:
- OTD: {kpis_antes.get('otd_pct', 0)}%
- Lead time: {kpis_antes.get('lead_time_h', 0)}h
- Gargalo: {kpis_antes.get('gargalo_ativo', 'N/A')}
- Setup: {kpis_antes.get('horas_setup_semana', 0)}h/semana

KPIs Depois:
- OTD: {kpis_depois.get('otd_pct', 0)}%
- Lead time: {kpis_depois.get('lead_time_h', 0)}h
- Gargalo: {kpis_depois.get('gargalo_ativo', 'N/A')}
- Setup: {kpis_depois.get('horas_setup_semana', 0)}h/semana

Decisões aplicadas: {', '.join(decisoes)}

Gera 3-6 frases curtas, cada uma começando com um verbo de ação ("Aplicámos", "Desviámos", "Reduzimos", etc.), sempre com números concretos. Foca em trade-offs e impactos mensuráveis.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "És um analista de produção que explica decisões de forma clara e numérica."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            explanations = response.choices[0].message.content.strip().split('\n')
            explanations = [e.strip('- ').strip() for e in explanations if e.strip()]
            return explanations[:6]
        except:
            return self._generate_heuristic_explanations(kpis_antes, kpis_depois, decisoes)
    
    def _generate_heuristic_explanations(self, kpis_antes: Dict, kpis_depois: Dict,
                                       decisoes: List[str]) -> List[str]:
        """Gera explicações heurísticas (fallback)"""
        explicacoes = []
        
        # Comparar KPIs
        otd_diff = kpis_depois.get('otd_pct', 0) - kpis_antes.get('otd_pct', 0)
        if otd_diff > 0:
            explicacoes.append(f"Melhorámos OTD em {otd_diff:.1f}pp → {kpis_depois.get('otd_pct', 0):.1f}% de ordens no prazo")
        
        lt_diff = kpis_antes.get('lead_time_h', 0) - kpis_depois.get('lead_time_h', 0)
        if lt_diff > 0:
            explicacoes.append(f"Reduzimos lead time em {lt_diff:.1f}h → {kpis_depois.get('lead_time_h', 0):.1f}h médio")
        
        setup_diff = kpis_antes.get('horas_setup_semana', 0) - kpis_depois.get('horas_setup_semana', 0)
        if setup_diff > 0:
            explicacoes.append(f"Poupámos {setup_diff:.1f}h de setup por semana → colagem de famílias")
        
        # Decisões
        if any("overlap" in d.lower() for d in decisoes):
            explicacoes.append("Aplicámos overlap entre operações → paralelização de atividades")
        
        if any("rota" in d.lower() or "alternativa" in d.lower() for d in decisoes):
            explicacoes.append("Utilizámos rotas alternativas → descongestionamento de gargalos")
        
        if not explicacoes:
            explicacoes.append("Plano otimizado com melhorias em eficiência e cumprimento de prazos")
        
        return explicacoes[:6]

