import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import joblib
from pathlib import Path

class InventoryPredictor:
    """
    Predição de procura intermitente usando Croston-SBA/TSB
    Fallback: Poisson-Gamma
    """
    def __init__(self, model_path: Optional[str] = None):
        self.demand_history = {}  # {SKU: [demands]}
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models"
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    def update_demand(self, sku: str, demand: float, date: Optional[pd.Timestamp] = None):
        """Atualiza histórico de procura"""
        if sku not in self.demand_history:
            self.demand_history[sku] = []
        self.demand_history[sku].append(demand)
        
        # Manter apenas últimos 365 dias
        if len(self.demand_history[sku]) > 365:
            self.demand_history[sku] = self.demand_history[sku][-365:]
    
    def predict_demand(self, sku: str, method: str = "croston") -> Tuple[float, float]:
        """
        Prediz demanda média (μ) e desvio padrão (σ)
        Retorna (mu, sigma) em unidades/dia
        """
        if sku not in self.demand_history or len(self.demand_history[sku]) == 0:
            # Fallback: usar valores padrão
            return (10.0, 5.0)
        
        demands = np.array(self.demand_history[sku])
        
        if method == "croston":
            return self._croston_sba(demands)
        elif method == "tsb":
            return self._tsb(demands)
        else:
            # Poisson-Gamma
            return self._poisson_gamma(demands)
    
    def _croston_sba(self, demands: np.ndarray) -> Tuple[float, float]:
        """Croston com Smoothing Bias Adjustment"""
        # Filtrar zeros
        non_zero = demands[demands > 0]
        if len(non_zero) == 0:
            return (0.0, 0.0)
        
        # Intervalo entre demandas não-zero
        intervals = []
        last_idx = -1
        for i, d in enumerate(demands):
            if d > 0:
                if last_idx >= 0:
                    intervals.append(i - last_idx)
                last_idx = i
        
        if len(intervals) == 0:
            intervals = [1]
        
        # Média de demanda não-zero
        avg_demand = np.mean(non_zero)
        # Média de intervalo
        avg_interval = np.mean(intervals) if intervals else 1.0
        
        # Demanda média diária
        mu = avg_demand / avg_interval
        
        # Desvio padrão (simplificado)
        sigma = np.std(non_zero) / avg_interval if len(non_zero) > 1 else mu * 0.5
        
        return (max(mu, 0.1), max(sigma, 0.1))
    
    def _tsb(self, demands: np.ndarray) -> Tuple[float, float]:
        """Teunter-Syntetos-Babai"""
        # Similar ao Croston mas com ajuste diferente
        return self._croston_sba(demands)  # Simplificado
    
    def _poisson_gamma(self, demands: np.ndarray) -> Tuple[float, float]:
        """Poisson-Gamma para demanda intermitente"""
        non_zero = demands[demands > 0]
        if len(non_zero) == 0:
            return (0.1, 0.1)
        
        # Estimativa de parâmetros Gamma
        mu = np.mean(non_zero)
        var = np.var(non_zero)
        
        # Parâmetros Gamma (alpha, beta)
        if var > 0:
            alpha = (mu ** 2) / var
            beta = mu / var
        else:
            alpha = 1.0
            beta = 1.0
        
        # Média e desvio
        mean_demand = alpha / beta if beta > 0 else mu
        std_demand = np.sqrt(alpha) / beta if beta > 0 else np.sqrt(mu)
        
        return (max(mean_demand, 0.1), max(std_demand, 0.1))
    
    def calculate_rop(self, sku: str, lead_time: float, service_level: float = 0.95,
                     method: str = "croston") -> Dict[str, float]:
        """
        Calcula ROP (Reorder Point) usando simulação Monte Carlo
        Retorna: ROP, probabilidade de stockout, cobertura esperada
        """
        mu, sigma = self.predict_demand(sku, method)
        
        # Demanda durante lead time
        mu_lt = mu * lead_time
        sigma_lt = sigma * np.sqrt(lead_time)
        
        # ROP usando distribuição normal
        z = stats.norm.ppf(service_level)
        rop = mu_lt + z * sigma_lt
        
        # Simulação Monte Carlo para validar (reduzido para performance)
        n_simulations = 1000  # Reduzido de 10000 para 1000 para melhorar performance
        lt_demands = np.random.normal(mu_lt, sigma_lt, n_simulations)
        lt_demands = np.maximum(lt_demands, 0)  # Não pode ser negativo
        
        # Probabilidade de stockout
        stockout_prob = np.mean(lt_demands > rop)
        
        # Cobertura esperada (dias)
        coverage = rop / mu if mu > 0 else 0
        
        return {
            "rop": round(rop, 1),
            "stockout_prob": round(stockout_prob, 3),
            "coverage_days": round(coverage, 1),
            "mu": round(mu, 2),
            "sigma": round(sigma, 2),
            "mu_lt": round(mu_lt, 2),
            "sigma_lt": round(sigma_lt, 2)
        }

