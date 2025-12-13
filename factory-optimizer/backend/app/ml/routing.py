import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path

class RoutingBandit:
    """
    Bandit Contextual para escolher rotas alternativas
    """
    def __init__(self, model_path: Optional[str] = None):
        self.rewards = {}  # {context: {route: [rewards]}}
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models"
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Inicializa com valores padrão"""
        # Contextos: (recurso, operacao)
        contexts = [
            ("M-01", "Transformação"),
            ("M-02", "Transformação"),
            ("M-05", "Acabamentos"),
            ("M-06", "Acabamentos"),
        ]
        
        for context in contexts:
            key = f"{context[0]}_{context[1]}"
            self.rewards[key] = {
                "A": [100, 95, 105, 98, 102],  # Histórico de recompensas
                "B": [110, 108, 112, 105, 109]  # Rota B geralmente melhor
            }
        
        try:
            joblib.dump(self.rewards, self.model_path / "routing_bandit.pkl")
        except:
            pass
    
    def choose_route(self, recurso: str, operacao: str, 
                    contexto: Optional[Dict] = None) -> str:
        """
        Escolhe rota usando Upper Confidence Bound (UCB)
        Retorna "A" ou "B"
        """
        key = f"{recurso}_{operacao}"
        
        if key not in self.rewards:
            # Novo contexto: usar rota A por padrão
            self.rewards[key] = {"A": [100], "B": [100]}
            return "A"
        
        route_rewards = self.rewards[key]
        
        # UCB: escolher rota com maior upper bound
        ucb_scores = {}
        total_pulls = sum(len(rewards) for rewards in route_rewards.values())
        
        for route, rewards in route_rewards.items():
            if len(rewards) == 0:
                ucb_scores[route] = float('inf')
                continue
            
            avg_reward = np.mean(rewards)
            confidence = np.sqrt(2 * np.log(total_pulls + 1) / len(rewards))
            ucb_scores[route] = avg_reward + confidence
        
        # Escolher rota com maior UCB
        best_route = max(ucb_scores.items(), key=lambda x: x[1])[0]
        return best_route
    
    def update_reward(self, recurso: str, operacao: str, route: str, 
                     lead_time: float, retrabalho: float = 0):
        """
        Atualiza recompensa após execução
        Recompensa = -lead_time - retrabalho*10
        """
        key = f"{recurso}_{operacao}"
        
        if key not in self.rewards:
            self.rewards[key] = {"A": [], "B": []}
        
        if route not in self.rewards[key]:
            self.rewards[key][route] = []
        
        reward = -lead_time - retrabalho * 10
        self.rewards[key][route].append(reward)
        
        # Manter apenas últimas 100 recompensas
        if len(self.rewards[key][route]) > 100:
            self.rewards[key][route] = self.rewards[key][route][-100:]
        
        # Salvar
        try:
            joblib.dump(self.rewards, self.model_path / "routing_bandit.pkl")
        except:
            pass  # Ignorar erro de escrita
    
    def get_regret(self) -> float:
        """Calcula regret acumulado"""
        total_regret = 0
        
        for key, route_rewards in self.rewards.items():
            if not route_rewards:
                continue
            
            best_avg = max(np.mean(rewards) for rewards in route_rewards.values() if rewards)
            for route, rewards in route_rewards.items():
                if rewards:
                    avg_reward = np.mean(rewards)
                    total_regret += max(0, best_avg - avg_reward) * len(rewards)
        
        return total_regret

