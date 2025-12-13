import pandas as pd
import numpy as np
from typing import Dict, Optional
import joblib
from pathlib import Path

class SetupTimePredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.setup_times = {}  # Cache de setups por família
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models"
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Inicializa com valores padrão baseados em família"""
        # Setup times por família (minutos)
        default_setups = {
            "ABR": 30,  # Abrasivos
            "MET": 45,  # Metais
            "PLA": 20,  # Plásticos
            "TEX": 35,  # Têxteis
            "DEF": 25   # Default
        }
        self.setup_times = default_setups.copy()
        
        # Salvar
        joblib.dump(self.setup_times, self.model_path / "setup_times.pkl")
    
    def predict(self, familia_anterior: str, familia_atual: str, 
                recurso: str, **kwargs) -> float:
        """
        Prediz tempo de setup entre famílias
        Retorna minutos
        """
        # Se mesma família, setup reduzido
        if familia_anterior == familia_atual or not familia_anterior:
            return 0.0
        
        # Identificar família atual
        familia = self._extract_family(familia_atual)
        
        # Tempo base da família
        base_time = self.setup_times.get(familia, self.setup_times["DEF"])
        
        # Variação baseada no recurso (alguns recursos têm setup mais rápido)
        if "M-01" in recurso or "M-02" in recurso:
            base_time *= 0.9  # 10% mais rápido
        elif "M-05" in recurso or "M-06" in recurso:
            base_time *= 1.1  # 10% mais lento
        
        # Adicionar incerteza (distribuição normal)
        uncertainty = np.random.normal(0, base_time * 0.1)  # 10% de variabilidade
        setup_time = max(base_time + uncertainty, 5.0)  # Mínimo 5 minutos
        
        return round(setup_time, 1)
    
    def _extract_family(self, sku_or_family: str) -> str:
        """Extrai família do SKU"""
        if "-" in sku_or_family:
            return sku_or_family.split("-")[0]
        elif len(sku_or_family) >= 3:
            return sku_or_family[:3]
        else:
            return "DEF"
    
    def update_setup_time(self, familia: str, setup_time: float):
        """Atualiza tempo de setup para uma família"""
        self.setup_times[familia] = setup_time
        joblib.dump(self.setup_times, self.model_path / "setup_times.pkl")
    
    def get_setup_times(self) -> Dict[str, float]:
        """Retorna todos os tempos de setup"""
        return self.setup_times.copy()

