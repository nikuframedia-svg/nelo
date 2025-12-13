import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from typing import Dict, List, Optional, Tuple, Any
import joblib
import json
from pathlib import Path

class BottleneckPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_columns = None
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models"
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._load_or_train()
    
    def _load_or_train(self):
        """Carrega modelo existente ou treina novo"""
        try:
            self.model = joblib.load(self.model_path / "bottleneck.pkl")
            self.feature_columns = joblib.load(self.model_path / "bottleneck_features.pkl")
        except:
            self._train_default()
    
    def _train_default(self):
        """Treina modelo padrão"""
        n_samples = 500
        data = {
            "utilizacao_prevista": np.random.uniform(50, 120, n_samples),
            "num_setups": np.random.randint(5, 30, n_samples),
            "staffing": np.random.randint(5, 20, n_samples),
            "indisponibilidades": np.random.uniform(0, 10, n_samples),
            "mix_abrasivos": np.random.uniform(0, 1, n_samples),
            "fila_atual": np.random.uniform(0, 100, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Target: gargalo (1 se utilização > 90% ou fila > 50)
        df["gargalo"] = ((df["utilizacao_prevista"] > 90) | 
                        (df["fila_atual"] > 50)).astype(int)
        
        X = df[["utilizacao_prevista", "num_setups", "staffing", 
                "indisponibilidades", "mix_abrasivos", "fila_atual"]]
        y = df["gargalo"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        self.feature_columns = X.columns.tolist()
        
        joblib.dump(self.model, self.model_path / "bottleneck.pkl")
        joblib.dump(self.feature_columns, self.model_path / "bottleneck_features.pkl")
    
    def predict_probability(self, utilizacao: float, num_setups: int, 
                           staffing: int, indisponibilidades: float = 0,
                           mix_abrasivos: float = 0.5, fila_atual: float = 0) -> float:
        """Prediz probabilidade de gargalo"""
        if self.model is None:
            # Fallback heurístico
            return 1.0 if utilizacao > 90 or fila_atual > 50 else 0.0
        
        features = np.array([[
            utilizacao, num_setups, staffing, 
            indisponibilidades, mix_abrasivos, fila_atual
        ]])
        
        try:
            prob = self.model.predict_proba(features)[0][1]
            return round(prob, 3)
        except:
            return 1.0 if utilizacao > 90 else 0.0
    
    def predict_bottleneck_probability(self, utilizacao_pct: float, queue_hours: float = 0.0,
                                      num_setups: int = 0, staffing: int = 10,
                                      indisponibilidades: float = 0.0, mix_abrasivos: float = 0.5) -> float:
        """Alias para predict_probability com nome mais descritivo"""
        return self.predict_probability(
            utilizacao=utilizacao_pct * 100,  # Convert to percentage
            num_setups=num_setups,
            staffing=staffing,
            indisponibilidades=indisponibilidades,
            mix_abrasivos=mix_abrasivos,
            fila_atual=queue_hours
        )
    
    def get_bottleneck_drivers(self, utilizacao: float, num_setups: int,
                              staffing: int, **kwargs) -> List[str]:
        """Retorna drivers do gargalo"""
        drivers = []
        
        if utilizacao > 90:
            drivers.append(f"Utilização alta ({utilizacao:.1f}%)")
        if num_setups > 20:
            drivers.append(f"Muitos setups ({num_setups})")
        if staffing < 10:
            drivers.append(f"Staffing baixo ({staffing})")
        if kwargs.get("fila_atual", 0) > 50:
            drivers.append(f"Fila grande ({kwargs['fila_atual']:.1f}h)")
        
        return drivers
    
    def fit_from_etl(self, df_bottlenecks: pd.DataFrame, min_samples: int = 30) -> Dict[str, Any]:
        """
        Retreina modelo com dados reais do scheduler.
        
        Args:
            df_bottlenecks: DataFrame com colunas: utilizacao_prevista, num_setups, staffing,
                          indisponibilidades, mix_abrasivos, fila_atual, gargalo_real
            min_samples: Número mínimo de amostras para retreinar
        
        Returns:
            Dict com métricas de performance
        """
        if df_bottlenecks is None or df_bottlenecks.empty or len(df_bottlenecks) < min_samples:
            return {"status": "insufficient_data", "samples": len(df_bottlenecks) if df_bottlenecks is not None else 0}
        
        required_cols = ["utilizacao_prevista", "fila_atual", "gargalo_real"]
        missing = [col for col in required_cols if col not in df_bottlenecks.columns]
        if missing:
            return {"status": "missing_columns", "missing": missing}
        
        df = df_bottlenecks.copy()
        df = df.dropna(subset=required_cols)
        
        if len(df) < min_samples:
            return {"status": "insufficient_data_after_clean", "samples": len(df)}
        
        # Features
        feature_cols = ["utilizacao_prevista", "num_setups", "staffing", 
                        "indisponibilidades", "mix_abrasivos", "fila_atual"]
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].fillna(0)
        y = df["gargalo_real"].astype(int)
        
        # Garantir que temos classes positivas e negativas
        if y.sum() == 0 or y.sum() == len(y):
            return {"status": "imbalanced_classes", "positive_samples": int(y.sum()), "total": len(y)}
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Treinar modelo
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        # Métricas
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0
        
        cv_f1_scores = cross_val_score(self.model, X_train, y_train, cv=kfold, scoring="f1")
        
        self.feature_columns = X.columns.tolist()
        
        # Salvar modelo
        joblib.dump(self.model, self.model_path / "bottleneck.pkl")
        joblib.dump(self.feature_columns, self.model_path / "bottleneck_features.pkl")
        
        # Salvar métricas
        metrics = {
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "cv_f1_mean": float(cv_f1_scores.mean()),
            "cv_f1_std": float(cv_f1_scores.std()),
            "samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "positive_samples": int(y.sum()),
            "updated_at": pd.Timestamp.now().isoformat(),
        }
        
        with open(self.model_path / "bottleneck_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance salvas"""
        metrics_path = self.model_path / "bottleneck_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                return json.load(f)
        return {"status": "no_metrics", "using_synthetic": True}

