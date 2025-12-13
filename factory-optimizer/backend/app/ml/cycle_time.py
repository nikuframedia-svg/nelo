import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Optional, Tuple, Any
import joblib
import json
from pathlib import Path

class CycleTimePredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model_p50 = None
        self.model_p90 = None
        self.feature_columns = None
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models"
        self.model_path = Path(model_path)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self._load_or_train()
    
    def _load_or_train(self):
        """Carrega modelos existentes ou treina novos"""
        try:
            self.model_p50 = joblib.load(self.model_path / "cycle_p50.pkl")
            self.model_p90 = joblib.load(self.model_path / "cycle_p90.pkl")
            self.feature_columns = joblib.load(self.model_path / "cycle_features.pkl")
        except:
            self._train_default()
    
    def _train_default(self):
        """Treina modelo padrão com dados sintéticos"""
        # Gerar dados sintéticos para treino
        n_samples = 1000
        data = {
            "sku": [f"SKU-{i%50:03d}" for i in range(n_samples)],
            "operacao": [f"OP-{i%10}" for i in range(n_samples)],
            "recurso": [f"M-{i%20+1:02d}" for i in range(n_samples)],
            "quantidade": np.random.randint(100, 1000, n_samples),
            "turno": np.random.choice(["A", "B", "C"], n_samples),
            "pessoas": np.random.randint(1, 5, n_samples),
            "overlap": np.random.uniform(0, 0.8, n_samples),
            "backlog": np.random.uniform(0, 100, n_samples),
            "fila": np.random.uniform(0, 50, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Target: tempo de ciclo (seg/unid)
        # Simulação: base + variação
        base_cycle = 10 + df["quantidade"] * 0.01 + np.random.normal(0, 2, n_samples)
        df["cycle_time"] = np.maximum(base_cycle, 5)
        
        # Features
        df_features = pd.get_dummies(df[["sku", "operacao", "recurso", "turno"]], drop_first=True)
        df_features = pd.concat([df_features, df[["quantidade", "pessoas", "overlap", "backlog", "fila"]]], axis=1)
        
        X = df_features
        y = df["cycle_time"]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Treinar P50 (mediana)
        self.model_p50 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model_p50.fit(X_train, y_train)
        
        # Treinar P90 (quantil 90)
        self.model_p90 = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.9,
            n_estimators=100,
            random_state=42
        )
        self.model_p90.fit(X_train, y_train)
        
        self.feature_columns = X.columns.tolist()
        
        # Salvar
        joblib.dump(self.model_p50, self.model_path / "cycle_p50.pkl")
        joblib.dump(self.model_p90, self.model_path / "cycle_p90.pkl")
        joblib.dump(self.feature_columns, self.model_path / "cycle_features.pkl")
    
    def predict_p50(self, sku: str, operacao: str, recurso: str, 
                    quantidade: int, **kwargs) -> float:
        """Prediz P50 do tempo de ciclo"""
        if self.model_p50 is None:
            return 10.0  # Fallback
        
        # Criar features
        features = self._create_features(sku, operacao, recurso, quantidade, **kwargs)
        
        # Predizer
        try:
            pred = self.model_p50.predict([features])[0]
            return max(pred, 5.0)  # Mínimo 5 seg
        except:
            return 10.0
    
    def predict_p90(self, sku: str, operacao: str, recurso: str,
                    quantidade: int, **kwargs) -> float:
        """Prediz P90 do tempo de ciclo"""
        if self.model_p90 is None:
            return 15.0  # Fallback
        
        features = self._create_features(sku, operacao, recurso, quantidade, **kwargs)
        try:
            pred = self.model_p90.predict([features])[0]
            return max(pred, 5.0)
        except:
            return 15.0
    
    def _create_features(self, sku: str, operacao: str, recurso: str,
                        quantidade: int, **kwargs) -> np.ndarray:
        """Cria vetor de features"""
        # Criar dict com todas as features
        feature_dict = {col: 0 for col in self.feature_columns}
        
        # Features categóricas (one-hot)
        for col in self.feature_columns:
            if col.startswith("sku_") and f"sku_{sku}" == col:
                feature_dict[col] = 1
            elif col.startswith("operacao_") and f"operacao_{operacao}" == col:
                feature_dict[col] = 1
            elif col.startswith("recurso_") and f"recurso_{recurso}" == col:
                feature_dict[col] = 1
            elif col.startswith("turno_") and kwargs.get("turno") and f"turno_{kwargs['turno']}" == col:
                feature_dict[col] = 1
        
        # Features numéricas
        if "quantidade" in self.feature_columns:
            feature_dict["quantidade"] = quantidade
        if "pessoas" in self.feature_columns:
            feature_dict["pessoas"] = kwargs.get("pessoas", 2)
        if "overlap" in self.feature_columns:
            feature_dict["overlap"] = kwargs.get("overlap", 0.0)
        if "backlog" in self.feature_columns:
            feature_dict["backlog"] = kwargs.get("backlog", 0.0)
        if "fila" in self.feature_columns:
            feature_dict["fila"] = kwargs.get("fila", 0.0)
        
        return np.array([feature_dict.get(col, 0) for col in self.feature_columns])
    
    def fit_from_etl(self, df_execucoes: pd.DataFrame, min_samples: int = 50) -> Dict[str, float]:
        """
        Retreina modelos com dados reais do ETL.
        
        Args:
            df_execucoes: DataFrame com colunas: sku, operacao, recurso, quantidade, 
                         pessoas, overlap, backlog, fila, turno, cycle_time_real
            min_samples: Número mínimo de amostras para retreinar
        
        Returns:
            Dict com métricas de performance
        """
        if df_execucoes is None or df_execucoes.empty or len(df_execucoes) < min_samples:
            return {"status": "insufficient_data", "samples": len(df_execucoes) if df_execucoes is not None else 0}
        
        required_cols = ["sku", "operacao", "recurso", "quantidade", "cycle_time_real"]
        missing = [col for col in required_cols if col not in df_execucoes.columns]
        if missing:
            return {"status": "missing_columns", "missing": missing}
        
        df = df_execucoes.copy()
        df = df.dropna(subset=required_cols)
        
        if len(df) < min_samples:
            return {"status": "insufficient_data_after_clean", "samples": len(df)}
        
        # Features
        df_features = pd.get_dummies(df[["sku", "operacao", "recurso"]], drop_first=True)
        optional_cols = ["pessoas", "overlap", "backlog", "fila", "turno"]
        for col in optional_cols:
            if col in df.columns:
                if col == "turno":
                    df_features = pd.concat([df_features, pd.get_dummies(df[[col]], drop_first=True)], axis=1)
                else:
                    df_features = pd.concat([df_features, df[[col]]], axis=1)
        
        # Alinhar features com modelo existente
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in df_features.columns:
                    df_features[col] = 0
            df_features = df_features[self.feature_columns]
        else:
            self.feature_columns = df_features.columns.tolist()
        
        X = df_features
        y = df["cycle_time_real"]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Cross-validation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Treinar P50
        self.model_p50 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model_p50.fit(X_train, y_train)
        cv_scores_p50 = cross_val_score(self.model_p50, X_train, y_train, cv=kfold, scoring="neg_mean_absolute_error")
        mae_p50 = -cv_scores_p50.mean()
        rmse_p50 = np.sqrt(mean_squared_error(y_test, self.model_p50.predict(X_test)))
        
        # Treinar P90
        self.model_p90 = GradientBoostingRegressor(
            loss="quantile",
            alpha=0.9,
            n_estimators=100,
            random_state=42
        )
        self.model_p90.fit(X_train, y_train)
        cv_scores_p90 = cross_val_score(self.model_p90, X_train, y_train, cv=kfold, scoring="neg_mean_absolute_error")
        mae_p90 = -cv_scores_p90.mean()
        rmse_p90 = np.sqrt(mean_squared_error(y_test, self.model_p90.predict(X_test)))
        
        # Salvar modelos
        joblib.dump(self.model_p50, self.model_path / "cycle_p50.pkl")
        joblib.dump(self.model_p90, self.model_path / "cycle_p90.pkl")
        joblib.dump(self.feature_columns, self.model_path / "cycle_features.pkl")
        
        # Salvar métricas
        metrics = {
            "p50": {"mae": float(mae_p50), "rmse": float(rmse_p50), "cv_mae_mean": float(-cv_scores_p50.mean()), "cv_mae_std": float(cv_scores_p50.std())},
            "p90": {"mae": float(mae_p90), "rmse": float(rmse_p90), "cv_mae_mean": float(-cv_scores_p90.mean()), "cv_mae_std": float(cv_scores_p90.std())},
            "samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "updated_at": pd.Timestamp.now().isoformat(),
        }
        
        with open(self.model_path / "cycle_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas de performance salvas"""
        metrics_path = self.model_path / "cycle_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                return json.load(f)
        return {"status": "no_metrics", "using_synthetic": True}

