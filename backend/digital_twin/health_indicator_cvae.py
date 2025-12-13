"""
════════════════════════════════════════════════════════════════════════════════════════════════════
HEALTH INDICATOR CVAE - Conditional Variational Autoencoder for Machine Health
════════════════════════════════════════════════════════════════════════════════════════════════════

Implementa um Conditional Variational Autoencoder (CVAE) para extrair Health Indicators (HI)
específicos por operação/máquina.

Input:
- Dados de monitorização da máquina (vibração, corrente, temperatura, etc.)
- Contexto da operação (op_code, machine_id, product_type)

Output:
- Health Indicator (HI) específico por operação/máquina: valor contínuo ∈ [0,1]
  - 1.0 = máquina em perfeito estado
  - 0.0 = máquina em estado crítico (falha iminente)

Arquitetura CVAE:
- Encoder: x + c → μ, σ (latent distribution)
- Reparametrization: z = μ + σ * ε (ε ~ N(0,1))
- Decoder: z + c → x̂ (reconstruction)
- Health Indicator: extraído do espaço latente

TODO[R&D]:
- Attention mechanism para features temporais
- Multi-task learning (HI + anomaly detection)
- Adversarial training para robustez
- Transformer-based encoder para séries temporais longas
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CVAEConfig:
    """Configuração do modelo CVAE."""
    # Dimensões
    input_dim: int = 64  # Dimensão do input (sensor features)
    context_dim: int = 32  # Dimensão do contexto (embeddings)
    latent_dim: int = 16  # Dimensão do espaço latente
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    
    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    beta: float = 1.0  # KL divergence weight (β-VAE)
    
    # Embeddings
    num_machines: int = 50
    num_operations: int = 100
    num_product_types: int = 20
    embedding_dim: int = 8
    
    # Health Indicator
    hi_threshold_warning: float = 0.5
    hi_threshold_critical: float = 0.3


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SensorSnapshot:
    """Snapshot de dados de sensores de uma máquina."""
    machine_id: str
    timestamp: datetime
    
    # Sensor readings (normalizados para [0, 1] ou z-score)
    vibration_x: float = 0.0
    vibration_y: float = 0.0
    vibration_z: float = 0.0
    vibration_rms: float = 0.0
    
    current_phase_a: float = 0.0
    current_phase_b: float = 0.0
    current_phase_c: float = 0.0
    current_rms: float = 0.0
    
    temperature_bearing: float = 0.0
    temperature_motor: float = 0.0
    temperature_ambient: float = 0.0
    
    acoustic_emission: float = 0.0
    oil_particle_count: float = 0.0
    pressure: float = 0.0
    
    # Derived features
    speed_rpm: float = 0.0
    load_percent: float = 0.0
    power_factor: float = 0.0
    
    # Raw additional features (extensible)
    extra_features: Dict[str, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Converte para vetor numérico para input no modelo."""
        base_features = [
            self.vibration_x, self.vibration_y, self.vibration_z, self.vibration_rms,
            self.current_phase_a, self.current_phase_b, self.current_phase_c, self.current_rms,
            self.temperature_bearing, self.temperature_motor, self.temperature_ambient,
            self.acoustic_emission, self.oil_particle_count, self.pressure,
            self.speed_rpm, self.load_percent, self.power_factor,
        ]
        # Pad or truncate to fixed size
        extra = list(self.extra_features.values())[:47]  # 17 base + 47 extra = 64
        extra = extra + [0.0] * (47 - len(extra))
        return np.array(base_features + extra, dtype=np.float32)


@dataclass
class OperationContext:
    """Contexto da operação atual."""
    machine_id: str
    op_code: str
    product_type: str
    
    # Optional metadata
    order_id: Optional[str] = None
    article_id: Optional[str] = None
    route_id: Optional[str] = None
    
    # Runtime info
    operation_duration_min: float = 0.0
    cycle_count: int = 0
    time_since_maintenance_hours: float = 0.0
    
    def to_indices(self, config: CVAEConfig) -> Tuple[int, int, int]:
        """Converte para índices de embedding."""
        # Hash-based indices (em produção, usar lookup tables)
        machine_idx = hash(self.machine_id) % config.num_machines
        op_idx = hash(self.op_code) % config.num_operations
        product_idx = hash(self.product_type) % config.num_product_types
        return machine_idx, op_idx, product_idx


@dataclass
class HealthIndicatorResult:
    """Resultado da inferência de Health Indicator."""
    machine_id: str
    timestamp: datetime
    
    # Health Indicator
    hi: float  # ∈ [0, 1], 1 = saudável, 0 = crítico
    hi_std: float  # Incerteza do HI
    
    # Latent space
    latent_mean: np.ndarray
    latent_std: np.ndarray
    
    # Reconstruction
    reconstruction_error: float
    
    # Classification
    status: str  # "HEALTHY", "WARNING", "CRITICAL"
    
    # Context
    op_code: Optional[str] = None
    product_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário serializável."""
        return {
            "machine_id": self.machine_id,
            "timestamp": self.timestamp.isoformat(),
            "hi": round(self.hi, 4),
            "hi_std": round(self.hi_std, 4),
            "reconstruction_error": round(self.reconstruction_error, 4),
            "status": self.status,
            "op_code": self.op_code,
            "product_type": self.product_type,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CVAE MODEL (PyTorch implementation)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch não disponível. CVAE funcionará em modo simulado.")


if TORCH_AVAILABLE:
    class CVAEEncoder(nn.Module):
        """Encoder do CVAE: x + c → μ, σ"""
        
        def __init__(self, config: CVAEConfig):
            super().__init__()
            self.config = config
            
            # Context embeddings
            self.machine_embedding = nn.Embedding(config.num_machines, config.embedding_dim)
            self.op_embedding = nn.Embedding(config.num_operations, config.embedding_dim)
            self.product_embedding = nn.Embedding(config.num_product_types, config.embedding_dim)
            
            # Input dimension = sensor features + context embeddings
            total_input_dim = config.input_dim + 3 * config.embedding_dim
            
            # MLP layers
            layers = []
            prev_dim = total_input_dim
            for hidden_dim in config.hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1),
                ])
                prev_dim = hidden_dim
            
            self.encoder = nn.Sequential(*layers)
            
            # Latent parameters
            self.fc_mu = nn.Linear(config.hidden_dims[-1], config.latent_dim)
            self.fc_logvar = nn.Linear(config.hidden_dims[-1], config.latent_dim)
        
        def forward(
            self,
            x: torch.Tensor,
            machine_idx: torch.Tensor,
            op_idx: torch.Tensor,
            product_idx: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # Get embeddings
            machine_emb = self.machine_embedding(machine_idx)
            op_emb = self.op_embedding(op_idx)
            product_emb = self.product_embedding(product_idx)
            
            # Concatenate input with context
            context = torch.cat([machine_emb, op_emb, product_emb], dim=-1)
            x_with_context = torch.cat([x, context], dim=-1)
            
            # Encode
            h = self.encoder(x_with_context)
            
            # Latent parameters
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            
            return mu, logvar


    class CVAEDecoder(nn.Module):
        """Decoder do CVAE: z + c → x̂"""
        
        def __init__(self, config: CVAEConfig):
            super().__init__()
            self.config = config
            
            # Context embeddings (shared with encoder in practice, but separate here for clarity)
            self.machine_embedding = nn.Embedding(config.num_machines, config.embedding_dim)
            self.op_embedding = nn.Embedding(config.num_operations, config.embedding_dim)
            self.product_embedding = nn.Embedding(config.num_product_types, config.embedding_dim)
            
            # Input dimension = latent + context embeddings
            total_input_dim = config.latent_dim + 3 * config.embedding_dim
            
            # MLP layers (reverse of encoder)
            layers = []
            prev_dim = total_input_dim
            for hidden_dim in reversed(config.hidden_dims):
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.1),
                ])
                prev_dim = hidden_dim
            
            self.decoder = nn.Sequential(*layers)
            
            # Output layer
            self.fc_out = nn.Linear(config.hidden_dims[0], config.input_dim)
        
        def forward(
            self,
            z: torch.Tensor,
            machine_idx: torch.Tensor,
            op_idx: torch.Tensor,
            product_idx: torch.Tensor,
        ) -> torch.Tensor:
            # Get embeddings
            machine_emb = self.machine_embedding(machine_idx)
            op_emb = self.op_embedding(op_idx)
            product_emb = self.product_embedding(product_idx)
            
            # Concatenate latent with context
            context = torch.cat([machine_emb, op_emb, product_emb], dim=-1)
            z_with_context = torch.cat([z, context], dim=-1)
            
            # Decode
            h = self.decoder(z_with_context)
            x_recon = self.fc_out(h)
            
            return x_recon


    class CVAE(nn.Module):
        """
        Conditional Variational Autoencoder for Health Indicator Extraction.
        
        O CVAE aprende a reconstruir dados de sensores condicionados ao contexto
        da operação. O Health Indicator é extraído do espaço latente:
        - Baixa reconstrução error + latent próximo da média → máquina saudável
        - Alta reconstrução error + latent afastado → máquina degradada
        
        TODO[R&D]:
        - Implementar variante β-VAE para melhor disentanglement
        - Adicionar loss auxiliar para previsão de falha
        - Attention mechanism para features temporais
        """
        
        def __init__(self, config: CVAEConfig):
            super().__init__()
            self.config = config
            self.encoder = CVAEEncoder(config)
            self.decoder = CVAEDecoder(config)
            
            # Health Indicator head (extrai HI do espaço latente)
            self.hi_head = nn.Sequential(
                nn.Linear(config.latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),  # Output ∈ [0, 1]
            )
        
        def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
            """Reparametrization trick: z = μ + σ * ε"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def forward(
            self,
            x: torch.Tensor,
            machine_idx: torch.Tensor,
            op_idx: torch.Tensor,
            product_idx: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            # Encode
            mu, logvar = self.encoder(x, machine_idx, op_idx, product_idx)
            
            # Sample latent
            z = self.reparameterize(mu, logvar)
            
            # Decode
            x_recon = self.decoder(z, machine_idx, op_idx, product_idx)
            
            # Health Indicator
            hi = self.hi_head(z)
            
            return x_recon, mu, logvar, hi
        
        def loss_function(
            self,
            x: torch.Tensor,
            x_recon: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            hi: torch.Tensor,
            hi_target: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            ELBO loss = Reconstruction + β * KL Divergence + HI Supervision (optional)
            """
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(x_recon, x, reduction='mean')
            
            # KL Divergence: KL(q(z|x,c) || p(z))
            # Closed form for Gaussian: -0.5 * sum(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Total loss
            total_loss = recon_loss + self.config.beta * kl_loss
            
            # Optional HI supervision
            hi_loss = 0.0
            if hi_target is not None:
                hi_loss = F.mse_loss(hi.squeeze(), hi_target, reduction='mean')
                total_loss = total_loss + hi_loss
            
            metrics = {
                "recon_loss": recon_loss.item(),
                "kl_loss": kl_loss.item(),
                "hi_loss": hi_loss if isinstance(hi_loss, float) else hi_loss.item(),
                "total_loss": total_loss.item(),
            }
            
            return total_loss, metrics
        
        def infer_health_indicator(
            self,
            x: Any,  # torch.Tensor or np.ndarray
            machine_idx: Any,  # torch.Tensor or int
            op_idx: Any,  # torch.Tensor or int
            product_idx: Any,  # torch.Tensor or int
            num_samples: int = 10,
        ) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
            """
            Inferir Health Indicator com estimativa de incerteza via Monte Carlo sampling.
            
            Returns:
                (hi_mean, hi_std, latent_mean, latent_std, reconstruction_error)
            """
            # Convert inputs to tensors if needed
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            if not isinstance(machine_idx, torch.Tensor):
                machine_idx = torch.tensor([machine_idx], dtype=torch.long)
            if not isinstance(op_idx, torch.Tensor):
                op_idx = torch.tensor([op_idx], dtype=torch.long)
            if not isinstance(product_idx, torch.Tensor):
                product_idx = torch.tensor([product_idx], dtype=torch.long)
            
            self.eval()
            with torch.no_grad():
                # Multiple forward passes for uncertainty estimation
                his = []
                latents = []
                recon_errors = []
                
                for _ in range(num_samples):
                    x_recon, mu, logvar, hi = self.forward(x, machine_idx, op_idx, product_idx)
                    z = self.reparameterize(mu, logvar)
                    
                    his.append(hi.item())
                    latents.append(z.cpu().numpy())
                    recon_errors.append(F.mse_loss(x_recon, x).item())
                
                hi_mean = np.mean(his)
                hi_std = np.std(his)
                latent_mean = np.mean(latents, axis=0).flatten()
                latent_std = np.std(latents, axis=0).flatten()
                recon_error = np.mean(recon_errors)
                
                return hi_mean, hi_std, latent_mean, latent_std, recon_error

else:
    # Fallback sem PyTorch
    class CVAE:
        """Versão simulada do CVAE (sem PyTorch)."""
        
        def __init__(self, config: CVAEConfig):
            self.config = config
            logger.info("CVAE em modo simulado (PyTorch não disponível)")
        
        def infer_health_indicator(
            self,
            x: np.ndarray,
            machine_idx: int,
            op_idx: int,
            product_idx: int,
            num_samples: int = 10,
        ) -> Tuple[float, float, np.ndarray, np.ndarray, float]:
            """Inferência simulada baseada em heurísticas."""
            # Simular HI baseado em features de vibração e temperatura
            vibration_score = 1.0 - min(1.0, np.mean(x[:4]) * 2)  # Vibration features
            temp_score = 1.0 - min(1.0, np.mean(x[8:11]) * 1.5)  # Temperature features
            
            hi_mean = 0.3 * vibration_score + 0.3 * temp_score + 0.4 * np.random.uniform(0.6, 1.0)
            hi_mean = np.clip(hi_mean, 0.0, 1.0)
            hi_std = np.random.uniform(0.02, 0.1)
            
            latent_mean = np.random.randn(self.config.latent_dim).astype(np.float32)
            latent_std = np.abs(np.random.randn(self.config.latent_dim)).astype(np.float32) * 0.1
            recon_error = np.random.uniform(0.01, 0.1)
            
            return hi_mean, hi_std, latent_mean, latent_std, recon_error


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_cvae(
    dataset: List[Tuple[SensorSnapshot, OperationContext, Optional[float]]],
    config: Optional[CVAEConfig] = None,
    save_path: Optional[Path] = None,
) -> CVAE:
    """
    Treinar o modelo CVAE.
    
    Args:
        dataset: Lista de (sensor_snapshot, operation_context, hi_target)
                 hi_target é opcional (para treino semi-supervisionado)
        config: Configuração do modelo
        save_path: Caminho para guardar pesos
    
    Returns:
        Modelo CVAE treinado
    
    TODO[R&D]:
    - Early stopping baseado em validation loss
    - Learning rate scheduling
    - Data augmentation para robustez
    - Curriculum learning (começar com exemplos fáceis)
    """
    config = config or CVAEConfig()
    model = CVAE(config)
    
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch não disponível. Retornando modelo simulado.")
        return model
    
    # Preparar dados
    X = []
    machine_indices = []
    op_indices = []
    product_indices = []
    hi_targets = []
    
    for snapshot, context, hi_target in dataset:
        X.append(snapshot.to_vector())
        m_idx, o_idx, p_idx = context.to_indices(config)
        machine_indices.append(m_idx)
        op_indices.append(o_idx)
        product_indices.append(p_idx)
        hi_targets.append(hi_target if hi_target is not None else -1.0)
    
    X = torch.tensor(np.array(X), dtype=torch.float32)
    machine_indices = torch.tensor(machine_indices, dtype=torch.long)
    op_indices = torch.tensor(op_indices, dtype=torch.long)
    product_indices = torch.tensor(product_indices, dtype=torch.long)
    hi_targets = torch.tensor(hi_targets, dtype=torch.float32)
    
    # DataLoader
    tensor_dataset = TensorDataset(X, machine_indices, op_indices, product_indices, hi_targets)
    dataloader = DataLoader(tensor_dataset, batch_size=config.batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_metrics = {"recon_loss": 0.0, "kl_loss": 0.0, "hi_loss": 0.0}
        
        for batch in dataloader:
            x, m_idx, o_idx, p_idx, hi_t = batch
            
            optimizer.zero_grad()
            
            x_recon, mu, logvar, hi = model(x, m_idx, o_idx, p_idx)
            
            # Use HI targets only where available (not -1)
            hi_target_batch = None
            if (hi_t >= 0).any():
                mask = hi_t >= 0
                if mask.sum() > 0:
                    hi_target_batch = hi_t[mask]
                    hi = hi.squeeze()[mask]
            
            loss, metrics = model.loss_function(x, x_recon, mu, logvar, hi, hi_target_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] += v
        
        # Log progress
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.4f}")
    
    # Save model
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
        }, save_path)
        logger.info(f"Modelo guardado em {save_path}")
    
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def infer_hi(
    model: CVAE,
    sensor_snapshot: SensorSnapshot,
    context: OperationContext,
    config: Optional[CVAEConfig] = None,
) -> HealthIndicatorResult:
    """
    Inferir Health Indicator para uma máquina/operação.
    
    Args:
        model: Modelo CVAE treinado
        sensor_snapshot: Dados de sensores atuais
        context: Contexto da operação
        config: Configuração (para thresholds)
    
    Returns:
        HealthIndicatorResult com HI, incerteza e status
    """
    config = config or CVAEConfig()
    
    # Preparar input
    x = sensor_snapshot.to_vector()
    m_idx, o_idx, p_idx = context.to_indices(config)
    
    if TORCH_AVAILABLE:
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        m_tensor = torch.tensor([m_idx], dtype=torch.long)
        o_tensor = torch.tensor([o_idx], dtype=torch.long)
        p_tensor = torch.tensor([p_idx], dtype=torch.long)
        
        hi_mean, hi_std, latent_mean, latent_std, recon_error = model.infer_health_indicator(
            x_tensor, m_tensor, o_tensor, p_tensor
        )
    else:
        hi_mean, hi_std, latent_mean, latent_std, recon_error = model.infer_health_indicator(
            x, m_idx, o_idx, p_idx
        )
    
    # Determine status
    if hi_mean >= config.hi_threshold_warning:
        status = "HEALTHY"
    elif hi_mean >= config.hi_threshold_critical:
        status = "WARNING"
    else:
        status = "CRITICAL"
    
    return HealthIndicatorResult(
        machine_id=sensor_snapshot.machine_id,
        timestamp=sensor_snapshot.timestamp,
        hi=float(hi_mean),
        hi_std=float(hi_std),
        latent_mean=latent_mean,
        latent_std=latent_std,
        reconstruction_error=float(recon_error),
        status=status,
        op_code=context.op_code,
        product_type=context.product_type,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_demo_dataset(
    num_samples: int = 1000,
    num_machines: int = 10,
) -> List[Tuple[SensorSnapshot, OperationContext, float]]:
    """
    Gerar dataset de demonstração para treino do CVAE.
    
    Simula máquinas em diferentes estados de degradação:
    - Saudáveis: baixa vibração, temperatura normal
    - Degradadas: vibração aumentada, temperatura elevada
    - Críticas: alta vibração, temperatura muito elevada
    """
    dataset = []
    now = datetime.now(timezone.utc)
    
    machines = [f"M-{i:03d}" for i in range(num_machines)]
    operations = ["OP-CUT", "OP-MILL", "OP-DRILL", "OP-TURN", "OP-GRIND"]
    products = ["PROD-A", "PROD-B", "PROD-C"]
    
    for i in range(num_samples):
        machine = np.random.choice(machines)
        op_code = np.random.choice(operations)
        product = np.random.choice(products)
        
        # Simular estado de degradação (0 = novo, 1 = crítico)
        degradation = np.random.beta(2, 5)  # Maioria saudável
        
        # Health indicator (inverso da degradação)
        hi_target = 1.0 - degradation
        
        # Gerar sensor readings baseados na degradação
        base_vibration = 0.1 + degradation * 0.5
        base_temp = 0.2 + degradation * 0.4
        
        snapshot = SensorSnapshot(
            machine_id=machine,
            timestamp=now,
            vibration_x=base_vibration + np.random.normal(0, 0.05),
            vibration_y=base_vibration + np.random.normal(0, 0.05),
            vibration_z=base_vibration + np.random.normal(0, 0.05),
            vibration_rms=base_vibration * 1.73 + np.random.normal(0, 0.05),
            current_phase_a=0.3 + degradation * 0.2 + np.random.normal(0, 0.02),
            current_phase_b=0.3 + degradation * 0.2 + np.random.normal(0, 0.02),
            current_phase_c=0.3 + degradation * 0.2 + np.random.normal(0, 0.02),
            current_rms=0.52 + degradation * 0.3 + np.random.normal(0, 0.03),
            temperature_bearing=base_temp + np.random.normal(0, 0.03),
            temperature_motor=base_temp * 0.8 + np.random.normal(0, 0.03),
            temperature_ambient=0.25 + np.random.normal(0, 0.02),
            acoustic_emission=0.2 + degradation * 0.4 + np.random.normal(0, 0.05),
            oil_particle_count=0.1 + degradation * 0.3 + np.random.normal(0, 0.02),
            pressure=0.5 + np.random.normal(0, 0.05),
            speed_rpm=0.7 + np.random.normal(0, 0.1),
            load_percent=0.6 + np.random.normal(0, 0.1),
            power_factor=0.85 - degradation * 0.1 + np.random.normal(0, 0.02),
        )
        
        context = OperationContext(
            machine_id=machine,
            op_code=op_code,
            product_type=product,
            time_since_maintenance_hours=degradation * 1000 + np.random.uniform(0, 100),
        )
        
        dataset.append((snapshot, context, hi_target))
    
    return dataset

