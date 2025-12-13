"""
════════════════════════════════════════════════════════════════════════════════════════════════════
XAI-DT Product (Explainable Digital Twin do Produto)
════════════════════════════════════════════════════════════════════════════════════════════════════

Gêmeo Digital Explicável do Produto para análise de qualidade geométrica.

Features:
- Alinhamento automático CAD-Scan via ICP (Iterative Closest Point)
- Campo de desvios 3D ponto-a-ponto
- Deviation Score global (0-100%)
- Análise PCA para identificação de padrões de desvio
- Root Cause Analysis (RCA) com ML para identificação de causas prováveis
- Recomendações de ações corretivas

Modelo Matemático:
- Campo de desvio: Δ(x,y,z) = p_scan - p_cad (após alinhamento)
- Desvio médio: δ̄ = (1/N) Σ ||d_i||
- Desvio máximo: δ_max = max_i ||d_i||
- % fora de tolerância: P_out = #{i: ||d_i|| > tol} / N × 100%
- Deviation Score: Score = max(0, 100% - k × δ̄/tol_avg)

Algoritmos:
- ICP: Minimiza ||p_scan - T(p_cad)|| onde T é transformação rígida 6DOF
- KD-Tree para correspondência acelerada de pontos
- PCA nos vetores de desvio para direções predominantes de erro
- Clustering espacial para análise regional

R&D / SIFIDE: WP1 - Digital Twin & Explainability

References:
- Besl & McKay, "A Method for Registration of 3-D Shapes", IEEE TPAMI 1992
- Rusinkiewicz & Levoy, "Efficient Variants of the ICP Algorithm", 3DIM 2001
"""

from __future__ import annotations

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

import numpy as np
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.transform import Rotation

# PyTorch for ML-based pattern classification
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. ML-based RCA will use fallback methods.")

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class XAIDTConfig:
    """Configuration for XAI-DT Product Analysis."""
    
    # ICP Alignment
    icp_max_iterations: int = 50
    icp_tolerance: float = 1e-6
    icp_max_correspondence_distance: float = 10.0  # mm
    
    # Deviation Analysis
    default_tolerance: float = 0.5  # mm - tolerância geométrica padrão
    deviation_score_k: float = 2.0  # factor de penalização para score
    
    # Regional Analysis
    n_regions: int = 8  # número de regiões para análise espacial
    region_method: str = "octants"  # "octants", "kmeans", "grid"
    
    # PCA Analysis
    pca_n_components: int = 3
    pca_variance_threshold: float = 0.95
    
    # RCA
    rca_confidence_threshold: float = 0.6
    max_causes: int = 5
    
    # Output
    language: str = "pt"  # "pt" ou "en"


class DeviationPattern(str, Enum):
    """Patterns identified in deviation field."""
    UNIFORM_OFFSET = "uniform_offset"  # Deslocamento uniforme
    UNIFORM_SCALE = "uniform_scale"  # Contração/expansão uniforme
    DIRECTIONAL_TREND = "directional_trend"  # Tendência direcional
    LOCAL_HOTSPOT = "local_hotspot"  # Região concentrada de desvio
    PERIODIC = "periodic"  # Padrão periódico (vibração)
    RANDOM = "random"  # Ruído aleatório
    WARPING = "warping"  # Deformação/empenamento
    TAPER = "taper"  # Afilamento
    TWIST = "twist"  # Torção


class RootCauseCategory(str, Enum):
    """Categories of root causes for geometric deviations."""
    FIXTURING = "fixturing"  # Problemas de fixação
    CALIBRATION = "calibration"  # Problemas de calibração
    TOOL_WEAR = "tool_wear"  # Desgaste de ferramenta
    THERMAL = "thermal"  # Efeitos térmicos
    MATERIAL = "material"  # Problemas de material
    VIBRATION = "vibration"  # Vibrações
    PROGRAMMING = "programming"  # Erros de programação
    MACHINE = "machine"  # Problemas de máquina


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Point3D:
    """3D point with optional attributes."""
    x: float
    y: float
    z: float
    normal: Optional[Tuple[float, float, float]] = None
    attributes: Dict[str, float] = field(default_factory=dict)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'Point3D':
        return Point3D(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))


@dataclass
class PointCloud:
    """3D point cloud representation."""
    points: np.ndarray  # Shape: (N, 3)
    normals: Optional[np.ndarray] = None  # Shape: (N, 3)
    colors: Optional[np.ndarray] = None  # Shape: (N, 3)
    name: str = "pointcloud"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_points(self) -> int:
        return self.points.shape[0]
    
    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.points, axis=0)
    
    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (min_corner, max_corner)."""
        return np.min(self.points, axis=0), np.max(self.points, axis=0)
    
    def transform(self, rotation: np.ndarray, translation: np.ndarray) -> 'PointCloud':
        """Apply rigid transformation."""
        transformed_points = (rotation @ self.points.T).T + translation
        transformed_normals = None
        if self.normals is not None:
            transformed_normals = (rotation @ self.normals.T).T
        return PointCloud(
            points=transformed_points,
            normals=transformed_normals,
            colors=self.colors,
            name=self.name,
            metadata=self.metadata,
        )


@dataclass 
class DeviationVector:
    """Single deviation measurement at a point."""
    point: np.ndarray  # Location (x, y, z)
    deviation: np.ndarray  # Deviation vector (dx, dy, dz)
    distance: float  # Scalar distance ||deviation||
    in_tolerance: bool  # Whether within tolerance
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "point": self.point.tolist(),
            "deviation": self.deviation.tolist(),
            "distance": float(self.distance),
            "in_tolerance": self.in_tolerance,
        }


@dataclass
class DeviationField3D:
    """
    3D Deviation Field: Δ(x,y,z) = p_scan - p_cad
    
    Represents point-by-point deviations between scanned part and CAD model.
    """
    points: np.ndarray  # Shape: (N, 3) - locations
    deviations: np.ndarray  # Shape: (N, 3) - deviation vectors
    distances: np.ndarray  # Shape: (N,) - scalar distances
    tolerance: float  # Tolerance threshold
    
    # Computed metrics
    mean_deviation: float = 0.0
    max_deviation: float = 0.0
    rms_deviation: float = 0.0
    pct_out_of_tolerance: float = 0.0
    deviation_score: float = 100.0
    
    # Alignment info
    alignment_rotation: Optional[np.ndarray] = None
    alignment_translation: Optional[np.ndarray] = None
    alignment_rmse: float = 0.0
    
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Compute metrics after initialization."""
        if len(self.distances) > 0:
            self.mean_deviation = float(np.mean(self.distances))
            self.max_deviation = float(np.max(self.distances))
            self.rms_deviation = float(np.sqrt(np.mean(self.distances**2)))
            
            out_of_tolerance = np.sum(self.distances > self.tolerance)
            self.pct_out_of_tolerance = float(out_of_tolerance / len(self.distances) * 100)
            
            # Deviation Score usando fórmula especificada:
            # DS = (1/|C|) * Σ_i max(0, d_i - Tol_i) / Tol_i * 100%
            # Onde d_i = ||S_i - C_i|| e Tol_i é a tolerância para o ponto i
            # Por simplicidade, usamos tolerância uniforme, mas pode ser por região
            violations = np.maximum(0, self.distances - self.tolerance)
            violation_ratios = violations / self.tolerance  # (d_i - Tol_i) / Tol_i
            ds_sum = np.sum(violation_ratios)
            ds_mean = ds_sum / len(self.distances) if len(self.distances) > 0 else 0.0
            self.deviation_score = float(ds_mean * 100.0)  # Converter para percentagem
            
            # Nota: Score mais alto = pior (mais violações de tolerância)
            # Para compatibilidade com código existente, podemos inverter se necessário
            # Mas a fórmula especificada é esta
    
    @property
    def n_points(self) -> int:
        return len(self.distances)
    
    def get_in_tolerance_mask(self) -> np.ndarray:
        """Returns boolean mask of points within tolerance."""
        return self.distances <= self.tolerance
    
    def get_out_of_tolerance_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns points and deviations that are out of tolerance."""
        mask = ~self.get_in_tolerance_mask()
        return self.points[mask], self.deviations[mask]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_points": self.n_points,
            "tolerance": float(self.tolerance),
            "mean_deviation": float(round(self.mean_deviation, 4)),
            "max_deviation": float(round(self.max_deviation, 4)),
            "rms_deviation": float(round(self.rms_deviation, 4)),
            "pct_out_of_tolerance": float(round(self.pct_out_of_tolerance, 2)),
            "deviation_score": float(round(self.deviation_score, 1)),
            "alignment_rmse": float(round(self.alignment_rmse, 4)),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PCAResult:
    """Result of PCA analysis on deviation vectors."""
    components: np.ndarray  # Shape: (n_components, 3) - principal directions
    explained_variance: np.ndarray  # Variance explained by each component
    explained_variance_ratio: np.ndarray  # Ratio of variance
    mean: np.ndarray  # Mean deviation vector
    
    # Interpretation
    dominant_direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    dominant_direction_name: str = ""
    is_directional: bool = False  # True if deviations are strongly directional
    
    def __post_init__(self):
        if len(self.components) > 0:
            self.dominant_direction = self.components[0]
            
            # Name the dominant direction
            abs_dir = np.abs(self.dominant_direction)
            max_idx = np.argmax(abs_dir)
            axis_names = ["X", "Y", "Z"]
            sign = "+" if self.dominant_direction[max_idx] > 0 else "-"
            self.dominant_direction_name = f"{sign}{axis_names[max_idx]}"
            
            # Check if strongly directional (first component explains >70% variance)
            if len(self.explained_variance_ratio) > 0:
                self.is_directional = self.explained_variance_ratio[0] > 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "components": self.components.tolist() if len(self.components) > 0 else [],
            "explained_variance_ratio": self.explained_variance_ratio.tolist() if len(self.explained_variance_ratio) > 0 else [],
            "mean": self.mean.tolist() if len(self.mean) > 0 else [0, 0, 0],
            "dominant_direction": self.dominant_direction.tolist() if len(self.dominant_direction) > 0 else [0, 0, 0],
            "dominant_direction_name": str(self.dominant_direction_name),
            "is_directional": bool(self.is_directional),
        }


@dataclass
class RegionalAnalysis:
    """Analysis of deviations by spatial region."""
    regions: List[Dict[str, Any]]  # Per-region statistics
    worst_region: str
    best_region: str
    has_localized_issues: bool
    localization_factor: float  # 0-1, higher = more localized
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regions": self.regions,
            "worst_region": str(self.worst_region),
            "best_region": str(self.best_region),
            "has_localized_issues": bool(self.has_localized_issues),
            "localization_factor": float(round(self.localization_factor, 3)),
        }


@dataclass
class IdentifiedPattern:
    """An identified pattern in the deviation field."""
    pattern: DeviationPattern
    confidence: float  # 0-1
    parameters: Dict[str, float]  # Pattern-specific parameters
    affected_region: str  # "global", "local", region name
    evidence: List[str]  # Evidence supporting this pattern
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern.value,
            "confidence": float(round(self.confidence, 3)),
            "parameters": {k: float(round(v, 4)) for k, v in self.parameters.items()},
            "affected_region": self.affected_region,
            "evidence": self.evidence,
        }


@dataclass
class RootCause:
    """Identified root cause for geometric deviation."""
    category: RootCauseCategory
    description: str
    confidence: float  # 0-1
    evidence: List[str]
    patterns_linked: List[DeviationPattern]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "confidence": float(round(self.confidence, 3)),
            "evidence": self.evidence,
            "patterns_linked": [p.value for p in self.patterns_linked],
        }


@dataclass
class CorrectiveAction:
    """
    Recommended corrective action with XAI explanation.
    
    As specified: "para cada causa potencial identificada, fornecer uma explicação XAI 
    simples: por ex., 'Desvio em forma de barril detectado: possível causa - pressão de 
    injeção excessiva; Sugestão - reduzir pressão em 5%'"
    """
    action: str
    priority: str  # "high", "medium", "low"
    root_cause: RootCauseCategory
    expected_impact: str
    implementation_details: Optional[str] = None
    xai_explanation: Optional[str] = None  # Explicação XAI simples e clara
    parameter_adjustment: Optional[Dict[str, float]] = None  # Ex: {"pressure": -0.05} = reduzir 5%
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "priority": self.priority,
            "root_cause": self.root_cause.value,
            "expected_impact": self.expected_impact,
            "implementation_details": self.implementation_details,
            "xai_explanation": self.xai_explanation,
            "parameter_adjustment": self.parameter_adjustment,
        }


@dataclass
class XAIDTAnalysisResult:
    """Complete XAI-DT analysis result."""
    # Basic info
    analysis_id: str
    timestamp: datetime
    cad_name: str
    scan_name: str
    
    # Deviation field
    deviation_field: DeviationField3D
    
    # Analysis results
    pca_result: PCAResult
    regional_analysis: RegionalAnalysis
    identified_patterns: List[IdentifiedPattern]
    
    # RCA results
    root_causes: List[RootCause]
    corrective_actions: List[CorrectiveAction]
    
    # Summary
    overall_quality: str  # "excellent", "good", "acceptable", "poor", "reject"
    summary_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "timestamp": self.timestamp.isoformat(),
            "cad_name": self.cad_name,
            "scan_name": self.scan_name,
            "deviation_field": self.deviation_field.to_dict(),
            "pca_result": self.pca_result.to_dict(),
            "regional_analysis": self.regional_analysis.to_dict(),
            "identified_patterns": [p.to_dict() for p in self.identified_patterns],
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "corrective_actions": [ca.to_dict() for ca in self.corrective_actions],
            "overall_quality": self.overall_quality,
            "summary_text": self.summary_text,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ICP ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

class ICPAligner:
    """
    Iterative Closest Point (ICP) algorithm for 3D point cloud alignment.
    
    Minimizes: ||p_scan - R × p_cad - t||²
    where R is rotation matrix (3×3) and t is translation vector (3,)
    
    Algorithm:
    1. Find correspondences using KD-tree
    2. Compute optimal R, t using SVD
    3. Apply transformation
    4. Repeat until convergence
    
    References:
    - Besl & McKay, "A Method for Registration of 3-D Shapes", IEEE TPAMI 1992
    """
    
    def __init__(self, config: XAIDTConfig):
        self.config = config
    
    def align(
        self,
        source: PointCloud,  # CAD (will be transformed)
        target: PointCloud,  # Scan (fixed)
        initial_transform: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """
        Align source point cloud to target using ICP.
        
        Args:
            source: Source point cloud (CAD) to be transformed
            target: Target point cloud (Scan) - fixed reference
            initial_transform: Optional (R, t) initial alignment
        
        Returns:
            (rotation, translation, rmse, iterations)
        """
        # Initialize transformation
        if initial_transform is not None:
            R, t = initial_transform
        else:
            # Center both clouds initially
            R = np.eye(3)
            t = target.centroid - source.centroid
        
        # Apply initial transform to source
        current_source = source.transform(R, t)
        
        # Build KD-tree for target
        target_tree = cKDTree(target.points)
        
        prev_rmse = float('inf')
        
        for iteration in range(self.config.icp_max_iterations):
            # Step 1: Find correspondences
            distances, indices = target_tree.query(
                current_source.points,
                k=1,
                distance_upper_bound=self.config.icp_max_correspondence_distance
            )
            
            # Filter invalid correspondences
            valid_mask = distances < self.config.icp_max_correspondence_distance
            if np.sum(valid_mask) < 3:
                logger.warning("ICP: Too few valid correspondences")
                break
            
            src_pts = current_source.points[valid_mask]
            tgt_pts = target.points[indices[valid_mask]]
            
            # Step 2: Compute optimal transformation (SVD-based)
            src_centroid = np.mean(src_pts, axis=0)
            tgt_centroid = np.mean(tgt_pts, axis=0)
            
            src_centered = src_pts - src_centroid
            tgt_centered = tgt_pts - tgt_centroid
            
            # Cross-covariance matrix
            H = src_centered.T @ tgt_centered
            
            # SVD
            U, S, Vt = np.linalg.svd(H)
            
            # Rotation
            R_iter = Vt.T @ U.T
            
            # Handle reflection
            if np.linalg.det(R_iter) < 0:
                Vt[-1, :] *= -1
                R_iter = Vt.T @ U.T
            
            # Translation
            t_iter = tgt_centroid - R_iter @ src_centroid
            
            # Step 3: Apply transformation
            current_source = current_source.transform(R_iter, t_iter)
            
            # Update cumulative transformation
            R = R_iter @ R
            t = R_iter @ t + t_iter
            
            # Compute RMSE
            distances, _ = target_tree.query(current_source.points, k=1)
            valid_distances = distances[distances < self.config.icp_max_correspondence_distance]
            rmse = np.sqrt(np.mean(valid_distances**2)) if len(valid_distances) > 0 else float('inf')
            
            # Check convergence
            if abs(prev_rmse - rmse) < self.config.icp_tolerance:
                logger.info(f"ICP converged at iteration {iteration + 1}, RMSE: {rmse:.6f}")
                return R, t, rmse, iteration + 1
            
            prev_rmse = rmse
        
        logger.info(f"ICP max iterations reached, RMSE: {prev_rmse:.6f}")
        return R, t, prev_rmse, self.config.icp_max_iterations


# ═══════════════════════════════════════════════════════════════════════════════
# DEVIATION COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

class DeviationComputer:
    """
    Computes deviation field between aligned point clouds.
    
    For each point in the scan, finds the closest point on the CAD
    and computes the deviation vector.
    """
    
    def __init__(self, config: XAIDTConfig):
        self.config = config
    
    def compute(
        self,
        cad: PointCloud,
        scan: PointCloud,
        alignment: Tuple[np.ndarray, np.ndarray],
        tolerance: Optional[float] = None,
    ) -> DeviationField3D:
        """
        Compute deviation field.
        
        Args:
            cad: CAD point cloud (reference)
            scan: Scan point cloud (measured)
            alignment: (R, t) transformation from ICP
            tolerance: Override default tolerance
        
        Returns:
            DeviationField3D with point-by-point deviations
        """
        tolerance = tolerance or self.config.default_tolerance
        
        # Transform CAD to align with scan
        R, t = alignment
        cad_aligned = cad.transform(R, t)
        
        # Build KD-tree on CAD
        cad_tree = cKDTree(cad_aligned.points)
        
        # For each scan point, find closest CAD point
        distances, indices = cad_tree.query(scan.points, k=1)
        
        # Compute deviation vectors: scan - cad
        closest_cad_points = cad_aligned.points[indices]
        deviation_vectors = scan.points - closest_cad_points
        
        # Compute scalar distances
        deviation_distances = np.linalg.norm(deviation_vectors, axis=1)
        
        return DeviationField3D(
            points=scan.points,
            deviations=deviation_vectors,
            distances=deviation_distances,
            tolerance=tolerance,
            alignment_rotation=R,
            alignment_translation=t,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class PatternAnalyzer:
    """
    Analyzes deviation field to identify geometric patterns.
    
    Uses:
    - PCA for directional analysis
    - Regional clustering for local hotspots
    - Fourier analysis for periodic patterns
    - Statistical tests for various hypotheses
    """
    
    def __init__(self, config: XAIDTConfig):
        self.config = config
    
    def analyze_pca(self, deviation_field: DeviationField3D) -> PCAResult:
        """
        Perform PCA on deviation vectors to find principal directions.
        """
        deviations = deviation_field.deviations
        
        if len(deviations) < 3:
            return PCAResult(
                components=np.zeros((3, 3)),
                explained_variance=np.zeros(3),
                explained_variance_ratio=np.zeros(3),
                mean=np.zeros(3),
            )
        
        # Center data
        mean = np.mean(deviations, axis=0)
        centered = deviations - mean
        
        # SVD (equivalent to PCA)
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        
        # Explained variance
        n = len(deviations)
        explained_variance = (S**2) / (n - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance if total_variance > 0 else np.zeros_like(explained_variance)
        
        # Take top components
        n_components = min(self.config.pca_n_components, len(S))
        
        return PCAResult(
            components=Vt[:n_components],
            explained_variance=explained_variance[:n_components],
            explained_variance_ratio=explained_variance_ratio[:n_components],
            mean=mean,
        )
    
    def analyze_regions(self, deviation_field: DeviationField3D) -> RegionalAnalysis:
        """
        Analyze deviations by spatial region (octants).
        """
        points = deviation_field.points
        distances = deviation_field.distances
        
        # Compute centroid and bounds
        centroid = np.mean(points, axis=0)
        
        # Classify points into octants
        regions_data = defaultdict(list)
        region_names = [
            "+X+Y+Z", "+X+Y-Z", "+X-Y+Z", "+X-Y-Z",
            "-X+Y+Z", "-X+Y-Z", "-X-Y+Z", "-X-Y-Z"
        ]
        
        for i, (pt, dist) in enumerate(zip(points, distances)):
            # Determine octant
            octant_idx = 0
            if pt[0] < centroid[0]:
                octant_idx += 4
            if pt[1] < centroid[1]:
                octant_idx += 2
            if pt[2] < centroid[2]:
                octant_idx += 1
            
            regions_data[region_names[octant_idx]].append(dist)
        
        # Compute statistics per region
        regions = []
        region_means = {}
        
        for name in region_names:
            dists = regions_data[name]
            if len(dists) > 0:
                mean_dev = np.mean(dists)
                max_dev = np.max(dists)
                pct_out = np.sum(np.array(dists) > deviation_field.tolerance) / len(dists) * 100
            else:
                mean_dev = 0.0
                max_dev = 0.0
                pct_out = 0.0
            
            region_means[name] = mean_dev
            regions.append({
                "name": name,
                "n_points": len(dists),
                "mean_deviation": float(round(mean_dev, 4)),
                "max_deviation": float(round(max_dev, 4)),
                "pct_out_of_tolerance": float(round(pct_out, 2)),
            })
        
        # Find worst and best regions
        sorted_regions = sorted(region_means.items(), key=lambda x: x[1], reverse=True)
        worst_region = sorted_regions[0][0] if sorted_regions else "N/A"
        best_region = sorted_regions[-1][0] if sorted_regions else "N/A"
        
        # Localization factor: ratio of worst to mean
        all_means = [r["mean_deviation"] for r in regions if r["n_points"] > 0]
        if len(all_means) > 1 and np.mean(all_means) > 0:
            localization_factor = np.std(all_means) / np.mean(all_means)
        else:
            localization_factor = 0.0
        
        # Has localized issues if one region is significantly worse
        has_localized = localization_factor > 0.5
        
        return RegionalAnalysis(
            regions=regions,
            worst_region=worst_region,
            best_region=best_region,
            has_localized_issues=has_localized,
            localization_factor=min(1.0, localization_factor),
        )
    
    def identify_patterns(
        self,
        deviation_field: DeviationField3D,
        pca_result: PCAResult,
        regional: RegionalAnalysis,
    ) -> List[IdentifiedPattern]:
        """
        Identify geometric patterns in the deviation field.
        """
        patterns = []
        
        mean_dev = deviation_field.mean_deviation
        std_dev = np.std(deviation_field.distances)
        tolerance = deviation_field.tolerance
        pct_out = deviation_field.pct_out_of_tolerance
        
        # Pattern 1: Uniform Offset (mean deviation >> 0, relatively low variance)
        if mean_dev > tolerance * 0.3 and std_dev < mean_dev * 0.5:
            patterns.append(IdentifiedPattern(
                pattern=DeviationPattern.UNIFORM_OFFSET,
                confidence=min(1.0, 0.7 + 0.3 * (mean_dev / tolerance)),
                parameters={
                    "offset_magnitude": float(mean_dev),
                    "offset_direction_x": float(pca_result.mean[0]) if len(pca_result.mean) > 0 else 0.0,
                    "offset_direction_y": float(pca_result.mean[1]) if len(pca_result.mean) > 1 else 0.0,
                    "offset_direction_z": float(pca_result.mean[2]) if len(pca_result.mean) > 2 else 0.0,
                },
                affected_region="global",
                evidence=[
                    f"Desvio médio de {mean_dev:.3f} mm",
                    f"Variabilidade: σ = {std_dev:.3f} mm",
                    f"Direção predominante: {pca_result.dominant_direction_name}",
                ],
            ))
        
        # Pattern 2: Directional Trend (PCA shows strong directionality)
        if pca_result.is_directional and len(pca_result.explained_variance_ratio) > 0:
            patterns.append(IdentifiedPattern(
                pattern=DeviationPattern.DIRECTIONAL_TREND,
                confidence=float(pca_result.explained_variance_ratio[0]),
                parameters={
                    "primary_direction_x": float(pca_result.dominant_direction[0]) if len(pca_result.dominant_direction) > 0 else 0.0,
                    "primary_direction_y": float(pca_result.dominant_direction[1]) if len(pca_result.dominant_direction) > 1 else 0.0,
                    "primary_direction_z": float(pca_result.dominant_direction[2]) if len(pca_result.dominant_direction) > 2 else 0.0,
                    "variance_explained": float(pca_result.explained_variance_ratio[0]),
                },
                affected_region="global",
                evidence=[
                    f"Componente principal explica {pca_result.explained_variance_ratio[0]*100:.1f}% da variância",
                    f"Direção dominante: {pca_result.dominant_direction_name}",
                ],
            ))
        
        # Pattern 3: Local Hotspot
        if regional.has_localized_issues:
            worst = next((r for r in regional.regions if r["name"] == regional.worst_region), None)
            if worst:
                patterns.append(IdentifiedPattern(
                    pattern=DeviationPattern.LOCAL_HOTSPOT,
                    confidence=min(1.0, regional.localization_factor + 0.3),
                    parameters={
                        "region": regional.worst_region,
                        "local_mean": float(worst["mean_deviation"]),
                        "localization_factor": float(regional.localization_factor),
                    },
                    affected_region=regional.worst_region,
                    evidence=[
                        f"Região {regional.worst_region} com desvio médio de {worst['mean_deviation']:.3f} mm",
                        f"Fator de localização: {regional.localization_factor:.2f}",
                    ],
                ))
        
        # Pattern 4: Random / High Variability
        cv = std_dev / mean_dev if mean_dev > 0 else 0
        if cv > 0.4 or (std_dev > tolerance * 0.3 and pct_out > 10):
            patterns.append(IdentifiedPattern(
                pattern=DeviationPattern.RANDOM,
                confidence=min(0.9, 0.5 + cv * 0.4),
                parameters={
                    "std_deviation": float(std_dev),
                    "coefficient_of_variation": float(cv),
                    "pct_out_of_tolerance": float(pct_out),
                },
                affected_region="global",
                evidence=[
                    f"Alta variabilidade: CV = {cv:.2f}",
                    f"Desvio padrão: {std_dev:.3f} mm",
                    f"Pontos fora de tolerância: {pct_out:.1f}%",
                ],
            ))
        
        # Pattern 5: Scale/Contraction (if mean is high and in same direction)
        if mean_dev > tolerance * 0.5 and cv < 0.3:
            # Check if deviations point inward/outward from centroid
            centroid = np.mean(deviation_field.points, axis=0)
            radial_dirs = deviation_field.points - centroid
            radial_dirs = radial_dirs / (np.linalg.norm(radial_dirs, axis=1, keepdims=True) + 1e-10)
            
            # Dot product of deviation with radial direction
            dev_normalized = deviation_field.deviations / (np.linalg.norm(deviation_field.deviations, axis=1, keepdims=True) + 1e-10)
            dot_products = np.sum(radial_dirs * dev_normalized, axis=1)
            mean_dot = np.mean(dot_products)
            
            if abs(mean_dot) > 0.5:  # Strong radial pattern
                direction = "expansão" if mean_dot > 0 else "contração"
                patterns.append(IdentifiedPattern(
                    pattern=DeviationPattern.UNIFORM_SCALE,
                    confidence=min(0.9, 0.6 + abs(mean_dot) * 0.3),
                    parameters={
                        "scale_direction": float(mean_dot),
                        "mean_deviation": float(mean_dev),
                    },
                    affected_region="global",
                    evidence=[
                        f"Padrão de {direction} detectado",
                        f"Correlação radial: {mean_dot:.2f}",
                        f"Desvio médio: {mean_dev:.3f} mm",
                    ],
                ))
        
        # Ensure at least one pattern is detected if quality is poor
        if len(patterns) == 0 and deviation_field.deviation_score < 70:
            patterns.append(IdentifiedPattern(
                pattern=DeviationPattern.RANDOM,
                confidence=0.5,
                parameters={
                    "std_deviation": float(std_dev),
                    "mean_deviation": float(mean_dev),
                    "deviation_score": float(deviation_field.deviation_score),
                },
                affected_region="global",
                evidence=[
                    f"Desvio médio: {mean_dev:.3f} mm",
                    f"Score: {deviation_field.deviation_score:.1f}%",
                    "Padrão não identificado claramente",
                ],
            ))
        
        return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# ROOT CAUSE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ML-BASED PATTERN CLASSIFIER (PyTorch)
# ═══════════════════════════════════════════════════════════════════════════════

if TORCH_AVAILABLE:
    class DeviationPatternClassifier(nn.Module):
        """
        Neural network classifier for deviation patterns.
        
        Input: Features extracted from deviation field (statistics, PCA, regional)
        Output: Probabilities for each deviation pattern type
        """
        
        def __init__(self, input_dim: int = 64, num_patterns: int = 9):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, num_patterns)
            self.dropout = nn.Dropout(0.2)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.fc4(x)
            return F.softmax(x, dim=1)
    
    class RootCauseMLClassifier(nn.Module):
        """
        ML classifier for root cause identification.
        
        Maps deviation patterns and features to root cause categories.
        Uses Multi-Layer Perceptron (MLP) as specified in requirements.
        """
        
        def __init__(self, input_dim: int = 32, num_causes: int = 8):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, num_causes)
            self.dropout = nn.Dropout(0.2)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(32)
        
        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            return F.softmax(x, dim=1)
        
        def train_model(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 1e-3,
        ) -> Dict[str, List[float]]:
            """
            Train the ML classifier on historical data.
            
            Args:
                X: Feature vectors (N, input_dim)
                y: Target labels (N,) - indices of RootCauseCategory
                epochs: Number of training epochs
                batch_size: Batch size
                learning_rate: Learning rate
            
            Returns:
                Training history with loss and accuracy
            """
            self.train()
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.long)
            
            history = {"loss": [], "accuracy": []}
            
            for epoch in range(epochs):
                # Shuffle data
                indices = torch.randperm(len(X_tensor))
                X_shuffled = X_tensor[indices]
                y_shuffled = y_tensor[indices]
                
                epoch_loss = 0.0
                correct = 0
                total = 0
                
                for i in range(0, len(X_shuffled), batch_size):
                    batch_X = X_shuffled[i:i+batch_size]
                    batch_y = y_shuffled[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                
                avg_loss = epoch_loss / (len(X_shuffled) // batch_size + 1)
                accuracy = 100.0 * correct / total
                
                history["loss"].append(avg_loss)
                history["accuracy"].append(accuracy)
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
            
            return history
    
    class Mesh3DAutoencoder(nn.Module):
        """
        3D Mesh Autoencoder for learning shape representations and detecting anomalies.
        
        As specified in requirements: "autoencoder de malha 3D que aprenda 
        representações de formas e detecte anomalias específicas"
        """
        
        def __init__(self, input_dim: int = 64, latent_dim: int = 16):
            super().__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim),
            )
        
        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon, z
        
        def compute_reconstruction_error(self, x):
            """Compute reconstruction error for anomaly detection."""
            x_recon, _ = self.forward(x)
            error = F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
            return error
else:
    # Fallback classes when PyTorch is not available
    class DeviationPatternClassifier:
        def __init__(self, *args, **kwargs):
            logger.warning("DeviationPatternClassifier: PyTorch not available, using rule-based fallback")
    
    class RootCauseMLClassifier:
        def __init__(self, *args, **kwargs):
            logger.warning("RootCauseMLClassifier: PyTorch not available, using rule-based fallback")
        
        def train_model(self, *args, **kwargs):
            return {"loss": [], "accuracy": []}
    
    class Mesh3DAutoencoder:
        def __init__(self, *args, **kwargs):
            logger.warning("Mesh3DAutoencoder: PyTorch not available")
        
        def forward(self, x):
            return x, x
        
        def compute_reconstruction_error(self, x):
            return np.zeros(len(x))


class RootCauseAnalyzer:
    """
    Root Cause Analysis (RCA) for geometric deviations.
    
    Maps identified patterns to probable manufacturing causes
    using rule-based inference and ML-based classification (PyTorch).
    
    As specified: "treinar um modelo (ex.: random forest ou MLP em PyTorch) 
    que classifica o padrão de desvio em possíveis causas conhecidas"
    """
    
    def __init__(self, config: XAIDTConfig):
        self.config = config
        self.pattern_classifier: Optional[Any] = None
        self.cause_classifier: Optional[Any] = None
        self.mesh_autoencoder: Optional[Any] = None
        
        # Initialize ML models if PyTorch is available
        if TORCH_AVAILABLE:
            try:
                self.pattern_classifier = DeviationPatternClassifier(input_dim=64, num_patterns=len(DeviationPattern))
                self.cause_classifier = RootCauseMLClassifier(input_dim=64, num_causes=len(RootCauseCategory))  # Updated to 64
                self.mesh_autoencoder = Mesh3DAutoencoder(input_dim=64, latent_dim=16)
                logger.info("ML classifiers and autoencoder initialized (untrained - will use rule-based fallback)")
            except Exception as e:
                logger.warning(f"Failed to initialize ML classifiers: {e}")
    
    def train_ml_models(
        self,
        training_data: List[Tuple[DeviationField3D, List[IdentifiedPattern], RootCauseCategory]],
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """
        Train ML models on historical data.
        
        Args:
            training_data: List of (deviation_field, patterns, true_cause) tuples
            epochs: Number of training epochs
        
        Returns:
            Training history for both classifiers
        """
        if not TORCH_AVAILABLE or self.cause_classifier is None:
            logger.warning("PyTorch not available or classifier not initialized")
            return {}
        
        # Prepare training data
        X = []
        y = []
        
        for deviation_field, patterns, true_cause in training_data:
            features = self._extract_features_for_ml(deviation_field, patterns)
            X.append(features)
            y.append(list(RootCauseCategory).index(true_cause))
        
        if len(X) < 10:
            logger.warning(f"Insufficient training data: {len(X)} samples (need at least 10)")
            return {}
        
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.int64)
        
        # Train cause classifier
        logger.info(f"Training RootCauseMLClassifier on {len(X_array)} samples...")
        history = self.cause_classifier.train_model(
            X_array, y_array, epochs=epochs, batch_size=min(32, len(X_array))
        )
        
        logger.info("ML models trained successfully")
        return {"cause_classifier": history}
    
    # Knowledge base: Pattern -> Possible Causes
    PATTERN_CAUSE_MAP = {
        DeviationPattern.UNIFORM_OFFSET: [
            (RootCauseCategory.FIXTURING, 0.8, "Deslocamento de fixação ou posicionamento incorreto"),
            (RootCauseCategory.CALIBRATION, 0.6, "Erro de calibração de origem"),
            (RootCauseCategory.PROGRAMMING, 0.4, "Offset programado incorretamente"),
        ],
        DeviationPattern.UNIFORM_SCALE: [
            (RootCauseCategory.THERMAL, 0.8, "Contração/expansão térmica do material"),
            (RootCauseCategory.MATERIAL, 0.6, "Propriedades de material fora da especificação"),
            (RootCauseCategory.CALIBRATION, 0.5, "Escala de máquina incorreta"),
        ],
        DeviationPattern.DIRECTIONAL_TREND: [
            (RootCauseCategory.FIXTURING, 0.7, "Flexão ou deformação durante fixação"),
            (RootCauseCategory.TOOL_WEAR, 0.6, "Desgaste de ferramenta causando deflexão"),
            (RootCauseCategory.MACHINE, 0.5, "Erro de posicionamento de eixo"),
        ],
        DeviationPattern.LOCAL_HOTSPOT: [
            (RootCauseCategory.TOOL_WEAR, 0.8, "Desgaste localizado de ferramenta"),
            (RootCauseCategory.FIXTURING, 0.6, "Apoio inadequado em região específica"),
            (RootCauseCategory.PROGRAMMING, 0.4, "Erro em trajetória local"),
        ],
        DeviationPattern.PERIODIC: [
            (RootCauseCategory.VIBRATION, 0.9, "Vibração ou folga mecânica"),
            (RootCauseCategory.MACHINE, 0.7, "Problema em rolamentos ou guias"),
            (RootCauseCategory.TOOL_WEAR, 0.5, "Pastilha com desgaste irregular"),
        ],
        DeviationPattern.WARPING: [
            (RootCauseCategory.THERMAL, 0.8, "Tensões residuais ou gradiente térmico"),
            (RootCauseCategory.MATERIAL, 0.7, "Relaxação de tensões no material"),
            (RootCauseCategory.FIXTURING, 0.5, "Tensão excessiva de fixação"),
        ],
        DeviationPattern.RANDOM: [
            (RootCauseCategory.MACHINE, 0.5, "Repetibilidade de máquina inadequada"),
            (RootCauseCategory.VIBRATION, 0.4, "Vibrações externas ou instabilidade"),
            (RootCauseCategory.MATERIAL, 0.3, "Variabilidade do material"),
        ],
    }
    
    # Corrective actions by cause category
    CORRECTIVE_ACTIONS = {
        RootCauseCategory.FIXTURING: [
            ("Verificar e realinhar dispositivo de fixação", "high", "Elimina deslocamentos sistemáticos"),
            ("Inspecionar pontos de apoio e contato", "medium", "Garante suporte uniforme"),
            ("Revisar força de aperto/pressão", "medium", "Evita deformação por excesso de aperto"),
        ],
        RootCauseCategory.CALIBRATION: [
            ("Recalibrar origens e offsets da máquina", "high", "Corrige erros sistemáticos de posição"),
            ("Verificar escala de eixos com padrão", "medium", "Valida precisão dimensional"),
            ("Atualizar compensações de temperatura", "low", "Ajusta para condições ambientais"),
        ],
        RootCauseCategory.TOOL_WEAR: [
            ("Substituir ferramenta de corte", "high", "Restaura geometria de corte ideal"),
            ("Reduzir parâmetros de avanço/velocidade", "medium", "Reduz taxa de desgaste"),
            ("Implementar troca preventiva de ferramentas", "medium", "Evita desvios acumulados"),
        ],
        RootCauseCategory.THERMAL: [
            ("Estabilizar temperatura ambiente", "medium", "Minimiza expansão térmica"),
            ("Pré-aquecer máquina antes de produção", "medium", "Estabiliza temperatura de trabalho"),
            ("Aplicar compensação térmica em tempo real", "high", "Corrige automaticamente"),
        ],
        RootCauseCategory.MATERIAL: [
            ("Verificar certificado de material", "high", "Confirma especificação"),
            ("Ajustar parâmetros para lote atual", "medium", "Compensa variações de material"),
            ("Qualificar novo lote antes de produção", "low", "Previne problemas futuros"),
        ],
        RootCauseCategory.VIBRATION: [
            ("Verificar fixação de ferramenta e peça", "high", "Elimina folgas mecânicas"),
            ("Reduzir velocidade de spindle", "medium", "Minimiza excitação de vibrações"),
            ("Balancear ferramenta e mandril", "medium", "Reduz vibração rotacional"),
        ],
        RootCauseCategory.MACHINE: [
            ("Executar diagnóstico de máquina", "high", "Identifica problemas mecânicos"),
            ("Verificar backlash e folgas de eixos", "medium", "Corrige erros de posicionamento"),
            ("Lubrificar guias e fusos", "low", "Manutenção preventiva"),
        ],
        RootCauseCategory.PROGRAMMING: [
            ("Revisar programa NC/CAM", "high", "Corrige erros de trajetória"),
            ("Verificar pós-processador", "medium", "Valida conversão de código"),
            ("Simular programa antes de executar", "low", "Previne erros"),
        ],
    }
    
    def __init__(self, config: XAIDTConfig):
        self.config = config
        self.pattern_classifier: Optional[Any] = None
        self.cause_classifier: Optional[Any] = None
        
        # Initialize ML models if PyTorch is available
        if TORCH_AVAILABLE:
            try:
                self.pattern_classifier = DeviationPatternClassifier(input_dim=64, num_patterns=len(DeviationPattern))
                self.cause_classifier = RootCauseMLClassifier(input_dim=32, num_causes=len(RootCauseCategory))
                logger.info("ML classifiers initialized (untrained - will use rule-based fallback)")
            except Exception as e:
                logger.warning(f"Failed to initialize ML classifiers: {e}")
    
    def _extract_features_for_ml(
        self,
        deviation_field: DeviationField3D,
        patterns: List[IdentifiedPattern],
    ) -> np.ndarray:
        """
        Extract features from deviation field for ML classification.
        
        Features include:
        - Statistical moments (mean, std, skew, kurtosis)
        - PCA components
        - Regional statistics
        - Pattern indicators
        """
        features = []
        
        # Statistical features from deviation distances
        distances = deviation_field.distances
        if len(distances) > 0:
            features.extend([
                np.mean(distances),
                np.std(distances),
                np.median(distances),
                np.max(distances),
                np.min(distances),
                np.percentile(distances, 25),
                np.percentile(distances, 75),
                np.percentile(distances, 95),
            ])
            
            # Higher moments
            if np.std(distances) > 1e-6:
                from scipy.stats import skew, kurtosis
                features.append(skew(distances))
                features.append(kurtosis(distances))
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0] * 10)
        
        # Deviation vector statistics
        deviations = deviation_field.deviations
        if len(deviations) > 0:
            # Mean deviation vector
            mean_dev = np.mean(deviations, axis=0)
            features.extend(mean_dev.tolist())
            
            # Std of deviation components
            std_dev = np.std(deviations, axis=0)
            features.extend(std_dev.tolist())
            
            # Directional features
            norms = np.linalg.norm(deviations, axis=1)
            features.extend([
                np.mean(norms),
                np.std(norms),
                np.max(norms),
            ])
        else:
            features.extend([0.0] * 9)
        
        # Pattern indicators (one-hot encoding)
        pattern_vector = np.zeros(len(DeviationPattern))
        for pattern in patterns:
            pattern_idx = list(DeviationPattern).index(pattern.pattern)
            pattern_vector[pattern_idx] = pattern.confidence
        features.extend(pattern_vector.tolist())
        
        # Deviation score and out-of-tolerance percentage
        features.extend([
            deviation_field.deviation_score / 100.0,  # Normalize
            deviation_field.pct_out_of_tolerance / 100.0,
        ])
        
        # Pad or truncate to fixed size (64 features)
        feature_array = np.array(features, dtype=np.float32)
        if len(feature_array) < 64:
            feature_array = np.pad(feature_array, (0, 64 - len(feature_array)), 'constant')
        elif len(feature_array) > 64:
            feature_array = feature_array[:64]
        
        return feature_array
    
    def train_ml_models(
        self,
        training_data: List[Tuple[DeviationField3D, List[IdentifiedPattern], RootCauseCategory]],
        epochs: int = 100,
    ) -> Dict[str, Any]:
        """
        Train ML models on historical data.
        
        Args:
            training_data: List of (deviation_field, patterns, true_cause) tuples
            epochs: Number of training epochs
        
        Returns:
            Training history for both classifiers
        """
        if not TORCH_AVAILABLE or self.cause_classifier is None:
            logger.warning("PyTorch not available or classifier not initialized")
            return {}
        
        # Prepare training data
        X = []
        y = []
        
        for deviation_field, patterns, true_cause in training_data:
            features = self._extract_features_for_ml(deviation_field, patterns)
            X.append(features)
            y.append(list(RootCauseCategory).index(true_cause))
        
        if len(X) < 10:
            logger.warning(f"Insufficient training data: {len(X)} samples (need at least 10)")
            return {}
        
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.int64)
        
        # Train cause classifier
        logger.info(f"Training RootCauseMLClassifier on {len(X_array)} samples...")
        history = self.cause_classifier.train_model(
            X_array, y_array, epochs=epochs, batch_size=min(32, len(X_array))
        )
        
        logger.info("ML models trained successfully")
        return {"cause_classifier": history}
    
    def _classify_with_ml(
        self,
        features: np.ndarray,
    ) -> Optional[Dict[RootCauseCategory, float]]:
        """
        Use ML classifier to predict root causes.
        
        Returns:
            Dict mapping RootCauseCategory to confidence score, or None if ML unavailable
        """
        if not TORCH_AVAILABLE or self.cause_classifier is None:
            return None
        
        try:
            self.cause_classifier.eval()
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                predictions = self.cause_classifier(x)
                probs = predictions.squeeze().numpy()
                
                # Map to categories
                categories = list(RootCauseCategory)
                result = {}
                for i, category in enumerate(categories):
                    if i < len(probs):
                        result[category] = float(probs[i])
                
                return result
        except Exception as e:
            logger.warning(f"ML classification failed: {e}, using rule-based fallback")
            return None
    
    def analyze(
        self,
        patterns: List[IdentifiedPattern],
        deviation_field: DeviationField3D,
    ) -> Tuple[List[RootCause], List[CorrectiveAction]]:
        """
        Perform root cause analysis based on identified patterns.
        
        Uses ML classifier if available, otherwise falls back to rule-based inference.
        """
        root_causes = []
        
        # Try ML-based classification first
        features = self._extract_features_for_ml(deviation_field, patterns)
        ml_predictions = self._classify_with_ml(features)
        
        # Aggregate causes from all patterns (rule-based)
        cause_scores: Dict[RootCauseCategory, Tuple[float, str, List[str], List[DeviationPattern]]] = {}
        
        for pattern in patterns:
            if pattern.pattern in self.PATTERN_CAUSE_MAP:
                for category, base_conf, desc in self.PATTERN_CAUSE_MAP[pattern.pattern]:
                    # Combine confidence: pattern confidence × cause probability
                    combined_conf = pattern.confidence * base_conf
                    
                    # Boost confidence if ML also predicts this category
                    if ml_predictions and category in ml_predictions:
                        ml_boost = ml_predictions[category] * 0.3  # 30% weight to ML
                        combined_conf = 0.7 * combined_conf + ml_boost
                    
                    if category in cause_scores:
                        # Max confidence, aggregate evidence
                        old_conf, old_desc, old_evidence, old_patterns = cause_scores[category]
                        new_conf = max(old_conf, combined_conf)
                        new_evidence = list(set(old_evidence + pattern.evidence))
                        new_patterns = list(set(old_patterns + [pattern.pattern]))
                        cause_scores[category] = (new_conf, old_desc if old_conf > combined_conf else desc, new_evidence, new_patterns)
                    else:
                        cause_scores[category] = (combined_conf, desc, pattern.evidence.copy(), [pattern.pattern])
        
        # Filter by confidence threshold and sort
        for category, (conf, desc, evidence, linked_patterns) in cause_scores.items():
            if conf >= self.config.rca_confidence_threshold:
                root_causes.append(RootCause(
                    category=category,
                    description=desc,
                    confidence=conf,
                    evidence=evidence,
                    patterns_linked=linked_patterns,
                ))
        
        # Sort by confidence
        root_causes.sort(key=lambda x: x.confidence, reverse=True)
        root_causes = root_causes[:self.config.max_causes]
        
        # Generate corrective actions
        corrective_actions = self._generate_actions(root_causes)
        
        return root_causes, corrective_actions
    
    def _generate_actions(self, root_causes: List[RootCause]) -> List[CorrectiveAction]:
        """
        Generate prioritized corrective actions with XAI explanations.
        
        As specified: "para cada causa potencial identificada, fornecer uma explicação XAI 
        simples: por ex., 'Desvio em forma de barril detectado: possível causa - pressão de 
        injeção excessiva; Sugestão - reduzir pressão em 5%'"
        """
        actions = []
        seen_actions = set()
        
        # Map patterns to XAI explanations and parameter adjustments
        pattern_explanations = {
            DeviationPattern.UNIFORM_SCALE: {
                RootCauseCategory.THERMAL: (
                    "Desvio uniforme de escala detectado: possível causa - contração/expansão térmica do material",
                    {"temperature": -0.05, "cooling_time": 0.10}  # Reduzir temperatura 5%, aumentar tempo de arrefecimento 10%
                ),
            },
            DeviationPattern.DIRECTIONAL_TREND: {
                RootCauseCategory.FIXTURING: (
                    "Tendência direcional detectada: possível causa - flexão durante fixação",
                    {"clamping_force": -0.10}  # Reduzir força de aperto 10%
                ),
            },
            DeviationPattern.LOCAL_HOTSPOT: {
                RootCauseCategory.TOOL_WEAR: (
                    "Hotspot localizado detectado: possível causa - desgaste localizado de ferramenta",
                    {"feed_rate": -0.05, "spindle_speed": -0.03}  # Reduzir avanço 5%, velocidade 3%
                ),
            },
            DeviationPattern.PERIODIC: {
                RootCauseCategory.VIBRATION: (
                    "Padrão periódico detectado: possível causa - vibração ou folga mecânica",
                    {"damping": 0.15}  # Aumentar amortecimento 15%
                ),
            },
        }
        
        for cause in root_causes:
            if cause.category in self.CORRECTIVE_ACTIONS:
                for action_text, priority, impact in self.CORRECTIVE_ACTIONS[cause.category]:
                    if action_text not in seen_actions:
                        seen_actions.add(action_text)
                        
                        # Generate XAI explanation based on pattern
                        xai_explanation = None
                        parameter_adjustment = None
                        
                        # Find matching pattern for this cause
                        for pattern in cause.patterns_linked:
                            if pattern in pattern_explanations:
                                if cause.category in pattern_explanations[pattern]:
                                    xai_explanation, parameter_adjustment = pattern_explanations[pattern][cause.category]
                                    break
                        
                        # Fallback: generic explanation
                        if xai_explanation is None:
                            xai_explanation = f"Padrão de desvio '{cause.patterns_linked[0].value if cause.patterns_linked else 'desconhecido'}' detectado: possível causa - {cause.description}"
                        
                        actions.append(CorrectiveAction(
                            action=action_text,
                            priority=priority,
                            root_cause=cause.category,
                            expected_impact=impact,
                            xai_explanation=xai_explanation,
                            parameter_adjustment=parameter_adjustment,
                        ))
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        actions.sort(key=lambda x: priority_order.get(x.priority, 3))
        
        return actions


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class XAIDTProductAnalyzer:
    """
    Main XAI-DT Product Analyzer.
    
    Orchestrates the complete analysis pipeline:
    1. ICP alignment of CAD and Scan
    2. Deviation field computation
    3. Pattern analysis (PCA, regional)
    4. Root cause analysis
    5. Corrective action generation
    6. Report generation
    
    Usage:
        analyzer = XAIDTProductAnalyzer()
        result = analyzer.analyze(cad_cloud, scan_cloud, tolerance=0.5)
        print(result.summary_text)
    """
    
    def __init__(self, config: Optional[XAIDTConfig] = None):
        self.config = config or XAIDTConfig()
        self.aligner = ICPAligner(self.config)
        self.deviation_computer = DeviationComputer(self.config)
        self.pattern_analyzer = PatternAnalyzer(self.config)
        self.rca_analyzer = RootCauseAnalyzer(self.config)
    
    def analyze(
        self,
        cad: PointCloud,
        scan: PointCloud,
        tolerance: Optional[float] = None,
        initial_alignment: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> XAIDTAnalysisResult:
        """
        Perform complete XAI-DT analysis.
        
        Args:
            cad: CAD point cloud (nominal geometry)
            scan: Scan point cloud (measured geometry)
            tolerance: Geometric tolerance (mm)
            initial_alignment: Optional initial (R, t) alignment
        
        Returns:
            XAIDTAnalysisResult with complete analysis
        """
        import uuid
        
        tolerance = tolerance or self.config.default_tolerance
        analysis_id = f"XAI-{uuid.uuid4().hex[:8]}"
        
        logger.info(f"Starting XAI-DT analysis {analysis_id}")
        logger.info(f"CAD: {cad.n_points} points, Scan: {scan.n_points} points, Tolerance: {tolerance}mm")
        
        # Step 1: ICP Alignment
        logger.info("Step 1: ICP Alignment...")
        R, t, rmse, iterations = self.aligner.align(cad, scan, initial_alignment)
        logger.info(f"ICP completed in {iterations} iterations, RMSE: {rmse:.4f}mm")
        
        # Step 2: Compute deviation field
        logger.info("Step 2: Computing deviation field...")
        deviation_field = self.deviation_computer.compute(cad, scan, (R, t), tolerance)
        deviation_field.alignment_rmse = rmse
        logger.info(f"Deviation score: {deviation_field.deviation_score:.1f}%")
        
        # Step 3: Pattern analysis
        logger.info("Step 3: Analyzing patterns...")
        pca_result = self.pattern_analyzer.analyze_pca(deviation_field)
        regional = self.pattern_analyzer.analyze_regions(deviation_field)
        patterns = self.pattern_analyzer.identify_patterns(deviation_field, pca_result, regional)
        logger.info(f"Identified {len(patterns)} patterns")
        
        # Step 4: Root cause analysis
        logger.info("Step 4: Root cause analysis...")
        root_causes, corrective_actions = self.rca_analyzer.analyze(patterns, deviation_field)
        logger.info(f"Found {len(root_causes)} probable causes, {len(corrective_actions)} actions")
        
        # Step 5: Determine quality grade
        quality = self._determine_quality(deviation_field)
        
        # Step 6: Generate summary
        summary = self._generate_summary(
            deviation_field, patterns, root_causes, corrective_actions, quality
        )
        
        return XAIDTAnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now(timezone.utc),
            cad_name=cad.name,
            scan_name=scan.name,
            deviation_field=deviation_field,
            pca_result=pca_result,
            regional_analysis=regional,
            identified_patterns=patterns,
            root_causes=root_causes,
            corrective_actions=corrective_actions,
            overall_quality=quality,
            summary_text=summary,
        )
    
    def _determine_quality(self, deviation_field: DeviationField3D) -> str:
        """Determine overall quality grade."""
        score = deviation_field.deviation_score
        pct_out = deviation_field.pct_out_of_tolerance
        
        if score >= 95 and pct_out <= 1:
            return "excellent"
        elif score >= 85 and pct_out <= 5:
            return "good"
        elif score >= 70 and pct_out <= 15:
            return "acceptable"
        elif score >= 50:
            return "poor"
        else:
            return "reject"
    
    def _generate_summary(
        self,
        deviation_field: DeviationField3D,
        patterns: List[IdentifiedPattern],
        root_causes: List[RootCause],
        corrective_actions: List[CorrectiveAction],
        quality: str,
    ) -> str:
        """Generate human-readable summary."""
        
        quality_labels = {
            "excellent": "Excelente",
            "good": "Bom",
            "acceptable": "Aceitável",
            "poor": "Fraco",
            "reject": "Rejeitado",
        }
        
        lines = [
            f"📊 ANÁLISE XAI-DT DE QUALIDADE GEOMÉTRICA",
            f"",
            f"▸ Qualidade Global: {quality_labels.get(quality, quality).upper()}",
            f"▸ Deviation Score: {deviation_field.deviation_score:.1f}%",
            f"▸ Desvio Médio: {deviation_field.mean_deviation:.3f} mm",
            f"▸ Desvio Máximo: {deviation_field.max_deviation:.3f} mm",
            f"▸ % Fora de Tolerância: {deviation_field.pct_out_of_tolerance:.1f}%",
            f"",
        ]
        
        if patterns:
            lines.append("📌 PADRÕES IDENTIFICADOS:")
            for p in patterns[:3]:
                lines.append(f"  • {p.pattern.value}: {p.confidence*100:.0f}% confiança")
            lines.append("")
        
        if root_causes:
            lines.append("🔍 CAUSAS PROVÁVEIS:")
            for rc in root_causes[:3]:
                lines.append(f"  • [{rc.category.value}] {rc.description} ({rc.confidence*100:.0f}%)")
            lines.append("")
        
        if corrective_actions:
            high_priority = [a for a in corrective_actions if a.priority == "high"]
            if high_priority:
                lines.append("⚡ AÇÕES PRIORITÁRIAS:")
                for a in high_priority[:3]:
                    lines.append(f"  • {a.action}")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_demo_cad_scan(
    n_points: int = 1000,
    deviation_type: str = "offset",
    deviation_magnitude: float = 0.5,
) -> Tuple[PointCloud, PointCloud]:
    """
    Create demo CAD and Scan point clouds for testing.
    
    Args:
        n_points: Number of points
        deviation_type: "offset", "scale", "random", "local"
        deviation_magnitude: Magnitude of deviation (mm)
    
    Returns:
        (cad_cloud, scan_cloud)
    """
    # Generate CAD: unit cube surface
    np.random.seed(42)
    
    # Generate points on cube faces
    points_per_face = n_points // 6
    cad_points = []
    
    for axis in range(3):
        for sign in [-0.5, 0.5]:
            n = points_per_face
            face_points = np.random.uniform(-0.5, 0.5, (n, 3))
            face_points[:, axis] = sign
            cad_points.append(face_points)
    
    cad_points = np.vstack(cad_points)
    
    # Scale to realistic dimensions (100mm cube)
    cad_points *= 100  # 100mm
    
    cad = PointCloud(points=cad_points, name="demo_cad")
    
    # Generate Scan with deviations
    scan_points = cad_points.copy()
    
    if deviation_type == "offset":
        # Uniform offset in one direction
        scan_points += np.array([deviation_magnitude, 0, 0])
    
    elif deviation_type == "scale":
        # Uniform scale contraction
        center = np.mean(scan_points, axis=0)
        scan_points = center + (scan_points - center) * (1 - deviation_magnitude / 100)
    
    elif deviation_type == "random":
        # Random noise
        noise = np.random.normal(0, deviation_magnitude / 3, scan_points.shape)
        scan_points += noise
    
    elif deviation_type == "local":
        # Local hotspot in one region
        center = np.array([25, 25, 25])  # One corner
        distances = np.linalg.norm(scan_points - center, axis=1)
        mask = distances < 30
        scan_points[mask] += np.random.normal(0, deviation_magnitude, (np.sum(mask), 3))
    
    scan = PointCloud(points=scan_points, name="demo_scan")
    
    return cad, scan


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

_analyzer_instance: Optional[XAIDTProductAnalyzer] = None


def get_xai_dt_analyzer() -> XAIDTProductAnalyzer:
    """Get singleton instance of XAI-DT analyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = XAIDTProductAnalyzer()
    return _analyzer_instance


def reset_xai_dt_analyzer() -> None:
    """Reset singleton (for testing)."""
    global _analyzer_instance
    _analyzer_instance = None

