"""
════════════════════════════════════════════════════════════════════════════════════════════════════
TRUST INDEX SERVICE - Advanced Field-Level Trust Index Calculation
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract D1 Implementation: Field-level Trust Index (0-100)

Service:
- TrustIndexService: Calculates trust index field-by-field and aggregates
- Integrates with DPP data (dpp_data_json or DppRecord fields)
- Persists results to database
- Logs to R&D module for WPX_TRUST_EVOLUTION experiments

Algorithm:
1. For each relevant field (carbon, water, recyclability, composition, origin):
   - Determine base DataSourceType (MEDIDO/REPORTADO/ESTIMADO/DESCONHECIDO)
   - Assign base score (MEDIDO=100, REPORTADO=85, ESTIMADO=65, DESCONHECIDO=0)
   - Apply adjustment factors:
     - Recency (A): f_A = 1.0 (<1 year), 0.95 (1-2 years), 0.9 (2-3 years), 0.85 (>3 years)
     - Third-party verification (B): f_B = 1.1 (audited), 1.0 (no audit), 0.8 (conflict)
     - Uncertainty (C): f_C = 1.05 (<0.1), 1.0 (0.1-0.2), 0.9 (0.2-0.5), 0.75 (>0.5)
     - Consistency vs peers (E): f_E = 1.0 (|z|<1), 0.95 (1<|z|<2), 0.8 (|z|>=2)
   - Score: score_field_raw = base_score * f_A * f_B * f_C * f_E (truncated to 0-100)

2. Global weighting:
   - Each field i has materiality w_i (sum = 1)
   - overall_trust = Σ_i (score_field_i * w_i)

3. Persist and log:
   - Update trust_index and trust_meta_json in database
   - Log to R&D if change > 5 points
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from duplios.trust_index_models import (
    DataSourceType,
    FieldTrustMeta,
    DPPTrustResult,
)
from duplios.dpp_models import DppRecord

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# FIELD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Materiality weights (sum should be ~1.0)
FIELD_MATERIALITY_WEIGHTS: Dict[str, float] = {
    "carbon_footprint_kg_co2eq": 0.40,  # Carbon is most material
    "water_m3": 0.25,                    # Water usage
    "energy_kwh": 0.15,                  # Energy consumption
    "recycled_content_pct": 0.10,        # Recycled content
    "recyclability_pct": 0.10,           # Recyclability
    # Additional fields can be added with lower weights
}

# Field mappings from DPP model to field keys
FIELD_MAPPINGS: Dict[str, str] = {
    "carbon_kg_co2eq": "carbon_footprint_kg_co2eq",
    "water_m3": "water_m3",
    "energy_kwh": "energy_kwh",
    "recycled_content_pct": "recycled_content_pct",
    "recyclability_pct": "recyclability_pct",
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRUST INDEX SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class TrustIndexService:
    """
    Advanced Trust Index calculation service.
    
    Calculates field-level trust scores and aggregates to overall trust index.
    """
    
    def __init__(self):
        self.last_trust_indices: Dict[UUID, float] = {}  # Track for R&D logging
    
    def calculate_for_dpp(
        self,
        dpp: DppRecord,
        db_session: Optional[Session] = None,
    ) -> DPPTrustResult:
        """
        Calculate trust index for a DPP.
        
        Args:
            dpp: DppRecord instance
            db_session: Optional database session for persistence
        
        Returns:
            DPPTrustResult with overall score and field breakdown
        """
        # Convert dpp.id to UUID (handle both int and UUID)
        if isinstance(dpp.id, int):
            dpp_id = UUID(int=dpp.id)
        elif isinstance(dpp.id, str):
            dpp_id = UUID(dpp.id)
        else:
            dpp_id = dpp.id
        
        # Track previous value for R&D logging
        previous_trust = self.last_trust_indices.get(dpp_id, dpp.trust_index or 60.0)
        
        # Extract field data from DPP
        field_data = self._extract_field_data(dpp)
        
        # Calculate field-level scores
        field_scores: Dict[str, float] = {}
        field_metas: Dict[str, FieldTrustMeta] = {}
        
        for field_key, field_value in field_data.items():
            if field_value is None:
                continue
            
            # Determine base data source type
            base_class = self._infer_data_source_type(dpp, field_key, field_value)
            
            # Extract metadata
            meta = self._extract_field_metadata(dpp, field_key, base_class)
            
            # Calculate field score
            field_score = self._calculate_field_score(meta)
            meta.field_score = field_score
            
            field_scores[field_key] = field_score
            field_metas[field_key] = meta
        
        # Calculate overall trust index (weighted average)
        overall_trust = self._calculate_overall_trust(field_scores, field_metas)
        
        # Generate key messages for UI
        key_messages = self._generate_key_messages(field_metas)
        
        result = DPPTrustResult(
            dpp_id=dpp_id,
            overall_trust_index=round(overall_trust, 1),
            field_scores=field_scores,
            field_metas=field_metas,
            key_messages=key_messages,
        )
        
        # Persist to database
        if db_session:
            self._persist_result(dpp, result, db_session)
        
        # Log to R&D if significant change
        if abs(result.overall_trust_index - previous_trust) > 5.0:
            self._log_to_rd(dpp_id, previous_trust, result.overall_trust_index, result)
        
        # Update tracking
        self.last_trust_indices[dpp_id] = result.overall_trust_index
        
        return result
    
    def _extract_field_data(self, dpp: DppRecord) -> Dict[str, Any]:
        """Extract field data from DPP record."""
        data = {}
        
        # Map DPP columns to field keys
        if dpp.carbon_kg_co2eq is not None:
            data["carbon_footprint_kg_co2eq"] = dpp.carbon_kg_co2eq
        if dpp.water_m3 is not None:
            data["water_m3"] = dpp.water_m3
        if dpp.energy_kwh is not None:
            data["energy_kwh"] = dpp.energy_kwh
        if dpp.recycled_content_pct is not None:
            data["recycled_content_pct"] = dpp.recycled_content_pct
        if dpp.recyclability_pct is not None:
            data["recyclability_pct"] = dpp.recyclability_pct
        
        # Try to extract from additional_data JSON
        if dpp.additional_data:
            try:
                additional = json.loads(dpp.additional_data)
                # Look for trust metadata
                if "trust_meta" in additional:
                    # Additional fields can be extracted here
                    pass
            except (json.JSONDecodeError, TypeError):
                pass
        
        return data
    
    def _infer_data_source_type(
        self,
        dpp: DppRecord,
        field_key: str,
        field_value: Any,
    ) -> DataSourceType:
        """
        Infer data source type for a field.
        
        Logic:
        - If value is None or missing: DESCONHECIDO
        - If field has "measured" tag in metadata: MEDIDO
        - If field has "reported" tag: REPORTADO
        - If field is derived from LCA engine: ESTIMADO
        - Default: ESTIMADO (with high uncertainty)
        """
        # Check additional_data for metadata
        if dpp.additional_data:
            try:
                additional = json.loads(dpp.additional_data)
                trust_meta = additional.get("trust_meta", {})
                field_meta = trust_meta.get(field_key, {})
                
                # Check explicit tags
                if field_meta.get("measured", False):
                    return DataSourceType.MEDIDO
                if field_meta.get("reported", False):
                    return DataSourceType.REPORTADO
                if field_meta.get("estimated", False):
                    return DataSourceType.ESTIMADO
                
                # Check source field
                source = field_meta.get("source", "").upper()
                if "MEASURED" in source or "MEDIDO" in source:
                    return DataSourceType.MEDIDO
                if "REPORTED" in source or "REPORTADO" in source:
                    return DataSourceType.REPORTADO
                if "ESTIMATED" in source or "ESTIMADO" in source:
                    return DataSourceType.ESTIMADO
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Heuristic: If value is 0 or very small, might be missing
        if field_value is None or (isinstance(field_value, (int, float)) and field_value == 0):
            return DataSourceType.DESCONHECIDO
        
        # Default: ESTIMADO (with high uncertainty)
        # This is conservative - if we don't know, assume it's estimated
        return DataSourceType.ESTIMADO
    
    def _extract_field_metadata(
        self,
        dpp: DppRecord,
        field_key: str,
        base_class: DataSourceType,
    ) -> FieldTrustMeta:
        """Extract metadata for a field."""
        # Default values
        measured_fraction = 0.0
        reported_fraction = 0.0
        estimated_fraction = 0.0
        unknown_fraction = 0.0
        recency_days = 365  # Default: 1 year old
        third_party_verified = False
        uncertainty_relative = 0.3  # Default: 30% uncertainty
        consistency_zscore = None
        
        # Try to extract from additional_data
        if dpp.additional_data:
            try:
                additional = json.loads(dpp.additional_data)
                trust_meta = additional.get("trust_meta", {})
                field_meta = trust_meta.get(field_key, {})
                
                # Extract fractions
                measured_fraction = field_meta.get("measured_fraction", 0.0)
                reported_fraction = field_meta.get("reported_fraction", 0.0)
                estimated_fraction = field_meta.get("estimated_fraction", 0.0)
                unknown_fraction = field_meta.get("unknown_fraction", 0.0)
                
                # Normalize fractions
                total = measured_fraction + reported_fraction + estimated_fraction + unknown_fraction
                if total > 0:
                    measured_fraction /= total
                    reported_fraction /= total
                    estimated_fraction /= total
                    unknown_fraction /= total
                
                # Recency
                last_updated_str = field_meta.get("last_updated")
                if last_updated_str:
                    try:
                        last_updated = datetime.fromisoformat(last_updated_str.replace("Z", "+00:00"))
                        recency_days = (datetime.now(timezone.utc) - last_updated.replace(tzinfo=timezone.utc)).days
                    except (ValueError, AttributeError):
                        pass
                else:
                    # Use DPP updated_at if available
                    if hasattr(dpp, "updated_at") and dpp.updated_at:
                        recency_days = (datetime.now(timezone.utc) - dpp.updated_at.replace(tzinfo=timezone.utc)).days
                
                # Third-party verification
                third_party_verified = field_meta.get("third_party_verified", False)
                
                # Uncertainty
                uncertainty_relative = field_meta.get("uncertainty_relative", 0.3)
                
                # Consistency z-score
                consistency_zscore = field_meta.get("consistency_zscore")
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Set fractions based on base_class if not explicitly set
        if measured_fraction == 0 and reported_fraction == 0 and estimated_fraction == 0:
            if base_class == DataSourceType.MEDIDO:
                measured_fraction = 1.0
            elif base_class == DataSourceType.REPORTADO:
                reported_fraction = 1.0
            elif base_class == DataSourceType.ESTIMADO:
                estimated_fraction = 1.0
            else:
                unknown_fraction = 1.0
        
        # Materiality weight
        materiality_weight = FIELD_MATERIALITY_WEIGHTS.get(field_key, 0.05)  # Default: 5%
        
        return FieldTrustMeta(
            field_key=field_key,
            base_class=base_class,
            measured_fraction=measured_fraction,
            reported_fraction=reported_fraction,
            estimated_fraction=estimated_fraction,
            unknown_fraction=unknown_fraction,
            recency_days=recency_days,
            third_party_verified=third_party_verified,
            uncertainty_relative=uncertainty_relative,
            materiality_weight=materiality_weight,
            consistency_zscore=consistency_zscore,
            last_updated=datetime.now(timezone.utc),
        )
    
    def _calculate_field_score(self, meta: FieldTrustMeta) -> float:
        """
        Calculate trust score for a field.
        
        Algorithm:
        1. Base score based on data source type:
           - MEDIDO: 100
           - REPORTADO: 85
           - ESTIMADO: 65
           - DESCONHECIDO: 0
        
        2. Apply adjustment factors:
           - Recency (A): f_A = 1.0 (<1 year), 0.95 (1-2 years), 0.9 (2-3 years), 0.85 (>3 years)
           - Third-party verification (B): f_B = 1.1 (audited), 1.0 (no audit), 0.8 (conflict)
           - Uncertainty (C): f_C = 1.05 (<0.1), 1.0 (0.1-0.2), 0.9 (0.2-0.5), 0.75 (>0.5)
           - Consistency vs peers (E): f_E = 1.0 (|z|<1), 0.95 (1<|z|<2), 0.8 (|z|>=2)
        
        3. Score: score_field_raw = base_score * f_A * f_B * f_C * f_E
        4. Truncate to 0-100
        """
        # Base score
        if meta.base_class == DataSourceType.MEDIDO:
            base_score = 100.0
        elif meta.base_class == DataSourceType.REPORTADO:
            base_score = 85.0
        elif meta.base_class == DataSourceType.ESTIMADO:
            base_score = 65.0
        else:  # DESCONHECIDO
            base_score = 0.0
        
        # Adjustment factor A: Recency
        if meta.recency_days < 365:
            f_A = 1.0
        elif meta.recency_days < 730:  # 1-2 years
            f_A = 0.95
        elif meta.recency_days < 1095:  # 2-3 years
            f_A = 0.9
        else:  # >3 years
            f_A = 0.85
        
        # Adjustment factor B: Third-party verification
        if meta.third_party_verified:
            f_B = 1.1
        else:
            f_B = 1.0
        
        # Adjustment factor C: Uncertainty
        if meta.uncertainty_relative < 0.1:
            f_C = 1.05
        elif meta.uncertainty_relative < 0.2:
            f_C = 1.0
        elif meta.uncertainty_relative < 0.5:
            f_C = 0.9
        else:
            f_C = 0.75
        
        # Adjustment factor E: Consistency vs peers
        if meta.consistency_zscore is None:
            f_E = 1.0  # No data, no penalty
        else:
            abs_z = abs(meta.consistency_zscore)
            if abs_z < 1.0:
                f_E = 1.0
            elif abs_z < 2.0:
                f_E = 0.95
            else:
                f_E = 0.8
        
        # Calculate raw score
        score_field_raw = base_score * f_A * f_B * f_C * f_E
        
        # Truncate to 0-100
        score_field = max(0.0, min(100.0, score_field_raw))
        
        return round(score_field, 1)
    
    def _calculate_overall_trust(
        self,
        field_scores: Dict[str, float],
        field_metas: Dict[str, FieldTrustMeta],
    ) -> float:
        """
        Calculate overall trust index (weighted average).
        
        Formula: overall_trust = Σ_i (score_field_i * w_i)
        where w_i is materiality_weight for field i
        """
        if not field_scores:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field_key, score in field_scores.items():
            meta = field_metas.get(field_key)
            if meta:
                weight = meta.materiality_weight
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            # Fallback: simple average
            return sum(field_scores.values()) / len(field_scores) if field_scores else 0.0
        
        # Normalize by total weight
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return overall
    
    def _generate_key_messages(self, field_metas: Dict[str, FieldTrustMeta]) -> list[str]:
        """Generate simplified key messages for UI."""
        messages = []
        
        # Map field keys to Portuguese names
        field_names = {
            "carbon_footprint_kg_co2eq": "Carbono",
            "water_m3": "Água",
            "energy_kwh": "Energia",
            "recycled_content_pct": "Conteúdo Reciclado",
            "recyclability_pct": "Reciclabilidade",
        }
        
        for field_key, meta in field_metas.items():
            field_name = field_names.get(field_key, field_key)
            
            # Build message
            parts = []
            
            # Base type
            if meta.base_class == DataSourceType.MEDIDO:
                parts.append("base medido")
            elif meta.base_class == DataSourceType.REPORTADO:
                parts.append("base reportado")
            elif meta.base_class == DataSourceType.ESTIMADO:
                parts.append("base estimado")
            else:
                parts.append("base desconhecido")
            
            # Verification
            if meta.third_party_verified:
                parts.append("+ auditado")
            
            # Uncertainty warning
            if meta.uncertainty_relative > 0.3:
                parts.append("(alta incerteza)")
            
            message = f"{field_name}: {' + '.join(parts)}"
            messages.append(message)
        
        return messages
    
    def _persist_result(
        self,
        dpp: DppRecord,
        result: DPPTrustResult,
        db_session: Session,
    ) -> None:
        """Persist trust index result to database."""
        # Update trust_index column
        dpp.trust_index = result.overall_trust_index
        
        # Update trust_meta_json in additional_data
        if not dpp.additional_data:
            additional = {}
        else:
            try:
                additional = json.loads(dpp.additional_data)
            except (json.JSONDecodeError, TypeError):
                additional = {}
        
        # Store trust metadata
        if "trust_meta" not in additional:
            additional["trust_meta"] = {}
        
        # Store field-level metadata
        for field_key, meta in result.field_metas.items():
            additional["trust_meta"][field_key] = {
                "field_score": meta.field_score,
                "base_class": meta.base_class if isinstance(meta.base_class, str) else meta.base_class.value,
                "measured_fraction": meta.measured_fraction,
                "reported_fraction": meta.reported_fraction,
                "estimated_fraction": meta.estimated_fraction,
                "unknown_fraction": meta.unknown_fraction,
                "recency_days": meta.recency_days,
                "third_party_verified": meta.third_party_verified,
                "uncertainty_relative": meta.uncertainty_relative,
                "materiality_weight": meta.materiality_weight,
                "consistency_zscore": meta.consistency_zscore,
                "last_updated": meta.last_updated.isoformat() if meta.last_updated else None,
            }
        
        # Store overall result metadata
        additional["trust_index_meta"] = {
            "overall_trust_index": result.overall_trust_index,
            "calculated_at": result.calculated_at.isoformat(),
            "calculation_version": result.calculation_version,
            "field_scores": result.field_scores,
        }
        
        dpp.additional_data = json.dumps(additional, default=str)
        
        db_session.commit()
    
    def _log_to_rd(
        self,
        dpp_id: UUID,
        old_trust: float,
        new_trust: float,
        result: DPPTrustResult,
    ) -> None:
        """
        Log trust index evolution to R&D module.
        
        As specified: Log to WPX_TRUST_EVOLUTION experiment type.
        """
        try:
            # Import R&D module (graceful if not available)
            from rd.experiments_core import log_experiment_event
            
            # Determine cause
            cause = "unknown"
            if new_trust > old_trust:
                # Check which fields improved
                # For now, use a simple heuristic
                cause = "dados atualizados"
            else:
                cause = "dados removidos ou expirados"
            
            # Log event
            log_experiment_event(
                experiment_type="WPX_TRUST_EVOLUTION",
                event_data={
                    "dpp_id": str(dpp_id),
                    "trust_index_old": old_trust,
                    "trust_index_new": new_trust,
                    "change": new_trust - old_trust,
                    "cause": cause,
                    "field_scores": result.field_scores,
                    "timestamp": result.calculated_at.isoformat(),
                },
            )
            
            logger.info(f"Logged trust index evolution for DPP {dpp_id}: {old_trust} -> {new_trust}")
        except ImportError:
            # R&D module not available, skip logging
            logger.debug("R&D module not available, skipping trust index logging")
        except Exception as e:
            logger.warning(f"Failed to log trust index to R&D: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_service_instance: Optional[TrustIndexService] = None


def get_trust_index_service() -> TrustIndexService:
    """Get singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = TrustIndexService()
    return _service_instance

