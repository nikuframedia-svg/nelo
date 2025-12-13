"""
════════════════════════════════════════════════════════════════════════════════
COMPLIANCE RADAR - ESPR / CBAM / CSRD Compliance Analysis
════════════════════════════════════════════════════════════════════════════════

Contract D3 Implementation: Compliance Radar for Duplios DPP

Service:
- ComplianceRadarService: Analyzes DPP compliance with ESPR, CBAM, CSRD
- Generates compliance scores (0-100) per regulation
- Identifies missing fields by category
- Provides clear recommendations
- Integrates with R&D (WPX_COMPLIANCE_EVOLUTION)

Algorithm:
1. Load compliance_rules.yaml
2. For each regulation (ESPR, CBAM, CSRD):
   - Evaluate each compliance block
   - Check if required fields exist in DPP
   - Calculate score based on presence and severity
   - Generate item-level status
3. Aggregate scores and identify critical gaps
4. Generate recommended actions
5. Log to R&D if significant change
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.orm import Session

from duplios.dpp_models import DppRecord
from duplios.compliance_models import (
    RegulationType,
    ComplianceStatus,
    ComplianceItemStatus,
    ComplianceRadarResult,
)

logger = logging.getLogger(__name__)

# Try to import yaml
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not available, will use JSON fallback for compliance rules")


# ═══════════════════════════════════════════════════════════════════════════════
# RULES LOADING
# ═══════════════════════════════════════════════════════════════════════════════

COMPLIANCE_RULES_PATH = Path(__file__).parent / "data" / "compliance_rules.yaml"

_compliance_rules_cache: Optional[Dict[str, Any]] = None


def _load_compliance_rules() -> Dict[str, Any]:
    """Load compliance rules from YAML file."""
    global _compliance_rules_cache
    
    if _compliance_rules_cache is not None:
        return _compliance_rules_cache
    
    try:
        if COMPLIANCE_RULES_PATH.exists():
            with open(COMPLIANCE_RULES_PATH, 'r', encoding='utf-8') as f:
                if HAS_YAML:
                    _compliance_rules_cache = yaml.safe_load(f)
                else:
                    # Fallback: try JSON
                    content = f.read()
                    try:
                        _compliance_rules_cache = json.loads(content)
                    except json.JSONDecodeError:
                        _compliance_rules_cache = _get_default_rules()
        else:
            logger.warning(f"Compliance rules file not found: {COMPLIANCE_RULES_PATH}")
            _compliance_rules_cache = _get_default_rules()
    except Exception as e:
        logger.error(f"Failed to load compliance rules: {e}")
        _compliance_rules_cache = _get_default_rules()
    
    return _compliance_rules_cache


def _get_default_rules() -> Dict[str, Any]:
    """Get default compliance rules (fallback)."""
    return {
        "espr": {
            "identification": {
                "required": True,
                "fields": ["gtin", "manufacturer_name"],
                "severity": 3,
                "description": "Identificação do produto",
            },
            "environmental_core": {
                "required": True,
                "fields": ["carbon_kg_co2eq"],
                "severity": 3,
                "description": "Pegada de carbono",
            },
        },
        "cbam": {
            "applicable_categories": ["steel", "aluminium"],
            "embedded_emissions": {
                "required": True,
                "fields": ["carbon_kg_co2eq"],
                "severity": 3,
                "description": "Emissões incorporadas",
            },
        },
        "csrd": {
            "e1_climate": {
                "required": True,
                "fields": ["carbon_kg_co2eq"],
                "severity": 3,
                "description": "E1 - Mudanças climáticas",
            },
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE RADAR SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class ComplianceRadarService:
    """
    Compliance Radar service for Duplios DPP.
    
    As specified in Contract D3:
    - Analyzes DPP compliance with ESPR, CBAM, CSRD
    - Generates scores (0-100) per regulation
    - Identifies missing fields by category
    - Provides clear recommendations
    - Integrates with R&D
    """
    
    def __init__(self):
        self.rules = _load_compliance_rules()
        self.last_scores: Dict[UUID, Dict[str, float]] = {}  # Track for R&D logging
    
    def analyze_dpp(
        self,
        dpp: DppRecord,
        db_session: Optional[Session] = None,
    ) -> ComplianceRadarResult:
        """
        Analyze DPP compliance with ESPR, CBAM, CSRD.
        
        As specified in Contract D3:
        - Evaluates each regulation separately
        - Calculates scores based on presence and severity
        - Identifies critical gaps
        - Generates recommended actions
        
        Args:
            dpp: DppRecord instance
            db_session: Optional database session for R&D logging
        
        Returns:
            ComplianceRadarResult with scores and item-level status
        """
        dpp_id = UUID(int=dpp.id) if isinstance(dpp.id, int) else dpp.id
        
        # Track previous scores for R&D logging
        previous_scores = self.last_scores.get(dpp_id, {})
        
        # Extract DPP data
        dpp_data = self._extract_dpp_data(dpp)
        
        # Evaluate ESPR
        espr_score, espr_items = self._evaluate_regulation(
            "espr",
            dpp_data,
            RegulationType.ESPR,
        )
        
        # Evaluate CBAM (check if applicable first)
        cbam_score, cbam_items = self._evaluate_cbam(dpp_data)
        
        # Evaluate CSRD
        csrd_score, csrd_items = self._evaluate_regulation(
            "csrd",
            dpp_data,
            RegulationType.CSRD,
        )
        
        # Identify critical gaps
        critical_gaps = self._identify_critical_gaps(espr_items, cbam_items, csrd_items)
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(
            espr_items, cbam_items, csrd_items, critical_gaps
        )
        
        result = ComplianceRadarResult(
            dpp_id=dpp_id,
            espr_score=round(espr_score, 1),
            cbam_score=round(cbam_score, 1) if cbam_score is not None else None,
            csrd_score=round(csrd_score, 1),
            espr_items=espr_items,
            cbam_items=cbam_items,
            csrd_items=csrd_items,
            critical_gaps=critical_gaps,
            recommended_actions=recommended_actions,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )
        
        # Log to R&D if significant change
        if db_session:
            self._log_to_rd(dpp_id, previous_scores, result)
        
        # Update tracking
        self.last_scores[dpp_id] = {
            "espr": result.espr_score,
            "cbam": result.cbam_score or 0.0,
            "csrd": result.csrd_score,
        }
        
        return result
    
    def _extract_dpp_data(self, dpp: DppRecord) -> Dict[str, Any]:
        """Extract DPP data as dictionary."""
        data = {
            "gtin": dpp.gtin,
            "product_name": dpp.product_name,
            "product_category": dpp.product_category,
            "manufacturer_name": dpp.manufacturer_name,
            "country_of_origin": dpp.country_of_origin,
            "carbon_kg_co2eq": dpp.carbon_kg_co2eq,
            "water_m3": dpp.water_m3,
            "energy_kwh": dpp.energy_kwh,
            "recycled_content_pct": dpp.recycled_content_pct,
            "recyclability_pct": dpp.recyclability_pct,
            "durability_score": dpp.durability_score,
            "reparability_score": dpp.reparability_score,
        }
        
        # Extract from additional_data JSON
        if dpp.additional_data:
            try:
                additional = json.loads(dpp.additional_data)
                # Merge additional data (with priority to direct fields)
                for key, value in additional.items():
                    if key not in data or data[key] is None:
                        data[key] = value
                
                # Extract nested fields
                if "materials" in additional:
                    data["materials"] = additional["materials"]
                if "certifications" in additional:
                    data["certifications"] = additional["certifications"]
                if "third_party_audits" in additional:
                    data["third_party_audits"] = additional["third_party_audits"]
            except (json.JSONDecodeError, TypeError):
                pass
        
        return data
    
    def _evaluate_regulation(
        self,
        regulation_key: str,
        dpp_data: Dict[str, Any],
        regulation_type: RegulationType,
    ) -> tuple[float, List[ComplianceItemStatus]]:
        """
        Evaluate compliance for a regulation.
        
        Algorithm:
        - For each block in regulation rules:
          - Check if all required fields exist and are not empty
          - Create ComplianceItemStatus
        - Calculate score:
          - For each item:
            - If required and present: contributes severity * 1.0
            - If required and missing: contributes 0
            - If optional and present: contributes severity * 0.5
          - Normalize to 0-100
        """
        regulation_rules = self.rules.get(regulation_key, {})
        items: List[ComplianceItemStatus] = []
        total_weight = 0.0
        achieved_weight = 0.0
        
        # Iterate over compliance blocks (skip applicable_categories for CBAM)
        for block_key, block_config in regulation_rules.items():
            if block_key == "applicable_categories":
                continue
            
            if not isinstance(block_config, dict):
                continue
            
            required = block_config.get("required", False)
            fields = block_config.get("fields", [])
            severity = block_config.get("severity", 1)
            description = block_config.get("description", block_key)
            
            # Check if fields are present
            present = self._check_fields_present(dpp_data, fields)
            
            # Calculate weight contribution
            if required:
                weight = severity * 1.0
                if present:
                    achieved_weight += weight
            else:
                weight = severity * 0.5
                if present:
                    achieved_weight += weight * 1.0  # Full credit for optional
            
            total_weight += weight
            
            # Create item status
            item = ComplianceItemStatus(
                key=f"{regulation_key}.{block_key}",
                description=description,
                required=required,
                present=present,
                severity=severity,
                notes=None if present else f"Faltam campos: {', '.join(self._get_missing_fields(dpp_data, fields))}",
            )
            items.append(item)
        
        # Calculate score (0-100)
        score = (achieved_weight / total_weight * 100.0) if total_weight > 0 else 0.0
        
        return score, items
    
    def _evaluate_cbam(
        self,
        dpp_data: Dict[str, Any],
    ) -> tuple[Optional[float], List[ComplianceItemStatus]]:
        """
        Evaluate CBAM compliance.
        
        First checks if product category is applicable.
        If not, returns None score.
        """
        cbam_rules = self.rules.get("cbam", {})
        applicable_categories = cbam_rules.get("applicable_categories", [])
        
        # Check if category is applicable
        category = (dpp_data.get("product_category") or "").lower()
        is_applicable = False
        
        for app_cat in applicable_categories:
            if app_cat.lower() in category or category in app_cat.lower():
                is_applicable = True
                break
        
        if not is_applicable:
            # CBAM not applicable for this product
            return None, []
        
        # Evaluate CBAM compliance
        score, items = self._evaluate_regulation("cbam", dpp_data, RegulationType.CBAM)
        
        return score, items
    
    def _check_fields_present(self, dpp_data: Dict[str, Any], fields: List[str]) -> bool:
        """Check if at least one field in the list is present and not empty."""
        for field in fields:
            value = dpp_data.get(field)
            if value is not None:
                # Check if value is meaningful
                if isinstance(value, (list, dict)):
                    if len(value) > 0:
                        return True
                elif isinstance(value, str):
                    if value.strip():
                        return True
                elif isinstance(value, (int, float)):
                    # For numeric fields, 0 might be valid (e.g., carbon=0)
                    return True
                else:
                    return True
        return False
    
    def _get_missing_fields(self, dpp_data: Dict[str, Any], fields: List[str]) -> List[str]:
        """Get list of missing fields."""
        missing = []
        for field in fields:
            value = dpp_data.get(field)
            if value is None:
                missing.append(field)
            elif isinstance(value, (list, dict)) and len(value) == 0:
                missing.append(field)
            elif isinstance(value, str) and not value.strip():
                missing.append(field)
        return missing
    
    def _identify_critical_gaps(
        self,
        espr_items: List[ComplianceItemStatus],
        cbam_items: List[ComplianceItemStatus],
        csrd_items: List[ComplianceItemStatus],
    ) -> List[str]:
        """Identify critical gaps (severity=3, missing, required)."""
        gaps = []
        
        for item in espr_items + cbam_items + csrd_items:
            if item.required and not item.present and item.severity == 3:
                gaps.append(f"{item.key.split('.')[0].upper()}: {item.description}")
        
        return gaps
    
    def _generate_recommended_actions(
        self,
        espr_items: List[ComplianceItemStatus],
        cbam_items: List[ComplianceItemStatus],
        csrd_items: List[ComplianceItemStatus],
        critical_gaps: List[str],
    ) -> List[str]:
        """Generate recommended actions ordered by severity."""
        actions = []
        
        # Start with critical gaps
        for gap in critical_gaps:
            actions.append(f"Preencher: {gap}")
        
        # Add other missing required items
        for item in espr_items + cbam_items + csrd_items:
            if item.required and not item.present and item.severity < 3:
                actions.append(f"Preencher: {item.description}")
        
        # Add optional items that would improve score
        for item in espr_items + cbam_items + csrd_items:
            if not item.required and not item.present and item.severity >= 2:
                actions.append(f"Recomendado: {item.description}")
        
        return actions[:5]  # Limit to top 5
    
    def _log_to_rd(
        self,
        dpp_id: UUID,
        previous_scores: Dict[str, float],
        result: ComplianceRadarResult,
    ) -> None:
        """Log compliance evolution to R&D module."""
        try:
            from rd.experiments_core import log_experiment_event
            
            # Check if significant change (>10 points in any score)
            significant_change = False
            if previous_scores:
                if abs(result.espr_score - previous_scores.get("espr", 0)) > 10:
                    significant_change = True
                if result.cbam_score and abs(result.cbam_score - previous_scores.get("cbam", 0)) > 10:
                    significant_change = True
                if abs(result.csrd_score - previous_scores.get("csrd", 0)) > 10:
                    significant_change = True
            else:
                significant_change = True  # First analysis
            
            if significant_change:
                log_experiment_event(
                    experiment_type="WPX_COMPLIANCE_EVOLUTION",
                    event_data={
                        "dpp_id": str(dpp_id),
                        "espr_score_old": previous_scores.get("espr", 0.0),
                        "espr_score_new": result.espr_score,
                        "cbam_score_old": previous_scores.get("cbam", 0.0),
                        "cbam_score_new": result.cbam_score,
                        "csrd_score_old": previous_scores.get("csrd", 0.0),
                        "csrd_score_new": result.csrd_score,
                        "critical_gaps": result.critical_gaps,
                        "timestamp": result.analyzed_at,
                    },
                )
                
                logger.info(f"Logged compliance evolution for DPP {dpp_id}")
        except ImportError:
            logger.debug("R&D module not available, skipping compliance logging")
        except Exception as e:
            logger.warning(f"Failed to log compliance to R&D: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_service_instance: Optional[ComplianceRadarService] = None


def get_compliance_radar_service() -> ComplianceRadarService:
    """Get singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ComplianceRadarService()
    return _service_instance


