"""
════════════════════════════════════════════════════════════════════════════════
GAP FILLING LITE - Simple Gap Filling for Duplios DPP
════════════════════════════════════════════════════════════════════════════════

Contract D2 Implementation: Gap Filling Lite without Ecoinvent

Features:
- Fills missing environmental fields using internal factor tables
- Applies contextual adjustments (country, energy, technology age)
- Returns values + uncertainty + "source = estimated" for Trust Index
- Automatic hook on DPP create/update
- Integration with Trust Index (Contract D1) and R&D

Algorithm:
1. Read product composition from DPP
2. For each material: get base factors, calculate CO2/water
3. Apply contextual adjustments (country, tech age)
4. Sum totals and calculate uncertainty
5. Update DPP with filled values and metadata
6. Recalculate Trust Index
7. Log to R&D (WPX_GAPFILL_LITE)

R&D / SIFIDE: WPX_GAPFILL_LITE - Gap filling experiments
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from duplios.dpp_models import DppRecord
from duplios.trust_index_service import get_trust_index_service

logger = logging.getLogger(__name__)

# Try to import yaml, fallback to JSON if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logger.warning("PyYAML not available, will use JSON fallback for gap factors")


# ═══════════════════════════════════════════════════════════════════════════════
# FACTOR DATABASE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

GAP_FACTORS_PATH = Path(__file__).parent / "data" / "gap_factors.yaml"

_gap_factors_cache: Optional[Dict[str, Any]] = None


def _get_default_factors() -> Dict[str, Any]:
    """Get default gap filling factors (fallback)."""
    return {
        "materials": {
            "steel": {
                "base_co2_kg_per_kg": 1.8,
                "base_water_m3_per_kg": 0.5,
                "base_recyclability": 0.95,
                "base_energy_kwh_per_kg": 6.0,
            },
            "polypropylene": {
                "base_co2_kg_per_kg": 1.9,
                "base_water_m3_per_kg": 0.3,
                "base_recyclability": 0.60,
                "base_energy_kwh_per_kg": 4.0,
            },
            "cotton": {
                "base_co2_kg_per_kg": 4.0,
                "base_water_m3_per_kg": 7.0,
                "base_recyclability": 0.70,
                "base_energy_kwh_per_kg": 2.0,
            },
            "default": {
                "base_co2_kg_per_kg": 2.5,
                "base_water_m3_per_kg": 1.0,
                "base_recyclability": 0.50,
                "base_energy_kwh_per_kg": 5.0,
            },
        },
        "countries": {
            "DEFAULT": {"energy_co2_factor_vs_eu": 1.0},
            "PT": {"energy_co2_factor_vs_eu": 0.6},
            "PL": {"energy_co2_factor_vs_eu": 1.8},
            "DE": {"energy_co2_factor_vs_eu": 0.9},
        },
        "tech_age_adjustment": {"young": 1.0, "mid": 1.1, "old": 1.3},
    }


def _load_gap_factors() -> Dict[str, Any]:
    """Load gap filling factors from YAML file."""
    global _gap_factors_cache
    
    if _gap_factors_cache is not None:
        return _gap_factors_cache
    
    try:
        if GAP_FACTORS_PATH.exists():
            with open(GAP_FACTORS_PATH, 'r', encoding='utf-8') as f:
                if HAS_YAML:
                    _gap_factors_cache = yaml.safe_load(f)
                else:
                    # Fallback: try to parse as JSON (if someone converts YAML to JSON)
                    content = f.read()
                    try:
                        _gap_factors_cache = json.loads(content)
                    except json.JSONDecodeError:
                        # Last resort: use minimal default
                        logger.warning("Could not parse gap factors file, using defaults")
                        _gap_factors_cache = _get_default_factors()
        else:
            logger.warning(f"Gap factors file not found: {GAP_FACTORS_PATH}, using defaults")
            _gap_factors_cache = _get_default_factors()
    except Exception as e:
        logger.error(f"Failed to load gap factors: {e}")
        _gap_factors_cache = _get_default_factors()
    
    return _gap_factors_cache


# ═══════════════════════════════════════════════════════════════════════════════
# GAP FILLING LITE SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class GapFillingLiteService:
    """
    Lite Gap Filling service for Duplios DPP.
    
    As specified in Contract D2:
    - Uses internal factor tables (no Ecoinvent dependency)
    - Applies contextual adjustments (country, energy, tech age)
    - Returns values + uncertainty + source metadata
    - Integrates with Trust Index and R&D
    """
    
    def __init__(self):
        self.factors = _load_gap_factors()
        self.default_uncertainty = 0.3  # ±30% for simple estimates
    
    def fill_for_dpp(
        self,
        dpp: DppRecord,
        db_session: Optional[Session] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Fill missing environmental fields for a DPP.
        
        As specified in Contract D2:
        - Reads composition from DPP
        - Calculates CO2, water, recyclability using material factors
        - Applies country and tech age adjustments
        - Updates DPP with filled values and metadata
        - Recalculates Trust Index
        
        Args:
            dpp: DppRecord instance
            db_session: Optional database session for persistence
            force: If True, overwrite existing values (default: False, only fill missing)
        
        Returns:
            Dict with:
            - filled_fields: List of field names that were filled
            - values: Dict of filled values
            - uncertainty: Dict of uncertainty values
            - source: "estimated_lite"
        """
        result = {
            "filled_fields": [],
            "values": {},
            "uncertainty": {},
            "source": "estimated_lite",
            "context": {},
        }
        
        # Step 1: Extract composition from DPP
        composition = self._extract_composition(dpp)
        if not composition:
            logger.debug(f"No composition found for DPP {dpp.id}, skipping gap fill")
            return result
        
        # Step 2: Calculate base values per material
        co2_total = 0.0
        water_total = 0.0
        energy_total = 0.0
        recyclability_weighted = 0.0
        total_mass = 0.0
        
        for material_name, mass_kg in composition.items():
            factors = self._get_material_factors(material_name)
            
            # Base calculations
            co2_m = factors["base_co2_kg_per_kg"] * mass_kg
            water_m = factors["base_water_m3_per_kg"] * mass_kg
            energy_m = factors.get("base_energy_kwh_per_kg", 5.0) * mass_kg
            recyclability_m = factors["base_recyclability"]
            
            co2_total += co2_m
            water_total += water_m
            energy_total += energy_m
            recyclability_weighted += recyclability_m * mass_kg
            total_mass += mass_kg
        
        # Step 3: Apply contextual adjustments
        country = dpp.country_of_origin or "DEFAULT"
        country_factor = self._get_country_factor(country)
        
        # Tech age adjustment (if available)
        tech_age_factor = self._get_tech_age_factor(dpp)
        
        # Apply adjustments to CO2 (energy-related)
        co2_total_adjusted = co2_total * country_factor * tech_age_factor
        
        # Recyclability: weighted average
        recyclability_estimated = recyclability_weighted / total_mass if total_mass > 0 else 0.5
        
        # Step 4: Determine uncertainty
        uncertainty_co2 = self.default_uncertainty
        uncertainty_water = self.default_uncertainty
        uncertainty_recyclability = self.default_uncertainty
        
        # Higher uncertainty if composition is incomplete or country unknown
        if country == "DEFAULT":
            uncertainty_co2 += 0.1
            uncertainty_water += 0.1
        
        # Step 5: Update DPP (only if fields are missing or force=True)
        filled_fields = []
        
        if (dpp.carbon_kg_co2eq is None or dpp.carbon_kg_co2eq == 0.0) or force:
            if force or dpp.carbon_kg_co2eq is None or dpp.carbon_kg_co2eq == 0.0:
                dpp.carbon_kg_co2eq = co2_total_adjusted
                filled_fields.append("carbon_kg_co2eq")
                result["values"]["carbon_kg_co2eq"] = co2_total_adjusted
                result["uncertainty"]["carbon_kg_co2eq"] = uncertainty_co2
        
        if (dpp.water_m3 is None or dpp.water_m3 == 0.0) or force:
            if force or dpp.water_m3 is None or dpp.water_m3 == 0.0:
                dpp.water_m3 = water_total
                filled_fields.append("water_m3")
                result["values"]["water_m3"] = water_total
                result["uncertainty"]["water_m3"] = uncertainty_water
        
        if (dpp.energy_kwh is None or dpp.energy_kwh == 0.0) or force:
            if force or dpp.energy_kwh is None or dpp.energy_kwh == 0.0:
                dpp.energy_kwh = energy_total
                filled_fields.append("energy_kwh")
                result["values"]["energy_kwh"] = energy_total
                result["uncertainty"]["energy_kwh"] = uncertainty_co2  # Similar to CO2
        
        if (dpp.recyclability_pct is None or dpp.recyclability_pct == 0.0) or force:
            if force or dpp.recyclability_pct is None or dpp.recyclability_pct == 0.0:
                dpp.recyclability_pct = recyclability_estimated * 100  # Convert to percentage
                filled_fields.append("recyclability_pct")
                result["values"]["recyclability_pct"] = recyclability_estimated * 100
                result["uncertainty"]["recyclability_pct"] = uncertainty_recyclability
        
        result["filled_fields"] = filled_fields
        
        # Step 6: Update metadata in additional_data
        if db_session and filled_fields:
            self._update_dpp_metadata(dpp, result, db_session)
            
            # Step 7: Recalculate Trust Index
            try:
                trust_service = get_trust_index_service()
                trust_service.calculate_for_dpp(dpp, db_session=db_session)
            except Exception as e:
                logger.warning(f"Failed to recalculate Trust Index: {e}")
            
            # Step 8: Log to R&D
            self._log_to_rd(dpp, result)
        
        result["context"] = {
            "country": country,
            "country_factor": country_factor,
            "tech_age_factor": tech_age_factor,
            "materials_used": list(composition.keys()),
            "total_mass_kg": total_mass,
        }
        
        return result
    
    def _extract_composition(self, dpp: DppRecord) -> Dict[str, float]:
        """
        Extract material composition from DPP.
        
        Tries multiple sources:
        1. additional_data.materials (list with material_name and mass_kg)
        2. additional_data.composition (dict with material -> mass)
        3. BOM from PDM (if linked)
        
        Returns:
            Dict mapping material_name -> mass_kg
        """
        composition = {}
        
        # Try additional_data JSON
        if dpp.additional_data:
            try:
                additional = json.loads(dpp.additional_data)
                
                # Format 1: materials list
                if "materials" in additional:
                    for mat in additional["materials"]:
                        name = mat.get("material_name", mat.get("name", "")).lower().replace(" ", "_")
                        mass = mat.get("mass_kg", mat.get("weight_kg", 0.0))
                        if name and mass > 0:
                            composition[name] = composition.get(name, 0.0) + mass
                
                # Format 2: composition dict
                if "composition" in additional:
                    for name, mass in additional["composition"].items():
                        name_clean = name.lower().replace(" ", "_")
                        if isinstance(mass, (int, float)) and mass > 0:
                            composition[name_clean] = composition.get(name_clean, 0.0) + mass
                
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logger.debug(f"Failed to parse additional_data: {e}")
        
        # If no composition found, try to infer from product category
        if not composition:
            # Heuristic: use default material based on category
            category = (dpp.product_category or "").lower()
            if "textile" in category or "têxtil" in category:
                # Assume 1kg for estimation
                composition["cotton"] = 1.0
            elif "metal" in category or "metálico" in category:
                composition["steel"] = 1.0
            elif "plastic" in category or "plástico" in category:
                composition["polypropylene"] = 1.0
            elif "electronic" in category or "eletrónico" in category:
                composition["pcb"] = 1.0
            else:
                # Default: use generic material
                composition["default"] = 1.0
        
        return composition
    
    def _get_material_factors(self, material_name: str) -> Dict[str, float]:
        """Get material factors from database."""
        materials = self.factors.get("materials", {})
        
        # Try exact match
        if material_name in materials:
            return materials[material_name]
        
        # Try normalized name
        normalized = material_name.lower().replace(" ", "_").replace("-", "_")
        if normalized in materials:
            return materials[normalized]
        
        # Try partial match
        for key, factors in materials.items():
            if key in normalized or normalized in key:
                return factors
        
        # Default fallback
        default = materials.get("default", {
            "base_co2_kg_per_kg": 2.5,
            "base_water_m3_per_kg": 1.0,
            "base_recyclability": 0.50,
            "base_energy_kwh_per_kg": 5.0,
        })
        
        logger.debug(f"Using default factors for material: {material_name}")
        return default
    
    def _get_country_factor(self, country_code: str) -> float:
        """Get country energy CO2 adjustment factor."""
        countries = self.factors.get("countries", {})
        
        # Try exact match
        if country_code.upper() in countries:
            return countries[country_code.upper()].get("energy_co2_factor_vs_eu", 1.0)
        
        # Default
        default = countries.get("DEFAULT", {"energy_co2_factor_vs_eu": 1.0})
        return default.get("energy_co2_factor_vs_eu", 1.0)
    
    def _get_tech_age_factor(self, dpp: DppRecord) -> float:
        """Get technology age adjustment factor."""
        tech_adjustments = self.factors.get("tech_age_adjustment", {})
        
        # Try to extract year from additional_data or use default
        if dpp.additional_data:
            try:
                additional = json.loads(dpp.additional_data)
                year_machine = additional.get("year_machine")
                if year_machine:
                    current_year = datetime.now().year
                    age = current_year - year_machine
                    
                    if age < 5:
                        return tech_adjustments.get("young", 1.0)
                    elif age < 15:
                        return tech_adjustments.get("mid", 1.1)
                    else:
                        return tech_adjustments.get("old", 1.3)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        
        # Default: assume modern (young)
        return tech_adjustments.get("young", 1.0)
    
    def _update_dpp_metadata(
        self,
        dpp: DppRecord,
        result: Dict[str, Any],
        db_session: Session,
    ) -> None:
        """Update DPP metadata with gap filling information."""
        if not dpp.additional_data:
            additional = {}
        else:
            try:
                additional = json.loads(dpp.additional_data)
            except (json.JSONDecodeError, TypeError):
                additional = {}
        
        # Store gap filling metadata
        if "gap_filling" not in additional:
            additional["gap_filling"] = {}
        
        gap_meta = additional["gap_filling"]
        gap_meta["last_filled_at"] = datetime.now(timezone.utc).isoformat()
        gap_meta["method"] = "lite"
        gap_meta["filled_fields"] = result["filled_fields"]
        gap_meta["uncertainty"] = result["uncertainty"]
        gap_meta["context"] = result["context"]
        
        # Update trust_meta for each filled field
        if "trust_meta" not in additional:
            additional["trust_meta"] = {}
        
        trust_meta = additional["trust_meta"]
        for field_key in result["filled_fields"]:
            if field_key not in trust_meta:
                trust_meta[field_key] = {}
            
            trust_meta[field_key].update({
                "source": "estimated_lite",
                "estimated": True,
                "uncertainty_relative": result["uncertainty"].get(field_key, 0.3),
                "gap_filled_at": datetime.now(timezone.utc).isoformat(),
            })
        
        dpp.additional_data = json.dumps(additional, default=str)
        db_session.commit()
    
    def _log_to_rd(self, dpp: DppRecord, result: Dict[str, Any]) -> None:
        """Log gap filling to R&D module."""
        try:
            from rd.experiments_core import log_experiment_event
            
            log_experiment_event(
                experiment_type="WPX_GAPFILL_LITE",
                event_data={
                    "dpp_id": str(dpp.id),
                    "filled_fields": result["filled_fields"],
                    "values": result["values"],
                    "uncertainty": result["uncertainty"],
                    "context": result["context"],
                    "method": "lite",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            
            logger.info(f"Logged gap filling lite for DPP {dpp.id}: {len(result['filled_fields'])} fields filled")
        except ImportError:
            logger.debug("R&D module not available, skipping gap filling log")
        except Exception as e:
            logger.warning(f"Failed to log gap filling to R&D: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_service_instance: Optional[GapFillingLiteService] = None


def get_gap_filling_lite_service() -> GapFillingLiteService:
    """Get singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = GapFillingLiteService()
    return _service_instance

