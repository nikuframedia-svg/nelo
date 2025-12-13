"""
════════════════════════════════════════════════════════════════════════════════════════════════════
DPP MODELS - Digital Product Passport for Duplios
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 5 Implementation: DPP + LCA Basic + Digital Identity

Models:
- DppRecord: Digital Product Passport linked to ItemRevision
- DigitalIdentity: RFID/QR/unique ID tracking with hash verification

This extends the existing DPP system with proper revision-level tracking.
"""

from __future__ import annotations

import enum
import hashlib
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Enum, Float, ForeignKey, Integer, 
    String, Text, Index
)
from sqlalchemy.orm import relationship

from duplios.models import Base, engine


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class DppStatus(str, enum.Enum):
    """Status of a DPP record."""
    DRAFT = "DRAFT"         # Being prepared
    VALIDATED = "VALIDATED" # Internally validated
    PUBLISHED = "PUBLISHED" # Publicly accessible


class VerificationStatus(str, enum.Enum):
    """Status of digital identity verification."""
    UNVERIFIED = "UNVERIFIED"   # Not yet verified
    VERIFIED = "VERIFIED"       # Successfully verified
    CONFLICT = "CONFLICT"       # Multiple records with same ID
    DUPLICATE = "DUPLICATE"     # Duplicate entry detected


# ═══════════════════════════════════════════════════════════════════════════════
# DPP RECORD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DppRecord(Base):
    """
    Digital Product Passport record linked to an ItemRevision.
    
    Contains sustainability data, compliance info, and LCA results.
    """
    __tablename__ = "dpp_records"
    
    id = Column(Integer, primary_key=True, index=True)
    item_revision_id = Column(Integer, 
                              ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                              nullable=False, 
                              unique=True)
    
    # Product identification
    gtin = Column(String(50), nullable=True, index=True)  # Global Trade Item Number
    product_name = Column(String(255), nullable=False)
    product_category = Column(String(100), nullable=True)
    
    # Manufacturer info
    manufacturer_name = Column(String(255), nullable=True)
    country_of_origin = Column(String(100), nullable=True)
    
    # LCA / Environmental data (from compute_simple_lca)
    carbon_kg_co2eq = Column(Float, default=0.0)       # Carbon footprint
    water_m3 = Column(Float, default=0.0)              # Water usage
    energy_kwh = Column(Float, default=0.0)            # Energy consumption
    
    # Circularity metrics
    recycled_content_pct = Column(Float, default=0.0)  # % recycled input
    recyclability_pct = Column(Float, default=0.0)     # % recyclable output
    
    # Durability & Reparability
    durability_score = Column(Integer, default=5)      # 1-10 scale
    reparability_score = Column(Integer, default=5)    # 1-10 scale
    
    # Trust & Completeness
    trust_index = Column(Float, default=60.0)          # 0-100 trust score
    data_completeness_pct = Column(Float, default=0.0) # % of fields filled
    
    # Status and publishing
    status = Column(Enum(DppStatus), nullable=False, default=DppStatus.DRAFT)
    qr_public_url = Column(String(500), nullable=True)
    
    # Additional data (JSON for extensibility)
    additional_data = Column(Text, nullable=True)  # JSON
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_lca_calculation_at = Column(DateTime, nullable=True)
    
    # Relationships
    item_revision = relationship("ItemRevision", back_populates="dpp_records")
    
    __table_args__ = (
        Index('ix_dpp_gtin', 'gtin'),
        Index('ix_dpp_status', 'status'),
    )
    
    def compute_data_completeness(self) -> float:
        """Calculate percentage of important fields that are filled."""
        fields_to_check = [
            self.gtin,
            self.product_name,
            self.product_category,
            self.manufacturer_name,
            self.country_of_origin,
            self.carbon_kg_co2eq > 0,
            self.recyclability_pct > 0,
        ]
        filled = sum(1 for f in fields_to_check if f)
        return (filled / len(fields_to_check)) * 100
    
    def compute_trust_index(self) -> float:
        """
        Calculate trust index based on data quality.
        
        Factors:
        - Data completeness (40%)
        - Has GTIN (20%)
        - Has LCA data (20%)
        - Has manufacturer info (10%)
        - Has reparability info (10%)
        """
        score = 0.0
        
        # Data completeness (40%)
        completeness = self.compute_data_completeness()
        score += (completeness / 100) * 40
        
        # Has GTIN (20%)
        if self.gtin:
            score += 20
        
        # Has LCA data (20%)
        if self.carbon_kg_co2eq > 0 or self.energy_kwh > 0:
            score += 20
        
        # Has manufacturer info (10%)
        if self.manufacturer_name and self.country_of_origin:
            score += 10
        
        # Has durability/reparability scores (10%)
        if self.durability_score and self.reparability_score:
            score += 10
        
        return min(100.0, score)
    
    def __repr__(self):
        return f"<DppRecord(id={self.id}, revision_id={self.item_revision_id}, status={self.status.value})>"


# ═══════════════════════════════════════════════════════════════════════════════
# DIGITAL IDENTITY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class DigitalIdentity(Base):
    """
    Digital identity for traceability.
    
    Links physical identifiers (RFID, QR, serial) to item revisions.
    Provides hash-based verification for authenticity.
    """
    __tablename__ = "digital_identities"
    
    id = Column(Integer, primary_key=True, index=True)
    item_revision_id = Column(Integer, 
                              ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                              nullable=False)
    
    # Physical identifier
    unique_item_id = Column(String(255), nullable=False, index=True)  # RFID, QR, serial, etc.
    
    # Identity verification
    identity_hash = Column(String(64), nullable=False)  # SHA256 hash
    
    # Blockchain/ledger (optional future use)
    blockchain_tx_id = Column(String(255), nullable=True)
    ledger_network = Column(String(100), nullable=True)  # e.g., "ethereum", "hyperledger"
    
    # Lineage tracking
    lineage_parent_id = Column(Integer, 
                               ForeignKey("digital_identities.id", ondelete="SET NULL"), 
                               nullable=True)
    
    # Verification status
    verification_status = Column(Enum(VerificationStatus), 
                                 nullable=False, 
                                 default=VerificationStatus.UNVERIFIED)
    last_verified_at = Column(DateTime, nullable=True)
    verification_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    item_revision = relationship("ItemRevision", back_populates="digital_identities")
    lineage_parent = relationship("DigitalIdentity", remote_side=[id])
    
    __table_args__ = (
        Index('ix_identity_unique_id', 'unique_item_id'),
        Index('ix_identity_hash', 'identity_hash'),
        Index('ix_identity_status', 'verification_status'),
    )
    
    @staticmethod
    def compute_hash(revision_id: int, unique_item_id: str, timestamp: datetime) -> str:
        """Compute SHA256 hash for identity verification."""
        data = f"{revision_id}:{unique_item_id}:{timestamp.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()
    
    def __repr__(self):
        return f"<DigitalIdentity(id={self.id}, unique_id={self.unique_item_id}, status={self.verification_status.value})>"


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCT CONFORMANCE SNAPSHOT (for Contract 6 - XAI-DT)
# ═══════════════════════════════════════════════════════════════════════════════

class ConformityStatus(str, enum.Enum):
    """Conformity status of a product scan."""
    IN_TOLERANCE = "IN_TOLERANCE"
    OUT_OF_TOLERANCE = "OUT_OF_TOLERANCE"
    CRITICAL = "CRITICAL"


class ProductConformanceSnapshot(Base):
    """
    Record of a product conformance analysis (scan vs CAD).
    
    Used by Digital Twin Product / XAI-DT functionality.
    """
    __tablename__ = "product_conformance_snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    revision_id = Column(Integer, 
                        ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                        nullable=False)
    
    scan_id = Column(String(255), nullable=False)  # Reference to scan data
    
    # Deviation metrics
    max_dev = Column(Float, default=0.0)
    mean_dev = Column(Float, default=0.0)
    rms_dev = Column(Float, default=0.0)
    
    # XAI score
    scalar_error_score = Column(Float, default=0.0)  # 0-100
    
    # Status
    conformity_status = Column(Enum(ConformityStatus), 
                               nullable=False, 
                               default=ConformityStatus.IN_TOLERANCE)
    
    # Explanation (JSON from XAI-DT)
    explanation = Column(Text, nullable=True)  # JSON
    
    # Context
    machine_id = Column(String(100), nullable=True)
    operator_id = Column(String(100), nullable=True)
    
    # Process parameters at time of scan
    process_params = Column(Text, nullable=True)  # JSON
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    item_revision = relationship("ItemRevision", foreign_keys=[revision_id])
    
    __table_args__ = (
        Index('ix_conformance_revision', 'revision_id'),
        Index('ix_conformance_scan', 'scan_id'),
    )
    
    def __repr__(self):
        return f"<ProductConformanceSnapshot(scan={self.scan_id}, score={self.scalar_error_score}, status={self.conformity_status.value})>"


# ═══════════════════════════════════════════════════════════════════════════════
# GOLDEN RUN MODEL (for Contract 6 - Process Optimization)
# ═══════════════════════════════════════════════════════════════════════════════

class GoldenRun(Base):
    """
    Record of optimal process parameters (golden run).
    
    Used for suggesting best process parameters.
    """
    __tablename__ = "golden_runs"
    
    id = Column(Integer, primary_key=True, index=True)
    revision_id = Column(Integer, 
                        ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                        nullable=False)
    operation_id = Column(Integer, 
                         ForeignKey("pdm_routing_operations.id", ondelete="CASCADE"), 
                         nullable=True)
    machine_id = Column(String(100), nullable=True)
    
    # Process parameters (JSON)
    process_params = Column(Text, nullable=False)  # JSON
    
    # KPIs (JSON)
    kpis = Column(Text, nullable=True)  # JSON: {cycle_time, scrap_rate, energy, quality_score}
    
    # Overall score (0-100)
    score = Column(Float, default=0.0)
    
    # Individual KPI fields (Contract 9)
    cycle_time = Column(Float, nullable=True)  # Cycle time in seconds
    yield_rate = Column(Float, nullable=True)  # 0.0-1.0 (1 - scrap_rate)
    energy_consumption = Column(Float, nullable=True)  # kWh
    
    # Source log reference (Contract 9)
    source_log_id = Column(Integer, nullable=True)  # FK to operation_execution_logs
    
    # Metadata
    run_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_golden_revision_op', 'revision_id', 'operation_id'),
    )
    
    def __repr__(self):
        return f"<GoldenRun(revision={self.revision_id}, op={self.operation_id}, score={self.score})>"


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE DPP TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def init_dpp_db() -> None:
    """Create all DPP tables."""
    Base.metadata.create_all(bind=engine)


# Auto-initialize on import
init_dpp_db()

