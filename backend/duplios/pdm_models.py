"""
════════════════════════════════════════════════════════════════════════════════════════════════════
PDM MODELS - Product Data Management for Duplios
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 5 Implementation: PDM Lite

Models:
- Item: Master product/component record
- ItemRevision: Version control for items
- BomLine: Bill of Materials structure
- RoutingOperation: Manufacturing routing steps
- ECR: Engineering Change Request
- ECO: Engineering Change Order

This provides a lightweight PLM/PDM system integrated with DPP.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float, ForeignKey, Integer, 
    String, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column

from duplios.models import Base, engine


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class ItemType(str, enum.Enum):
    """Type of item in the PDM system."""
    FINISHED = "FINISHED"           # Finished goods
    SEMI_FINISHED = "SEMI_FINISHED" # Work-in-progress / assemblies
    RAW_MATERIAL = "RAW_MATERIAL"   # Raw materials
    TOOLING = "TOOLING"             # Tools and fixtures
    PACKAGING = "PACKAGING"         # Packaging materials


class RevisionStatus(str, enum.Enum):
    """Status of an item revision."""
    DRAFT = "DRAFT"         # In development
    RELEASED = "RELEASED"   # Active for production
    OBSOLETE = "OBSOLETE"   # No longer valid


class ECRStatus(str, enum.Enum):
    """Status of Engineering Change Request."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class Item(Base):
    """
    Master record for a product, component, or material.
    
    Each Item can have multiple revisions (versions).
    """
    __tablename__ = "pdm_items"
    
    id = Column(Integer, primary_key=True, index=True)
    sku = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    type = Column(Enum(ItemType), nullable=False, default=ItemType.FINISHED)
    unit = Column(String(20), nullable=False, default="pcs")  # pcs, kg, m, etc.
    family = Column(String(100), nullable=True)  # Product family/category
    weight_kg = Column(Float, nullable=True)  # Unit weight
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    revisions = relationship("ItemRevision", back_populates="item", cascade="all, delete-orphan")
    ecrs = relationship("ECR", back_populates="item", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Item(sku={self.sku}, name={self.name}, type={self.type.value})>"


# ═══════════════════════════════════════════════════════════════════════════════
# ITEM REVISION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ItemRevision(Base):
    """
    A specific version/revision of an Item.
    
    Contains BOM and Routing specific to this revision.
    """
    __tablename__ = "pdm_item_revisions"
    
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("pdm_items.id", ondelete="CASCADE"), nullable=False)
    
    code = Column(String(10), nullable=False)  # "A", "B", "C", "01", "02", etc.
    status = Column(Enum(RevisionStatus), nullable=False, default=RevisionStatus.DRAFT)
    
    effective_from = Column(DateTime, nullable=True)  # When revision becomes active
    effective_to = Column(DateTime, nullable=True)    # When revision expires
    notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    item = relationship("Item", back_populates="revisions")
    bom_lines = relationship("BomLine", 
                            back_populates="parent_revision",
                            foreign_keys="BomLine.parent_revision_id",
                            cascade="all, delete-orphan")
    routing_operations = relationship("RoutingOperation", 
                                     back_populates="revision",
                                     cascade="all, delete-orphan")
    
    # DPP records for this revision
    dpp_records = relationship("DppRecord", back_populates="item_revision", cascade="all, delete-orphan")
    digital_identities = relationship("DigitalIdentity", back_populates="item_revision", cascade="all, delete-orphan")
    
    # Engineering attachments
    attachments = relationship("Attachment", back_populates="revision", cascade="all, delete-orphan")
    
    # Unique constraint: one revision code per item
    __table_args__ = (
        UniqueConstraint('item_id', 'code', name='uix_item_revision_code'),
        Index('ix_revision_status', 'status'),
    )
    
    def __repr__(self):
        return f"<ItemRevision(item_id={self.item_id}, code={self.code}, status={self.status.value})>"


# ═══════════════════════════════════════════════════════════════════════════════
# BOM LINE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class BomLine(Base):
    """
    Bill of Materials line item.
    
    Links a parent revision to its component revisions.
    """
    __tablename__ = "pdm_bom_lines"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Parent (the product being built)
    parent_revision_id = Column(Integer, 
                                ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                                nullable=False)
    
    # Component (what goes into the parent)
    component_revision_id = Column(Integer, 
                                   ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                                   nullable=False)
    
    qty_per_unit = Column(Float, nullable=False, default=1.0)  # Quantity needed per parent unit
    scrap_rate = Column(Float, nullable=False, default=0.0)    # Expected scrap % (0.0 - 1.0)
    
    # Optional fields
    position = Column(String(50), nullable=True)  # Position/reference on assembly
    notes = Column(Text, nullable=True)
    
    # Relationships
    parent_revision = relationship("ItemRevision", 
                                  back_populates="bom_lines",
                                  foreign_keys=[parent_revision_id])
    component_revision = relationship("ItemRevision", 
                                      foreign_keys=[component_revision_id])
    
    __table_args__ = (
        Index('ix_bom_parent', 'parent_revision_id'),
        Index('ix_bom_component', 'component_revision_id'),
    )
    
    def __repr__(self):
        return f"<BomLine(parent={self.parent_revision_id}, component={self.component_revision_id}, qty={self.qty_per_unit})>"


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING OPERATION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RoutingOperation(Base):
    """
    Manufacturing routing operation for a revision.
    
    Defines the sequence of operations to produce the item.
    """
    __tablename__ = "pdm_routing_operations"
    
    id = Column(Integer, primary_key=True, index=True)
    revision_id = Column(Integer, 
                        ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                        nullable=False)
    
    op_code = Column(String(50), nullable=False)  # Operation code (e.g., "CUT", "WELD", "PAINT")
    sequence = Column(Integer, nullable=False, default=10)  # Order in routing (10, 20, 30...)
    machine_group = Column(String(100), nullable=True)  # Machine/work center group
    
    # Standard times (in minutes)
    nominal_setup_time = Column(Float, nullable=False, default=0.0)  # Setup time per batch
    nominal_cycle_time = Column(Float, nullable=False, default=0.0)  # Processing time per unit
    
    # Optional
    tool_id = Column(String(100), nullable=True)  # Required tooling
    description = Column(Text, nullable=True)
    name = Column(String(255), nullable=True)  # Friendly name for the operation
    
    # Poka-Yoke flags (Contract 9)
    is_critical = Column(Boolean, default=False)  # Requires work instructions
    requires_inspection = Column(Boolean, default=False)  # Quality checkpoint
    
    # Relationships
    revision = relationship("ItemRevision", back_populates="routing_operations")
    work_instructions = relationship("WorkInstruction", back_populates="routing_operation", cascade="all, delete-orphan")
    
    __table_args__ = (
        UniqueConstraint('revision_id', 'sequence', name='uix_routing_sequence'),
        Index('ix_routing_revision', 'revision_id'),
    )
    
    def __repr__(self):
        return f"<RoutingOperation(revision={self.revision_id}, seq={self.sequence}, op={self.op_code})>"


# ═══════════════════════════════════════════════════════════════════════════════
# ECR MODEL (Engineering Change Request)
# ═══════════════════════════════════════════════════════════════════════════════

class ECR(Base):
    """
    Engineering Change Request.
    
    Formal request to modify an item (design, BOM, routing, etc.)
    """
    __tablename__ = "pdm_ecr"
    
    id = Column(Integer, primary_key=True, index=True)
    item_id = Column(Integer, ForeignKey("pdm_items.id", ondelete="CASCADE"), nullable=False)
    
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    reason = Column(Text, nullable=True)  # Why the change is needed
    priority = Column(String(20), nullable=True, default="MEDIUM")  # LOW, MEDIUM, HIGH, CRITICAL
    
    status = Column(Enum(ECRStatus), nullable=False, default=ECRStatus.OPEN)
    
    # Requestor info
    requested_by = Column(String(100), nullable=True)
    requested_at = Column(DateTime, default=datetime.utcnow)
    
    # Closure info
    closed_at = Column(DateTime, nullable=True)
    closed_by = Column(String(100), nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    item = relationship("Item", back_populates="ecrs")
    ecos = relationship("ECO", back_populates="ecr", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ECR(id={self.id}, item_id={self.item_id}, status={self.status.value})>"


# ═══════════════════════════════════════════════════════════════════════════════
# ECO MODEL (Engineering Change Order)
# ═══════════════════════════════════════════════════════════════════════════════

class Attachment(Base):
    """
    Engineering attachment (CAD, PDF, work instructions, quality plans).
    
    Stores references to files (external storage) linked to item revisions.
    """
    __tablename__ = "pdm_attachments"
    
    id = Column(Integer, primary_key=True, index=True)
    revision_id = Column(Integer, 
                        ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                        nullable=False)
    
    # File metadata
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # "CAD", "PDF", "IMAGE", "WORK_INSTRUCTION", "QUALITY_PLAN", etc.
    file_path = Column(String(500), nullable=True)  # Path to external storage (S3, local, etc.)
    file_url = Column(String(500), nullable=True)  # URL if stored externally
    file_size_bytes = Column(Integer, nullable=True)
    mime_type = Column(String(100), nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    uploaded_by = Column(String(100), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    revision = relationship("ItemRevision", back_populates="attachments")
    
    __table_args__ = (
        Index('ix_attachment_revision', 'revision_id'),
        Index('ix_attachment_type', 'file_type'),
    )
    
    def __repr__(self):
        return f"<Attachment(revision={self.revision_id}, file={self.file_name}, type={self.file_type})>"


class ECO(Base):
    """
    Engineering Change Order.
    
    Implements an ECR by creating a new revision.
    """
    __tablename__ = "pdm_eco"
    
    id = Column(Integer, primary_key=True, index=True)
    ecr_id = Column(Integer, ForeignKey("pdm_ecr.id", ondelete="CASCADE"), nullable=False)
    
    # Revision transition
    from_revision_id = Column(Integer, 
                             ForeignKey("pdm_item_revisions.id", ondelete="SET NULL"), 
                             nullable=True)
    to_revision_id = Column(Integer, 
                           ForeignKey("pdm_item_revisions.id", ondelete="SET NULL"), 
                           nullable=True)
    
    # Approval
    approved_by = Column(String(100), nullable=True)
    approved_at = Column(DateTime, nullable=True)
    
    # Implementation
    implemented_at = Column(DateTime, nullable=True)
    implementation_notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    ecr = relationship("ECR", back_populates="ecos")
    from_revision = relationship("ItemRevision", foreign_keys=[from_revision_id])
    to_revision = relationship("ItemRevision", foreign_keys=[to_revision_id])
    
    def __repr__(self):
        return f"<ECO(id={self.id}, ecr_id={self.ecr_id}, from={self.from_revision_id}, to={self.to_revision_id})>"


# ═══════════════════════════════════════════════════════════════════════════════
# WORK INSTRUCTION MODEL (for Contract 8)
# ═══════════════════════════════════════════════════════════════════════════════

class WorkInstruction(Base):
    """
    Work instruction for a routing operation.
    
    Contains step-by-step instructions and quality checklists.
    """
    __tablename__ = "pdm_work_instructions"
    
    id = Column(Integer, primary_key=True, index=True)
    revision_id = Column(Integer, 
                        ForeignKey("pdm_item_revisions.id", ondelete="CASCADE"), 
                        nullable=False)
    operation_id = Column(Integer, 
                         ForeignKey("pdm_routing_operations.id", ondelete="CASCADE"), 
                         nullable=False)
    
    title = Column(String(255), nullable=False)
    steps = Column(Text, nullable=True)  # JSON: [{order, text, image_url?}]
    quality_checklist = Column(Text, nullable=True)  # JSON: [{order, question, type, spec_min?, spec_max?}]
    
    # Metadata
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    item_revision = relationship("ItemRevision", foreign_keys=[revision_id])
    routing_operation = relationship("RoutingOperation", back_populates="work_instructions")
    
    __table_args__ = (
        UniqueConstraint('revision_id', 'operation_id', name='uix_work_instruction'),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZE PDM TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def init_pdm_db() -> None:
    """Create all PDM tables."""
    Base.metadata.create_all(bind=engine)


# Auto-initialize on import
init_pdm_db()

