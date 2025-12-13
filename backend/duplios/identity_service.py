"""
════════════════════════════════════════════════════════════════════════════════════════════════════
IDENTITY SERVICE - Digital Identity Management for Duplios
════════════════════════════════════════════════════════════════════════════════════════════════════

Contract 5 Implementation: Digital Identity

Services for:
- Ingesting new digital identities (RFID, QR, serial numbers)
- Verifying identities
- Managing lineage/traceability
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from duplios.dpp_models import DigitalIdentity, VerificationStatus
from duplios.models import SessionLocal

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# INGEST IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_identity(
    item_revision_id: int,
    unique_item_id: str,
    lineage_parent_id: Optional[int] = None,
    db: Optional[Session] = None,
) -> DigitalIdentity:
    """
    Ingest a new digital identity for a product instance.
    
    Args:
        item_revision_id: ID of the ItemRevision this identity belongs to
        unique_item_id: Physical identifier (RFID, QR code, serial number)
        lineage_parent_id: Optional parent identity for traceability
        db: Database session
    
    Returns:
        DigitalIdentity record with status UNVERIFIED
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        now = datetime.utcnow()
        
        # Compute hash
        identity_hash = DigitalIdentity.compute_hash(
            item_revision_id, 
            unique_item_id, 
            now
        )
        
        # Check for existing identity with same unique_item_id
        existing = db.query(DigitalIdentity).filter(
            DigitalIdentity.unique_item_id == unique_item_id
        ).first()
        
        if existing:
            logger.warning(
                f"Identity with unique_id {unique_item_id} already exists. "
                "Creating new record but marking potential conflict."
            )
        
        # Create new identity
        identity = DigitalIdentity(
            item_revision_id=item_revision_id,
            unique_item_id=unique_item_id,
            identity_hash=identity_hash,
            lineage_parent_id=lineage_parent_id,
            verification_status=VerificationStatus.UNVERIFIED,
            created_at=now,
        )
        
        db.add(identity)
        db.commit()
        db.refresh(identity)
        
        logger.info(
            f"Ingested identity {identity.id} for revision {item_revision_id}: "
            f"{unique_item_id[:20]}..."
        )
        
        return identity
        
    finally:
        if close_session:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════

def verify_identity(
    unique_item_id: str,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """
    Verify a digital identity.
    
    Logic:
    - 0 records found: Return UNVERIFIED status
    - 1 record found: Mark as VERIFIED, update last_verified_at
    - >1 records found: Mark all as CONFLICT
    
    Args:
        unique_item_id: Physical identifier to verify
        db: Database session
    
    Returns:
        Dict with verification result and identity data
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        now = datetime.utcnow()
        
        # Find all identities with this unique_item_id
        identities = db.query(DigitalIdentity).filter(
            DigitalIdentity.unique_item_id == unique_item_id
        ).all()
        
        count = len(identities)
        
        if count == 0:
            # Not found
            logger.info(f"Identity verification: {unique_item_id} not found")
            return {
                "status": "UNVERIFIED",
                "message": "Identity not found in system",
                "unique_item_id": unique_item_id,
                "identity": None,
            }
        
        elif count == 1:
            # Single match - verify
            identity = identities[0]
            identity.verification_status = VerificationStatus.VERIFIED
            identity.last_verified_at = now
            identity.verification_count = (identity.verification_count or 0) + 1
            
            db.commit()
            db.refresh(identity)
            
            logger.info(
                f"Identity verified: {unique_item_id} -> "
                f"revision {identity.item_revision_id}"
            )
            
            return {
                "status": "VERIFIED",
                "message": "Identity verified successfully",
                "unique_item_id": unique_item_id,
                "identity": _identity_to_dict(identity),
            }
        
        else:
            # Multiple matches - conflict
            logger.warning(
                f"Identity conflict: {unique_item_id} matches {count} records"
            )
            
            # Mark all as conflict
            for identity in identities:
                identity.verification_status = VerificationStatus.CONFLICT
                identity.last_verified_at = now
            
            db.commit()
            
            return {
                "status": "CONFLICT",
                "message": f"Multiple identities ({count}) found with same ID - conflict detected",
                "unique_item_id": unique_item_id,
                "identities": [_identity_to_dict(i) for i in identities],
            }
        
    finally:
        if close_session:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _identity_to_dict(identity: DigitalIdentity) -> Dict[str, Any]:
    """Convert DigitalIdentity to dictionary."""
    return {
        "id": identity.id,
        "item_revision_id": identity.item_revision_id,
        "unique_item_id": identity.unique_item_id,
        "identity_hash": identity.identity_hash[:16] + "...",  # Truncate for display
        "verification_status": identity.verification_status.value,
        "last_verified_at": identity.last_verified_at.isoformat() if identity.last_verified_at else None,
        "verification_count": identity.verification_count or 0,
        "created_at": identity.created_at.isoformat() if identity.created_at else None,
        "has_lineage": identity.lineage_parent_id is not None,
    }


def get_identity_by_id(
    identity_id: int,
    db: Optional[Session] = None,
) -> Optional[DigitalIdentity]:
    """Get identity by ID."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        return db.query(DigitalIdentity).filter(
            DigitalIdentity.id == identity_id
        ).first()
    finally:
        if close_session:
            db.close()


def get_identities_for_revision(
    revision_id: int,
    db: Optional[Session] = None,
) -> List[DigitalIdentity]:
    """Get all identities for a revision."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        return db.query(DigitalIdentity).filter(
            DigitalIdentity.item_revision_id == revision_id
        ).order_by(DigitalIdentity.created_at.desc()).all()
    finally:
        if close_session:
            db.close()


def get_identity_lineage(
    identity_id: int,
    db: Optional[Session] = None,
) -> List[DigitalIdentity]:
    """
    Get lineage chain for an identity.
    
    Traces back through lineage_parent_id to build full chain.
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        chain = []
        current_id = identity_id
        max_depth = 100  # Prevent infinite loops
        
        while current_id and len(chain) < max_depth:
            identity = db.query(DigitalIdentity).filter(
                DigitalIdentity.id == current_id
            ).first()
            
            if not identity:
                break
            
            chain.append(identity)
            current_id = identity.lineage_parent_id
        
        return chain
        
    finally:
        if close_session:
            db.close()


def mark_duplicate(
    identity_id: int,
    db: Optional[Session] = None,
) -> Optional[DigitalIdentity]:
    """Mark an identity as duplicate."""
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        identity = db.query(DigitalIdentity).filter(
            DigitalIdentity.id == identity_id
        ).first()
        
        if identity:
            identity.verification_status = VerificationStatus.DUPLICATE
            identity.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(identity)
        
        return identity
        
    finally:
        if close_session:
            db.close()


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def batch_ingest_identities(
    item_revision_id: int,
    unique_item_ids: List[str],
    db: Optional[Session] = None,
) -> List[DigitalIdentity]:
    """
    Ingest multiple identities at once.
    
    Useful for batch production scenarios.
    """
    close_session = False
    if db is None:
        db = SessionLocal()
        close_session = True
    
    try:
        now = datetime.utcnow()
        identities = []
        
        for unique_id in unique_item_ids:
            identity_hash = DigitalIdentity.compute_hash(
                item_revision_id, 
                unique_id, 
                now
            )
            
            identity = DigitalIdentity(
                item_revision_id=item_revision_id,
                unique_item_id=unique_id,
                identity_hash=identity_hash,
                verification_status=VerificationStatus.UNVERIFIED,
                created_at=now,
            )
            
            db.add(identity)
            identities.append(identity)
        
        db.commit()
        
        for identity in identities:
            db.refresh(identity)
        
        logger.info(
            f"Batch ingested {len(identities)} identities for revision {item_revision_id}"
        )
        
        return identities
        
    finally:
        if close_session:
            db.close()

