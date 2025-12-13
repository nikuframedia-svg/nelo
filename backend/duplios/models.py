"""SQLAlchemy models for Duplios DPP module."""
from __future__ import annotations

import os
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, JSON, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DUPLIOS_DATABASE_URL", "sqlite:///duplios.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


class DPPModel(Base):
    __tablename__ = "duplios_dpp"

    id = Column(Integer, primary_key=True, index=True)
    dpp_id = Column(String, unique=True, index=True, default=lambda: str(uuid.uuid4()))
    qr_slug = Column(String, unique=True, index=True)
    gtin = Column(String, index=True, nullable=False)
    product_name = Column(String, nullable=False)
    product_category = Column(String, nullable=True)
    status = Column(String, default="draft", index=True)
    manufacturer_name = Column(String, nullable=True)
    country_of_origin = Column(String, nullable=True)
    trust_index = Column(Float, default=60.0)
    carbon_footprint_kg_co2eq = Column(Float, default=0.0)
    recyclability_percent = Column(Float, default=0.0)
    data_completeness_percent = Column(Float, default=0.0)
    qr_public_url = Column(String, nullable=True)
    dpp_data = Column(JSON, nullable=False)
    created_by_user_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Initialize table on import
init_db()

# Seed sample DPPs
try:
    from duplios.seed_data import seed_dpps
    _seeded = seed_dpps()
    if _seeded > 0:
        print(f"[Duplios] Seeded {_seeded} sample DPPs")
except Exception as e:
    print(f"[Duplios] Seed skipped: {e}")
