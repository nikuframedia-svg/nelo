"""QR code generation for Duplios."""
from __future__ import annotations

import io

import qrcode

from duplios.models import DPPModel, SessionLocal


def generate_dpp_qrcode(dpp: DPPModel) -> bytes:
    payload = dpp.qr_public_url or f"duplios:{dpp.qr_slug}"
    qr_img = qrcode.make(payload)
    buffer = io.BytesIO()
    qr_img.save(buffer, format="PNG")
    return buffer.getvalue()


def get_qr_png_bytes(dpp_id: str) -> bytes:
    session = SessionLocal()
    try:
        dpp = session.query(DPPModel).filter(DPPModel.dpp_id == dpp_id).first()
        if not dpp:
            raise ValueError("DPP não encontrado")
        return generate_dpp_qrcode(dpp)
    finally:
        session.close()
