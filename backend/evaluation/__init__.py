"""
ProdPlan 4.0 - Evaluation Engine

Rigorous evaluation framework for:
- Plan quality assessment (KPIs)
- Data quality measurement (SNR)
- Model performance metrics
- Comparative analysis

Core Concept: Signal-to-Noise Ratio (SNR)
=========================================
SNR quantifies the ratio of meaningful signal to random variation:

    SNR = Var(signal) / Var(noise)
    SNR_dB = 10 * log10(SNR)

High SNR (>10):  Strong signal, reliable predictions
Medium SNR (3-10): Moderate signal, predictions useful with caution
Low SNR (<3):    Weak signal, predictions unreliable

R&D / SIFIDE: WP4 - Evaluation & Explainability
"""

from data_quality import (
    SignalNoiseAnalyzer,
    DataQualityReport,
    SNRResult,
    SNRClass,
    compute_snr,
    snr_to_db,
    db_to_snr,
    snr_to_r_squared,
    r_squared_to_snr,
    classify_snr,
    compute_confidence,
    interpret_snr,
    snr_processing_time,
    snr_setup_matrix,
    snr_forecast,
    snr_rul,
)

__all__ = [
    # Data Quality & SNR
    "SignalNoiseAnalyzer",
    "DataQualityReport",
    "SNRResult",
    "SNRClass",
    "compute_snr",
    "snr_to_db",
    "db_to_snr",
    "snr_to_r_squared",
    "r_squared_to_snr",
    "classify_snr",
    "compute_confidence",
    "interpret_snr",
    "snr_processing_time",
    "snr_setup_matrix",
    "snr_forecast",
    "snr_rul",
]
