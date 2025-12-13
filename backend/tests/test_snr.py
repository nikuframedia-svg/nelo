"""
═══════════════════════════════════════════════════════════════════════════════
                    PRODPLAN 4.0 — SNR Unit Tests
═══════════════════════════════════════════════════════════════════════════════

Tests for Signal-to-Noise Ratio computations in data_quality.py.

Run with: python -m pytest backend/tests/test_snr.py -v
"""

import numpy as np
import pandas as pd
import pytest

from backend.evaluation.data_quality import (
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
    SNRClass,
)


class TestSNRBasics:
    """Test basic SNR computation functions."""
    
    def test_snr_constant_signal(self):
        """Constant signal = zero noise = infinite SNR."""
        values = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        snr = compute_snr(values)
        assert snr >= 50.0  # Should be very high
    
    def test_snr_pure_noise(self):
        """Pure noise = low SNR."""
        np.random.seed(42)
        values = np.random.randn(100)  # Standard normal noise
        snr = compute_snr(values)
        assert snr < 2.0  # Should be low
    
    def test_snr_with_groups(self):
        """SNR with group structure (ANOVA)."""
        # Group A: values around 10
        # Group B: values around 20
        values = np.array([10, 11, 9, 10, 20, 21, 19, 20])
        groups = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
        
        snr = compute_snr(values, groups, method='anova')
        assert snr > 10.0  # Strong group separation
    
    def test_snr_no_group_separation(self):
        """SNR when groups have same mean."""
        values = np.array([10, 11, 9, 10, 10, 11, 9, 10])
        groups = np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
        
        snr = compute_snr(values, groups, method='anova')
        assert snr < 1.0  # No group effect
    
    def test_snr_minimum_samples(self):
        """SNR with too few samples returns 0."""
        values = np.array([1.0, 2.0])
        snr = compute_snr(values)
        assert snr == 0.0


class TestSNRConversions:
    """Test SNR conversion functions."""
    
    def test_snr_to_db_conversion(self):
        """Test SNR to dB conversion."""
        assert abs(snr_to_db(1.0) - 0.0) < 0.01  # SNR=1 → 0 dB
        assert abs(snr_to_db(10.0) - 10.0) < 0.01  # SNR=10 → 10 dB
        assert abs(snr_to_db(100.0) - 20.0) < 0.01  # SNR=100 → 20 dB
    
    def test_db_to_snr_conversion(self):
        """Test dB to SNR conversion."""
        assert abs(db_to_snr(0.0) - 1.0) < 0.01
        assert abs(db_to_snr(10.0) - 10.0) < 0.01
        assert abs(db_to_snr(20.0) - 100.0) < 0.01
    
    def test_snr_to_r_squared(self):
        """Test SNR to R² conversion."""
        assert abs(snr_to_r_squared(1.0) - 0.5) < 0.01  # SNR=1 → R²=0.5
        assert abs(snr_to_r_squared(3.0) - 0.75) < 0.01  # SNR=3 → R²=0.75
        assert abs(snr_to_r_squared(9.0) - 0.9) < 0.01  # SNR=9 → R²=0.9
    
    def test_r_squared_to_snr(self):
        """Test R² to SNR conversion."""
        assert abs(r_squared_to_snr(0.5) - 1.0) < 0.01
        assert abs(r_squared_to_snr(0.75) - 3.0) < 0.01
        assert abs(r_squared_to_snr(0.9) - 9.0) < 0.01
    
    def test_roundtrip_conversion(self):
        """Test that conversions are inverses."""
        for snr in [0.5, 1.0, 5.0, 10.0, 50.0]:
            assert abs(db_to_snr(snr_to_db(snr)) - snr) < 0.001
            
            r2 = snr_to_r_squared(snr)
            assert abs(r_squared_to_snr(r2) - snr) < 0.001


class TestSNRClassification:
    """Test SNR classification functions."""
    
    def test_classify_snr_excellent(self):
        assert classify_snr(15.0) == SNRClass.EXCELLENT
        assert classify_snr(10.0) == SNRClass.EXCELLENT
    
    def test_classify_snr_high(self):
        assert classify_snr(7.0) == SNRClass.HIGH
        assert classify_snr(5.0) == SNRClass.HIGH
    
    def test_classify_snr_medium(self):
        assert classify_snr(3.0) == SNRClass.MEDIUM
        assert classify_snr(2.0) == SNRClass.MEDIUM
    
    def test_classify_snr_low(self):
        assert classify_snr(1.5) == SNRClass.LOW
        assert classify_snr(1.0) == SNRClass.LOW
    
    def test_classify_snr_poor(self):
        assert classify_snr(0.5) == SNRClass.POOR
        assert classify_snr(0.1) == SNRClass.POOR
    
    def test_compute_confidence(self):
        """Test confidence score computation."""
        assert compute_confidence(0.0) == 0.0
        assert compute_confidence(1.0) > 0.2
        assert compute_confidence(3.0) > 0.5
        assert compute_confidence(10.0) > 0.9
        assert compute_confidence(100.0) > 0.99
    
    def test_interpret_snr(self):
        """Test SNR interpretation."""
        level, desc, conf = interpret_snr(5.0)
        assert level == "HIGH"
        assert "fiáv" in desc.lower() or "confiável" in desc.lower() or "previsibilidade" in desc.lower()
        assert 0.5 < conf < 1.0


class TestDomainSpecificSNR:
    """Test domain-specific SNR functions."""
    
    def test_snr_processing_time(self):
        """Test SNR for processing times."""
        plan_df = pd.DataFrame({
            'op_code': ['CUT', 'CUT', 'CUT', 'ASM', 'ASM', 'ASM'],
            'duration_min': [10, 11, 9, 20, 21, 19],
        })
        
        result = snr_processing_time(plan_df)
        
        assert result.snr_value > 0
        assert result.snr_class in ['EXCELLENT', 'HIGH', 'MEDIUM', 'LOW', 'POOR']
        assert 0 <= result.confidence_score <= 1
        assert result.sample_size == 6
    
    def test_snr_processing_time_empty(self):
        """Test SNR with empty data."""
        plan_df = pd.DataFrame({'op_code': [], 'duration_min': []})
        result = snr_processing_time(plan_df)
        
        assert result.snr_value == 0.0
        assert result.snr_class == "POOR"
    
    def test_snr_setup_matrix(self):
        """Test SNR for setup matrix."""
        setup_df = pd.DataFrame({
            'from_setup_family': ['A', 'A', 'B', 'B'],
            'to_setup_family': ['A', 'B', 'A', 'B'],
            'setup_time_min': [5, 15, 15, 5],
        })
        
        result = snr_setup_matrix(setup_df)
        
        assert result.snr_value > 0
        assert result.sample_size == 4
    
    def test_snr_forecast(self):
        """Test SNR for forecast evaluation."""
        actual = np.array([100, 102, 98, 105, 97, 103])
        predicted = np.array([100, 101, 99, 104, 98, 102])
        
        result = snr_forecast(actual, predicted, model_name='ARIMA')
        
        assert result.snr_value > 5.0  # Good prediction
        assert result.snr_class in ['EXCELLENT', 'HIGH']
        assert 'ARIMA' in result.description
        assert 'mape_pct' in result.details
    
    def test_snr_forecast_poor(self):
        """Test SNR for poor forecast."""
        actual = np.array([100, 102, 98, 105, 97, 103])
        predicted = np.array([50, 150, 20, 200, 10, 180])  # Bad predictions
        
        result = snr_forecast(actual, predicted)
        
        assert result.snr_value < 2.0  # Poor prediction
    
    def test_snr_rul(self):
        """Test SNR for RUL degradation signal."""
        # Linear degradation with some noise
        t = np.arange(0, 100)
        degradation = 100 - 0.5 * t + np.random.randn(100) * 2
        
        result = snr_rul(degradation)
        
        assert result.snr_value > 5.0  # Clear trend
        assert 'trend_slope' in result.details
        assert result.details['trend_slope'] < 0  # Decreasing
    
    def test_snr_rul_with_threshold(self):
        """Test RUL estimation with failure threshold."""
        t = np.arange(0, 50)
        degradation = 100 - t  # Linear degradation
        
        result = snr_rul(degradation, failure_threshold=0)
        
        assert result.details['rul_estimate'] is not None
        assert result.details['rul_estimate'] > 0  # Should predict future failure


class TestSNREdgeCases:
    """Test edge cases for SNR computation."""
    
    def test_snr_nan_values(self):
        """Test handling of NaN values."""
        values = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])
        snr = compute_snr(values)
        assert snr > 0  # Should handle NaN gracefully
    
    def test_snr_negative_values(self):
        """Test with negative values."""
        values = np.array([-10, -5, 0, 5, 10])
        snr = compute_snr(values)
        assert snr > 0
    
    def test_snr_zero_variance(self):
        """Test with zero variance group."""
        values = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 20.0])
        groups = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
        snr = compute_snr(values, groups, method='anova')
        assert snr >= 50.0  # Very high SNR (infinite in theory)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])



