"""
Tests for CalibrationMetricsCalculator and CalibrationMetrics.

Covers NMBE, CV(RMSE), R², RMSE, MBE, MAE, compute_all_metrics,
is_calibrated, and get_status against ASHRAE/IPMVP standards.
"""

import pytest
import numpy as np
import pandas as pd

from energyplus_mcp_server.utils.calibration import (
    CalibrationMetricsCalculator,
    CalibrationMetrics,
    CalibrationStandard,
)

calc = CalibrationMetricsCalculator


# ---------------------------------------------------------------------------
# compute_nmbe
# ---------------------------------------------------------------------------

class TestComputeNMBE:
    def test_hand_calculated(self):
        """NMBE = 100 * sum(M-S) / ((n-1) * mean(M))"""
        m = np.array([100.0, 200.0, 300.0])
        s = np.array([110.0, 190.0, 280.0])
        # sum(M-S) = -10+10+20 = 20; (n-1)*mean(M) = 2*200 = 400
        expected = 100.0 * 20.0 / 400.0  # 5.0
        assert calc.compute_nmbe(m, s) == pytest.approx(expected)

    def test_positive_bias(self):
        """Simulation under-predicts → positive NMBE."""
        m = np.array([100.0, 100.0, 100.0])
        s = np.array([90.0, 90.0, 90.0])
        assert calc.compute_nmbe(m, s) > 0

    def test_negative_bias(self):
        """Simulation over-predicts → negative NMBE."""
        m = np.array([100.0, 100.0, 100.0])
        s = np.array([110.0, 110.0, 110.0])
        assert calc.compute_nmbe(m, s) < 0

    def test_n1_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            calc.compute_nmbe(np.array([1.0]), np.array([1.0]))

    def test_mean_zero_raises(self):
        with pytest.raises(ValueError, match="cannot be zero"):
            calc.compute_nmbe(np.array([1.0, -1.0]), np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# compute_cvrmse
# ---------------------------------------------------------------------------

class TestComputeCVRMSE:
    def test_hand_calculated(self):
        m = np.array([100.0, 200.0, 300.0])
        s = np.array([110.0, 190.0, 280.0])
        residuals = m - s  # [-10, 10, 20]
        rmse = np.sqrt(np.sum(residuals**2) / 2)  # sqrt((100+100+400)/2)=sqrt(300)
        expected = 100.0 * rmse / np.mean(m)
        assert calc.compute_cvrmse(m, s) == pytest.approx(expected)

    def test_perfect_agreement(self):
        m = np.array([100.0, 200.0, 300.0])
        assert calc.compute_cvrmse(m, m.copy()) == pytest.approx(0.0)

    def test_n1_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            calc.compute_cvrmse(np.array([1.0]), np.array([1.0]))

    def test_mean_zero_raises(self):
        with pytest.raises(ValueError, match="cannot be zero"):
            calc.compute_cvrmse(np.array([1.0, -1.0]), np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# compute_r_squared
# ---------------------------------------------------------------------------

class TestComputeRSquared:
    def test_perfect_fit(self):
        m = np.array([10.0, 20.0, 30.0])
        assert calc.compute_r_squared(m, m.copy()) == pytest.approx(1.0)

    def test_no_fit(self):
        """Simulated = mean of measured → R² = 0."""
        m = np.array([10.0, 20.0, 30.0])
        s = np.full_like(m, np.mean(m))
        assert calc.compute_r_squared(m, s) == pytest.approx(0.0)

    def test_negative_r_squared(self):
        """Simulated worse than mean → R² < 0."""
        m = np.array([10.0, 20.0, 30.0])
        s = np.array([30.0, 10.0, 20.0])  # anti-correlated
        r2 = calc.compute_r_squared(m, s)
        assert r2 < 0

    def test_identical_measured_ss_tot_zero_perfect(self):
        """All measured identical, simulated matches → R² = 1.0."""
        m = np.array([5.0, 5.0, 5.0])
        s = np.array([5.0, 5.0, 5.0])
        assert calc.compute_r_squared(m, s) == pytest.approx(1.0)

    def test_identical_measured_ss_tot_zero_imperfect(self):
        """All measured identical, simulated differs → R² = 0.0."""
        m = np.array([5.0, 5.0, 5.0])
        s = np.array([6.0, 4.0, 5.0])
        assert calc.compute_r_squared(m, s) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------

class TestComputeRMSE:
    def test_hand_calculated(self):
        m = np.array([100.0, 200.0, 300.0])
        s = np.array([110.0, 190.0, 280.0])
        expected = np.sqrt(np.sum((m - s)**2) / 2)
        assert calc.compute_rmse(m, s) == pytest.approx(expected)

    def test_n1_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            calc.compute_rmse(np.array([1.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# compute_mbe
# ---------------------------------------------------------------------------

class TestComputeMBE:
    def test_positive_bias(self):
        m = np.array([100.0, 200.0])
        s = np.array([90.0, 180.0])
        # mean(M-S) = mean([10, 20]) = 15
        assert calc.compute_mbe(m, s) == pytest.approx(15.0)

    def test_negative_bias(self):
        m = np.array([100.0, 200.0])
        s = np.array([110.0, 220.0])
        assert calc.compute_mbe(m, s) < 0


# ---------------------------------------------------------------------------
# compute_mae
# ---------------------------------------------------------------------------

class TestComputeMAE:
    def test_known_values(self):
        m = np.array([100.0, 200.0])
        s = np.array([110.0, 180.0])
        # mean(|M-S|) = mean([10, 20]) = 15
        assert calc.compute_mae(m, s) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_accepts_list(self, sample_arrays):
        measured, simulated = sample_arrays
        result = calc.compute_all_metrics(measured.tolist(), simulated.tolist())
        assert isinstance(result, CalibrationMetrics)
        assert result.n_points == 12

    def test_accepts_ndarray(self, sample_arrays):
        measured, simulated = sample_arrays
        result = calc.compute_all_metrics(measured, simulated)
        assert isinstance(result, CalibrationMetrics)

    def test_accepts_series(self, sample_arrays):
        measured, simulated = sample_arrays
        result = calc.compute_all_metrics(
            pd.Series(measured), pd.Series(simulated)
        )
        assert isinstance(result, CalibrationMetrics)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            calc.compute_all_metrics([1.0, 2.0], [1.0])

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            calc.compute_all_metrics([1.0], [1.0])

    def test_nan_rejection(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            calc.compute_all_metrics([1.0, float("nan")], [1.0, 2.0])

    def test_inf_rejection(self):
        with pytest.raises(ValueError, match="NaN or Inf"):
            calc.compute_all_metrics([1.0, float("inf")], [1.0, 2.0])


# ---------------------------------------------------------------------------
# CalibrationMetrics.is_calibrated
# ---------------------------------------------------------------------------

class TestIsCalibrated:
    def test_ashrae_monthly_pass(self, sample_arrays):
        measured, simulated = sample_arrays
        metrics = calc.compute_all_metrics(measured, simulated)
        # Our sample data is well-calibrated
        assert metrics.is_calibrated(CalibrationStandard.ASHRAE_14_MONTHLY)

    def test_ashrae_monthly_fail(self):
        """Intentionally bad data should fail."""
        m = np.array([100.0, 200.0, 300.0])
        s = np.array([200.0, 100.0, 500.0])  # terrible fit
        metrics = calc.compute_all_metrics(m, s)
        assert not metrics.is_calibrated(CalibrationStandard.ASHRAE_14_MONTHLY)

    def test_ipmvp_no_r2_check(self):
        """IPMVP monthly has no R² threshold — only NMBE and CV(RMSE)."""
        # Build metrics that pass NMBE/CVRMSE but have low R²
        # We'll just verify the standard has no r2_limit
        from energyplus_mcp_server.utils.calibration import CalibrationThresholds
        thresholds = CalibrationThresholds.get_thresholds(CalibrationStandard.IPMVP_MONTHLY)
        assert thresholds.r2_limit is None


# ---------------------------------------------------------------------------
# CalibrationMetrics.get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_structure(self, sample_arrays):
        measured, simulated = sample_arrays
        metrics = calc.compute_all_metrics(measured, simulated)
        status = metrics.get_status(CalibrationStandard.ASHRAE_14_MONTHLY)

        assert "standard" in status
        assert "is_calibrated" in status
        assert "metrics" in status
        assert "nmbe" in status["metrics"]
        assert "cvrmse" in status["metrics"]
        # ASHRAE monthly includes R²
        assert "r_squared" in status["metrics"]

        # Each metric sub-dict should have value, limit, passed, margin
        for key in ("nmbe", "cvrmse", "r_squared"):
            sub = status["metrics"][key]
            assert "value" in sub
            assert "limit" in sub
            assert "passed" in sub
            assert "margin" in sub

    def test_ipmvp_no_r2_in_status(self, sample_arrays):
        measured, simulated = sample_arrays
        metrics = calc.compute_all_metrics(measured, simulated)
        status = metrics.get_status(CalibrationStandard.IPMVP_MONTHLY)
        assert "r_squared" not in status["metrics"]
