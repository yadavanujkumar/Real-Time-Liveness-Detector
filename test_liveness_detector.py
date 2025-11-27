"""
Unit tests for liveness_detector.py

Tests the core functions without requiring a webcam.
"""

import numpy as np
import pytest
from liveness_detector import (
    calculate_ear,
    LivenessState,
    EAR_THRESHOLD,
    BLINKS_REQUIRED,
    SPOOF_TIMEOUT
)


class TestCalculateEAR:
    """Tests for the Eye Aspect Ratio calculation."""

    def test_open_eye_returns_high_ear(self):
        """Open eye should return EAR above threshold."""
        # Simulate an open eye with vertical distance roughly equal to horizontal
        eye_landmarks = np.array([
            [0, 50],    # p1: left corner
            [20, 30],   # p2: upper left
            [40, 30],   # p3: upper right
            [60, 50],   # p4: right corner
            [40, 70],   # p5: lower right
            [20, 70],   # p6: lower left
        ])
        ear = calculate_ear(eye_landmarks)
        assert ear > EAR_THRESHOLD, f"Open eye EAR ({ear}) should be > {EAR_THRESHOLD}"

    def test_closed_eye_returns_low_ear(self):
        """Closed eye should return EAR below threshold."""
        # Simulate a closed eye with very small vertical distance
        eye_landmarks = np.array([
            [0, 50],    # p1: left corner
            [20, 48],   # p2: upper left (close to center)
            [40, 48],   # p3: upper right (close to center)
            [60, 50],   # p4: right corner
            [40, 52],   # p5: lower right (close to center)
            [20, 52],   # p6: lower left (close to center)
        ])
        ear = calculate_ear(eye_landmarks)
        assert ear < EAR_THRESHOLD, f"Closed eye EAR ({ear}) should be < {EAR_THRESHOLD}"

    def test_horizontal_distance_zero_returns_zero(self):
        """Should handle division by zero gracefully."""
        eye_landmarks = np.array([
            [50, 50],   # p1: same as p4
            [50, 30],
            [50, 30],
            [50, 50],   # p4: same as p1
            [50, 70],
            [50, 70],
        ])
        ear = calculate_ear(eye_landmarks)
        assert ear == 0.0

    def test_symmetric_eye_returns_consistent_ear(self):
        """Symmetric eye landmarks should produce consistent EAR."""
        eye_landmarks = np.array([
            [0, 50],
            [15, 35],
            [35, 35],
            [50, 50],
            [35, 65],
            [15, 65],
        ])
        ear1 = calculate_ear(eye_landmarks)
        ear2 = calculate_ear(eye_landmarks)
        assert ear1 == ear2


class TestLivenessState:
    """Tests for the LivenessState enum."""

    def test_scanning_state_exists(self):
        """SCANNING state should exist."""
        assert LivenessState.SCANNING.value == 1

    def test_verified_state_exists(self):
        """VERIFIED state should exist."""
        assert LivenessState.VERIFIED.value == 2

    def test_spoof_state_exists(self):
        """SPOOF state should exist."""
        assert LivenessState.SPOOF.value == 3


class TestConstants:
    """Tests for configuration constants."""

    def test_ear_threshold_is_positive(self):
        """EAR threshold should be a positive number."""
        assert EAR_THRESHOLD > 0

    def test_ear_threshold_is_reasonable(self):
        """EAR threshold should be in a reasonable range (0.1 - 0.5)."""
        assert 0.1 <= EAR_THRESHOLD <= 0.5

    def test_blinks_required_is_positive(self):
        """Blinks required should be positive."""
        assert BLINKS_REQUIRED > 0

    def test_spoof_timeout_is_reasonable(self):
        """Spoof timeout should be reasonable (1-30 seconds)."""
        assert 1 <= SPOOF_TIMEOUT <= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
