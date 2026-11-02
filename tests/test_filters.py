"""
Unit tests for filtering pipeline and deadzone filter.
"""

import numpy as np
import pytest
from wearable_mouse.processing.filters import (
    LowPassFilter, DeadZoneFilter, MovingAverageFilter, ExponentialSmoothingFilter, FilterPipeline
)


def test_deadzone_filter():
    dz = DeadZoneFilter(threshold=0.05)

    # Sub-threshold motion vector should be zeroed
    small_vec = np.array([0.01, -0.02, 0.01])
    out_small = dz.process(small_vec)
    np.testing.assert_allclose(out_small, [0.0, 0.0, 0.0])

    # Above-threshold motion vector should pass smoothly
    large_vec = np.array([0.5, 0.0, 0.0])
    out_large = dz.process(large_vec)
    assert np.linalg.norm(out_large) > 0.0
    assert np.linalg.norm(out_large) < np.linalg.norm(large_vec)


def test_lowpass_filter():
    lp = LowPassFilter(cutoff_freq=10.0, sample_rate=100.0)

    # Step response smoothing
    v1 = np.array([1.0, 0.0, 0.0])
    out1 = lp.process(v1) # First sample initializes filter
    np.testing.assert_allclose(out1, v1)

    v2 = np.array([0.0, 0.0, 0.0])
    out2 = lp.process(v2)
    # Output should decay gradually towards zero, not jump instantly
    assert out2[0] > 0.0
    assert out2[0] < 1.0


def test_filter_pipeline():
    pipeline = FilterPipeline([
        DeadZoneFilter(0.05),
        ExponentialSmoothingFilter(0.5)
    ])

    vec = np.array([0.01, 0.01, 0.0])
    res = pipeline.process(vec)
    np.testing.assert_allclose(res, [0.0, 0.0, 0.0])
