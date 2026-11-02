"""
Unit tests for uncertainty radius estimation and confidence scoring.
"""

import numpy as np
import pytest
from wearable_mouse.config import Config
from wearable_mouse.processing.motion_detector import MotionState, MovementType
from wearable_mouse.processing.uncertainty import UncertaintyEstimator


def test_uncertainty_expansion_under_tremor():
    config = Config()
    estimator = UncertaintyEstimator(config)

    normal_motion = MotionState(
        is_stationary=False,
        velocity=np.array([0.05, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        magnitude=0.05,
        confidence=0.9,
        movement_type=MovementType.INTENTIONAL
    )

    u_normal = estimator.update(np.array([0.05, 0.0, 0.0]), np.array([0.05, 0.0, 0.0]), normal_motion)

    # Tremor motion state with fluctuating noisy gyro inputs
    tremor_motion = MotionState(
        is_stationary=False,
        velocity=np.array([0.1, 0.2, 0.0]),
        direction=np.array([0.5, 0.8, 0.0]),
        magnitude=0.22,
        confidence=0.5,
        movement_type=MovementType.TREMOR
    )

    for i in range(10):
        noise = np.random.normal(0, 0.1, size=3)
        u_tremor = estimator.update(np.array([0.1, 0.2, 0.0]) + noise, np.array([0.1, 0.2, 0.0]), tremor_motion)

    assert u_tremor.uncertainty_radius > u_normal.uncertainty_radius
    assert u_tremor.confidence_score < u_normal.confidence_score
