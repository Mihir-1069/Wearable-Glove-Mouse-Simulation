"""
Unit tests for calibration and stationary detection.
"""

import numpy as np
import pytest
from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.processing.calibration import StationaryDetector, Calibrator, CalibrationState


def test_stationary_detector():
    config = Config()
    detector = StationaryDetector(config)

    # Static samples should yield True stationary state after window fills
    static_sample = IMUSample(timestamp=0.0, accel=[0.0, 0.0, 9.81], gyro=[0.001, 0.001, -0.001])
    for i in range(config.stationary_window_size + 5):
        is_stat = detector.update(static_sample)

    assert is_stat is True

    # Active motion sample should yield False stationary state
    motion_sample = IMUSample(timestamp=1.0, accel=[1.0, 2.0, 9.81], gyro=[0.5, -0.4, 0.8])
    for i in range(config.stationary_window_size):
        is_stat = detector.update(motion_sample)

    assert is_stat is False


def test_calibrator_gyro_bias_estimation():
    config = Config(calibration_duration=0.5, sample_rate=100.0)  # 50 samples
    calibrator = Calibrator(config)

    known_bias = np.array([0.05, -0.03, 0.02])

    calibrator.start_calibration()
    assert calibrator.state == CalibrationState.CALIBRATING

    for i in range(75):
        sample = IMUSample(timestamp=i*0.01, accel=[0.0, 0.0, 9.81], gyro=known_bias.copy())
        state = calibrator.update(sample)

    assert state == CalibrationState.CALIBRATED
    np.testing.assert_allclose(calibrator.gyro_bias, known_bias, atol=1e-3)

    # Apply calibration subtraction
    raw_sample = IMUSample(timestamp=1.0, accel=[0.0, 0.0, 9.81], gyro=known_bias + np.array([0.1, 0.0, 0.0]))
    calibrated = calibrator.apply(raw_sample)
    np.testing.assert_allclose(calibrated.gyro, [0.1, 0.0, 0.0], atol=1e-3)
