"""
Calibration and Stationary Detection module.
Estimates zero-rate gyro bias and accelerometer baseline gravity when hand is held static.
Prevents continuous gyro bias drift resets during active hand movement.
"""

from enum import Enum, auto
from typing import Optional, List
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample


class CalibrationState(Enum):
    UNCALIBRATED = auto()
    CALIBRATING = auto()
    CALIBRATED = auto()


class StationaryDetector:
    """
    Monitors recent sliding window of IMU samples to determine if hand is stationary.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.window_size = self.config.stationary_window_size
        self.gyro_history: List[np.ndarray] = []
        self.accel_history: List[np.ndarray] = []

    def update(self, sample: IMUSample) -> bool:
        """
        Add sample to sliding window and evaluate stationary state.
        Returns True if hand is currently stationary.
        """
        self.gyro_history.append(sample.gyro)
        self.accel_history.append(sample.accel)

        if len(self.gyro_history) > self.window_size:
            self.gyro_history.pop(0)
            self.accel_history.pop(0)

        if len(self.gyro_history) < self.window_size:
            return False

        # Calculate mean gyro magnitude and accel magnitude variance over window
        gyro_mags = [np.linalg.norm(g) for g in self.gyro_history]
        accel_mags = [np.linalg.norm(a) for a in self.accel_history]

        avg_gyro_mag = float(np.mean(gyro_mags))
        gyro_var = float(np.var(gyro_mags))
        accel_var = float(np.var(accel_mags))

        is_stationary = (
            (avg_gyro_mag < self.config.stationary_gyro_threshold or gyro_var < 0.001) and
            accel_var < self.config.stationary_accel_threshold and
            avg_gyro_mag < 0.3
        )
        return is_stationary


class Calibrator:
    """
    Manages initial calibration sequence.
    Collects static IMU samples over stationary_duration (default 2 seconds),
    computes zero-rate gyro bias offset and baseline gravity vector, then locks calibration.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.state = CalibrationState.UNCALIBRATED
        self.stationary_detector = StationaryDetector(self.config)

        self.calibration_samples: List[IMUSample] = []
        self.required_samples = int(self.config.calibration_duration * self.config.sample_rate)

        # Calibration outputs
        self.gyro_bias = np.zeros(3)
        self.accel_baseline = np.array([0.0, 0.0, 9.81])

    def start_calibration(self) -> None:
        """Trigger calibration mode."""
        self.state = CalibrationState.CALIBRATING
        self.calibration_samples.clear()

    def update(self, sample: IMUSample) -> CalibrationState:
        """
        Process incoming IMUSample during calibration sequence.
        """
        is_stationary = self.stationary_detector.update(sample)

        if self.state == CalibrationState.CALIBRATING:
            if is_stationary:
                self.calibration_samples.append(sample)
                if len(self.calibration_samples) >= self.required_samples:
                    # Complete calibration
                    gyros = np.array([s.gyro for s in self.calibration_samples])
                    accels = np.array([s.accel for s in self.calibration_samples])

                    self.gyro_bias = np.mean(gyros, axis=0)
                    self.accel_baseline = np.mean(accels, axis=0)
                    self.state = CalibrationState.CALIBRATED
            else:
                # Reset collection if movement occurred during calibration
                self.calibration_samples.clear()

        elif self.state == CalibrationState.UNCALIBRATED:
            # Auto-start calibration if stationary
            if is_stationary:
                self.start_calibration()
                self.calibration_samples.append(sample)

        return self.state

    def apply(self, sample: IMUSample) -> IMUSample:
        """
        Subtract estimated gyro bias from raw IMUSample.
        Only applies bias subtraction if calibration is complete.
        """
        if self.state == CalibrationState.CALIBRATED:
            calibrated_gyro = sample.gyro - self.gyro_bias
            return IMUSample(
                timestamp=sample.timestamp,
                accel=sample.accel,
                gyro=calibrated_gyro
            )
        return sample
