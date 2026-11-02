"""
Dataclasses for IMU and multi-sensor telemetry samples.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class IMUSample:
    """
    Single 6-axis IMU measurement frame.

    Attributes:
        timestamp: Time of sample in seconds.
        accel: Accelerometer readings (x, y, z) in m/s^2 or g.
        gyro: Gyroscope readings (x, y, z) in rad/s.
    """
    timestamp: float
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        # Enforce numpy arrays with float64
        self.accel = np.asarray(self.accel, dtype=np.float64)
        self.gyro = np.asarray(self.gyro, dtype=np.float64)

    @property
    def accel_magnitude(self) -> float:
        """Returns Euclidean norm of accelerometer vector."""
        return float(np.linalg.norm(self.accel))

    @property
    def gyro_magnitude(self) -> float:
        """Returns Euclidean norm of gyroscope vector."""
        return float(np.linalg.norm(self.gyro))


@dataclass
class SensorData:
    """
    Unified container for IMU and optional flex/pressure sensor inputs.
    """
    imu: IMUSample
    flex_sensors: Optional[Dict[str, float]] = None
