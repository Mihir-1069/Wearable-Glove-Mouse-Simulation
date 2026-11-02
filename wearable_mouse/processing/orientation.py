"""
Orientation Estimation module.
Combines Accelerometer and Gyroscope telemetry via a Complementary Filter to track 3D hand orientation (roll, pitch, yaw).
"""

from typing import Optional, Tuple
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample


class OrientationEstimator:
    """
    Tracks hand orientation angles [roll, pitch, yaw] in radians.
    Uses Complementary Filter:
    - Gyro integration for high-frequency dynamic tracking
    - Accelerometer pitch/roll calculation for low-frequency tilt correction
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.alpha = self.config.complementary_alpha
        # Orientation state in radians: [roll, pitch, yaw]
        self.orientation = np.zeros(3)

    def reset(self) -> None:
        self.orientation = np.zeros(3)

    def update(self, sample: IMUSample, dt: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update orientation using IMUSample frame.

        Returns:
            Tuple of (orientation [roll, pitch, yaw], angular_velocity [gx, gy, gz]).
        """
        step_dt = dt if dt is not None else self.config.dt

        # 1. Accelerometer tilt angles (roll & pitch relative to gravity)
        ax, ay, az = sample.accel
        accel_roll = np.arctan2(ay, az)
        accel_pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))

        # 2. Integrate gyroscope angular rates
        gx, gy, gz = sample.gyro
        gyro_roll = self.orientation[0] + gx * step_dt
        gyro_pitch = self.orientation[1] + gy * step_dt
        gyro_yaw = self.orientation[2] + gz * step_dt  # Yaw cannot be absolute-corrected by accel gravity

        # 3. Complementary Filter blending
        roll = self.alpha * gyro_roll + (1.0 - self.alpha) * accel_roll
        pitch = self.alpha * gyro_pitch + (1.0 - self.alpha) * accel_pitch
        yaw = gyro_yaw

        self.orientation = np.array([roll, pitch, yaw])
        return self.orientation.copy(), sample.gyro.copy()
