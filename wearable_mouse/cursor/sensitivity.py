"""
Sensitivity Scaling & Acceleration module.
Maps angular hand motion rates (rad/s) into screen pixel velocity (pixels/s)
with non-linear acceleration curves and maximum speed caps.
"""

from typing import Optional, Tuple
import numpy as np
from wearable_mouse.config import Config


class SensitivityScaler:
    """
    Applies gain, deadzone, dynamic velocity acceleration, and maximum speed bounds.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def scale_motion(self, gyro_vector: np.ndarray, dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Convert 3D gyro rates (gx, gy, gz) into 2D screen coordinate delta (dx, dy).

        Hand Mapping:
        - Hand Yaw (gz): Horizontal cursor movement (Left/Right -> dx)
        - Hand Pitch (gy): Vertical cursor movement (Up/Down -> dy)

        Returns:
            Tuple of (dx, dy) pixel displacement for step dt.
        """
        step_dt = dt if dt is not None else self.config.dt
        gx, gy, gz = gyro_vector

        # Raw velocity candidates in pixels/sec
        # Note: Invert Pitch (gy) to match natural hand movement (pitch up -> move up)
        raw_vx = gz * self.config.base_sensitivity_x
        raw_vy = -gy * self.config.base_sensitivity_y

        speed = np.hypot(raw_vx, raw_vy)
        if speed < 1e-6:
            return 0.0, 0.0

        # Non-linear velocity acceleration curve
        if self.config.enable_acceleration and speed > 100.0:
            # Acceleration multiplier based on hand movement speed
            speed_ratio = speed / 500.0
            accel_mult = 1.0 + np.power(speed_ratio, self.config.acceleration_exponent - 1.0)
            raw_vx *= accel_mult
            raw_vy *= accel_mult
            speed = np.hypot(raw_vx, raw_vy)

        # Cap at maximum allowed cursor speed
        if speed > self.config.max_cursor_velocity:
            scale = self.config.max_cursor_velocity / speed
            raw_vx *= scale
            raw_vy *= scale

        # Calculate pixel displacement for time delta dt
        dx = raw_vx * step_dt
        dy = raw_vy * step_dt
        return dx, dy
