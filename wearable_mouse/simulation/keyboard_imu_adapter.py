"""
Keyboard to IMU Signal Adapter.
Translates interactive WASD / Arrow keypresses into synthetic 3D angular velocity vectors (rad/s),
ensuring keyboard control passes through the complete IMU noise, tremor, filtering, and uncertainty pipeline.
"""

from typing import Dict, Optional, Set
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.simulation.imu_simulator import IMUSimulator
from wearable_mouse.sensors.imu_data import IMUSample


class KeyboardIMUAdapter:
    """
    Adapter converting keyboard directional key states into realistic physical IMU motion streams.
    """

    def __init__(self, simulator: IMUSimulator, config: Optional[Config] = None):
        self.simulator = simulator
        self.config = config or Config()
        self.active_keys: Set[str] = set()

        # Target gyro rates when keys are held (rad/s)
        self.base_rate: float = 0.4
        self.current_gyro: np.ndarray = np.zeros(3)
        self.smoothing_alpha: float = 0.25  # Smooth key acceleration / deceleration

    def key_down(self, key_char: str) -> None:
        """Register keypress down."""
        self.active_keys.add(key_char.lower())

    def key_up(self, key_char: str) -> None:
        """Register keypress release."""
        self.active_keys.discard(key_char.lower())

    def clear_keys(self) -> None:
        """Reset key state."""
        self.active_keys.clear()
        self.current_gyro = np.zeros(3)

    def update_step(self, dt: Optional[float] = None) -> IMUSample:
        """
        Calculates target gyro rates from active keypresses and steps IMUSimulator.
        """
        step_dt = dt if dt is not None else self.config.dt

        # Compute raw direction vector from key states
        # Horizontal Yaw (gz): 'd' or 'Right' -> move right (gz > 0); 'a' or 'Left' -> move left (gz < 0)
        # Vertical Pitch (gy): 'w' or 'Up' -> pitch up (gy < 0 -> cursor dy < 0); 's' or 'Down' -> pitch down (gy > 0 -> cursor dy > 0)
        wz_target = 0.0
        wy_target = 0.0

        if 'd' in self.active_keys or 'right' in self.active_keys:
            wz_target += self.base_rate
        if 'a' in self.active_keys or 'left' in self.active_keys:
            wz_target -= self.base_rate

        if 'w' in self.active_keys or 'up' in self.active_keys:
            wy_target += self.base_rate  # Pitch up (gy > 0) -> sensitivity maps -gy -> dy < 0 (Up)
        if 's' in self.active_keys or 'down' in self.active_keys:
            wy_target -= self.base_rate  # Pitch down (gy < 0) -> sensitivity maps -gy -> dy > 0 (Down)

        target_gyro = np.array([0.0, wy_target, wz_target])

        # Smooth key acceleration ramp
        self.current_gyro = self.smoothing_alpha * target_gyro + (1.0 - self.smoothing_alpha) * self.current_gyro

        # Set physical motion in IMUSimulator (which adds noise, bias drift, and 3D tremor)
        self.simulator.set_motion(self.current_gyro)
        return self.simulator.step(step_dt)
