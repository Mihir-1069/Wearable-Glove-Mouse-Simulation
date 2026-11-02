"""
IMU Simulator module.
Generates synthetic 6-axis IMU telemetry incorporating physical hand dynamics,
involuntary hand tremor, sensor noise, zero-rate gyro bias, and linear gyro drift.
"""

from typing import Optional, Tuple
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample


class IMUSimulator:
    """
    Configurable 6-Axis IMU Sensor Simulator.

    Simulates realistic MPU6050 output dynamics:
    - True physical motion (3D acceleration + 3D angular velocity)
    - Gravitational baseline vector (9.81 m/s^2 along gravity axis)
    - Zero-rate gyro bias (constant offset)
    - Linear gyro bias drift over time
    - Sinusoidal & stochastic hand tremor (~8-12 Hz)
    - Additive White Gaussian Noise (AWGN)
    """

    GRAVITY = 9.81  # m/s^2

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.dt = self.config.dt
        self.current_time = 0.0

        # Motion inputs (true physical state)
        self.true_gyro = np.zeros(3)   # True angular velocity (rad/s)
        self.true_accel = np.array([0.0, 0.0, self.GRAVITY])  # True linear acceleration + gravity

        # Deterministic random generator initialization
        self.seed = self.config.random_seed
        self.rng = np.random.default_rng(self.seed)

        # Generate initial static gyro bias vector
        self.gyro_bias = self.rng.normal(0.0, self.config.gyro_bias_std, size=3)

    def set_seed(self, seed: int) -> None:
        """Reset random seed for deterministic reproduction."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.gyro_bias = self.rng.normal(0.0, self.config.gyro_bias_std, size=3)

    def set_motion(self, true_gyro: np.ndarray, true_accel: Optional[np.ndarray] = None) -> None:
        """
        Set true physical hand motion vectors.

        Args:
            true_gyro: [wx, wy, wz] angular rates in rad/s.
            true_accel: Optional [ax, ay, az] linear accel in m/s^2.
        """
        self.true_gyro = np.asarray(true_gyro, dtype=np.float64)
        if true_accel is not None:
            self.true_accel = np.asarray(true_accel, dtype=np.float64)

    def reset(self) -> None:
        """Reset timeline and state back to initial values."""
        self.current_time = 0.0
        self.true_gyro = np.zeros(3)
        self.true_accel = np.array([0.0, 0.0, self.GRAVITY])
        self.set_seed(self.seed)

    def step(self, dt: Optional[float] = None) -> IMUSample:
        """
        Advance simulator by one step dt and return synthetic IMUSample.
        """
        step_dt = dt if dt is not None else self.dt
        self.current_time += step_dt

        # 1. Linear Gyro Drift: bias increases slowly over time
        drift = self.config.gyro_drift_rate * self.current_time
        current_bias = self.gyro_bias + drift

        # 2. Hand Tremor Simulation: sinusoidal oscillating components at tremor_frequency
        t = self.current_time
        freq = self.config.tremor_frequency
        amp = self.config.tremor_amplitude
        # Multi-axis phase shifts for realistic 3D tremor orbit
        tremor_gyro = amp * np.array([
            np.sin(2 * np.pi * freq * t),
            np.cos(2 * np.pi * freq * t + 0.5),
            np.sin(2 * np.pi * (freq * 0.9) * t + 1.2)
        ])
        tremor_accel = (amp * 2.0) * np.array([
            np.cos(2 * np.pi * freq * t),
            np.sin(2 * np.pi * freq * t),
            0.0
        ])

        # 3. Additive Gaussian Sensor Noise
        gyro_noise = self.rng.normal(0.0, self.config.gyro_noise_std, size=3)
        accel_noise = self.rng.normal(0.0, self.config.accel_noise_std, size=3)

        # 4. Composite Synthetic Outputs
        sim_gyro = self.true_gyro + current_bias + tremor_gyro + gyro_noise
        sim_accel = self.true_accel + tremor_accel + accel_noise

        return IMUSample(
            timestamp=self.current_time,
            accel=sim_accel,
            gyro=sim_gyro
        )
