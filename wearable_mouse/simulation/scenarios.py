"""
Predefined Motion Scenarios for testing and GUI simulation controls.
"""

from enum import Enum, auto
from typing import Generator, Tuple
import numpy as np
from wearable_mouse.simulation.imu_simulator import IMUSimulator
from wearable_mouse.sensors.imu_data import IMUSample


class MotionScenario(Enum):
    STATIONARY = "Stationary Hand"
    SLOW_ROTATION = "Slow Rotation"
    FAST_ROTATION = "Fast Rotation"
    LEFT_RIGHT_SWEEP = "Left-Right Motion Sweep"
    UP_DOWN_SWEEP = "Up-Down Motion Sweep"
    TREMOR_ONLY = "Tremor Only"
    HIGH_NOISE_DRIFT = "High Noise & Drift"
    TARGET_CLICK_SEQUENCE = "Target Acquisition Sequence"
    AIR_REST_CONTINUITY = "Air to Rest to Air Continuity"


class ScenarioGenerator:
    """
    Drives IMUSimulator through predefined structured motion patterns.
    """

    def __init__(self, simulator: IMUSimulator):
        self.simulator = simulator

    def run_scenario_step(self, scenario: MotionScenario, step_index: int) -> IMUSample:
        """
        Calculates motion state for step_index according to scenario pattern and steps simulator.
        """
        t = step_index * self.simulator.dt

        if scenario == MotionScenario.STATIONARY:
            gyro = np.zeros(3)
            accel = np.array([0.0, 0.0, 9.81])

        elif scenario == MotionScenario.SLOW_ROTATION:
            # Smooth low-frequency yaw rotation (0.2 Hz)
            gyro = np.array([0.0, 0.0, 0.2 * np.sin(2 * np.pi * 0.2 * t)])
            accel = np.array([0.0, 0.0, 9.81])

        elif scenario == MotionScenario.FAST_ROTATION:
            # Rapid 1.5 Hz rotation
            gyro = np.array([0.0, 0.0, 1.2 * np.sin(2 * np.pi * 1.5 * t)])
            accel = np.array([0.0, 0.0, 9.81])

        elif scenario == MotionScenario.LEFT_RIGHT_SWEEP:
            # Alternating left and right horizontal hand yaw sweeps
            wz = 0.5 * np.sin(2 * np.pi * 0.5 * t)
            gyro = np.array([0.0, 0.0, wz])
            accel = np.array([0.0, 0.0, 9.81])

        elif scenario == MotionScenario.UP_DOWN_SWEEP:
            # Pitch tilt movement up and down
            wy = 0.5 * np.sin(2 * np.pi * 0.5 * t)
            gyro = np.array([0.0, wy, 0.0])
            accel = np.array([0.0, 0.0, 9.81])

        elif scenario == MotionScenario.TREMOR_ONLY:
            # No intentional motion, simulator tremor handles output
            gyro = np.zeros(3)
            accel = np.array([0.0, 0.0, 9.81])

        elif scenario == MotionScenario.HIGH_NOISE_DRIFT:
            # Zero intentional motion with drift
            gyro = np.zeros(3)
            accel = np.array([0.0, 0.0, 9.81])

        elif scenario == MotionScenario.AIR_REST_CONTINUITY:
            # Scenario timeline:
            # 0-2s: AIR movement (wz = 0.3 rad/s)
            # 2-3s: Transition to REST (stationary low variance)
            # 3-5s: Stationary REST with background noise
            # 5-7s: Intentional movement while RESTing (wz = 0.15 rad/s)
            # 7-8s: Transition back to AIR (lifting hand, high motion spike)
            # 8-10s: Additional AIR movement (wy = 0.3 rad/s)
            cycle_time = t % 10.0
            if cycle_time < 2.0:
                gyro = np.array([0.0, 0.0, 0.3])
                accel = np.array([0.0, 0.0, 9.81])
            elif cycle_time < 3.0:
                gyro = np.zeros(3)
                accel = np.array([0.0, 0.0, 9.81])
            elif cycle_time < 5.0:
                gyro = np.zeros(3)
                accel = np.array([0.0, 0.0, 9.81])
            elif cycle_time < 7.0:
                gyro = np.array([0.0, 0.0, 0.15])
                accel = np.array([0.0, 0.0, 9.81])
            elif cycle_time < 8.0:
                gyro = np.array([0.8, 0.8, 0.0])  # Lift gesture rotation spike
                accel = np.array([1.5, 1.5, 9.81])
            else:
                gyro = np.array([0.0, 0.3, 0.0])
                accel = np.array([0.0, 0.0, 9.81])

        elif scenario == MotionScenario.TARGET_CLICK_SEQUENCE:
            # Simulated point-and-click movement phases:
            cycle_time = t % 10.0
            if cycle_time < 2.0:
                gyro = np.zeros(3)
            elif cycle_time < 4.0:
                gyro = np.array([0.0, 0.0, 0.4 * np.sin(np.pi * (cycle_time - 2.0) / 2.0)])
            elif cycle_time < 5.0:
                gyro = np.zeros(3)
            elif cycle_time < 7.0:
                gyro = np.array([0.0, -0.3, -0.4]) * np.sin(np.pi * (cycle_time - 5.0) / 2.0)
            else:
                gyro = np.zeros(3)
            accel = np.array([0.0, 0.0, 9.81])
        else:
            gyro = np.zeros(3)
            accel = np.array([0.0, 0.0, 9.81])

        self.simulator.set_motion(gyro, accel)
        return self.simulator.step()
