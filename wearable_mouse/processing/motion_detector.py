"""
Motion Detection & Classification module.
Analyzes processed IMU angular rates to classify current hand movement state.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List
import numpy as np
from wearable_mouse.config import Config


class MovementType(Enum):
    STATIONARY = auto()
    TREMOR = auto()
    INTENTIONAL = auto()
    RAPID = auto()


@dataclass
class MotionState:
    """
    Rich motion state analysis output.
    """
    is_stationary: bool
    velocity: np.ndarray          # 2D/3D velocity vector (rad/s or pixels/s)
    direction: np.ndarray         # Unit direction vector
    magnitude: float              # Scalar speed/rate magnitude
    confidence: float             # Motion confidence (0.0 to 1.0)
    movement_type: MovementType   # Classified movement type


class MotionDetector:
    """
    Classifies motion dynamics based on magnitude, variance, and frequency band energy.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.history: List[float] = []
        self.window_size = 15

    def update(self, filtered_gyro: np.ndarray, is_stationary_calib: bool = False) -> MotionState:
        """
        Classify current frame filtered gyro vector into MotionState.
        """
        mag = float(np.linalg.norm(filtered_gyro))
        self.history.append(mag)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Direction vector
        if mag > 1e-6:
            direction = filtered_gyro / mag
        else:
            direction = np.zeros_like(filtered_gyro)

        # Variance over recent window
        variance = float(np.var(self.history)) if len(self.history) >= 5 else 0.0

        # Classification thresholds
        if is_stationary_calib or mag < self.config.stationary_gyro_threshold:
            movement_type = MovementType.STATIONARY
            is_stationary = True
            confidence = max(0.0, 1.0 - mag / self.config.stationary_gyro_threshold)
        elif mag > 1.5:  # > 1.5 rad/s (~85 deg/s) is rapid movement
            movement_type = MovementType.RAPID
            is_stationary = False
            confidence = min(1.0, mag / 2.0)
        elif variance > 0.005 and mag < 0.2:
            # Micro oscillations with high variance = tremor / involuntary movement
            movement_type = MovementType.TREMOR
            is_stationary = False
            confidence = 0.6
        else:
            movement_type = MovementType.INTENTIONAL
            is_stationary = False
            confidence = 0.95

        return MotionState(
            is_stationary=is_stationary,
            velocity=filtered_gyro.copy(),
            direction=direction,
            magnitude=mag,
            confidence=confidence,
            movement_type=movement_type
        )
