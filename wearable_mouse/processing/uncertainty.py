"""
Uncertainty Model.
Estimates spatial acquisition uncertainty radius and motion confidence score
based on sensor noise, tremor energy, cursor velocity, and signal stability.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.processing.motion_detector import MotionState, MovementType


@dataclass
class UncertaintyState:
    uncertainty_radius: float  # Radius in screen pixels
    confidence_score: float    # Overall confidence (0.0 = high uncertainty, 1.0 = highly certain)
    tremor_energy: float       # Calculated tremor component energy
    noise_estimate: float      # Estimated sensor noise variance


class UncertaintyEstimator:
    """
    Computes dynamic spatial uncertainty circle for error-tolerant target acquisition.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.raw_mag_history: List[float] = []
        self.filtered_mag_history: List[float] = []
        self.window_size = 20

    def update(self, raw_gyro: np.ndarray, filtered_gyro: np.ndarray, motion_state: MotionState) -> UncertaintyState:
        """
        Calculates uncertainty radius and confidence score for the current frame.
        """
        raw_mag = float(np.linalg.norm(raw_gyro))
        filt_mag = float(np.linalg.norm(filtered_gyro))

        self.raw_mag_history.append(raw_mag)
        self.filtered_mag_history.append(filt_mag)
        if len(self.raw_mag_history) > self.window_size:
            self.raw_mag_history.pop(0)
            self.filtered_mag_history.pop(0)

        # 1. Noise estimate from difference between raw and filtered signal variance
        noise_est = abs(float(np.var(self.raw_mag_history)) - float(np.var(self.filtered_mag_history)))

        # 2. Tremor energy calculation (boosted when movement_type is TREMOR)
        tremor_factor = 1.0
        if motion_state.movement_type == MovementType.TREMOR:
            tremor_factor = 2.5
        tremor_energy = float(np.std(self.filtered_mag_history)) * tremor_factor

        # 3. Cursor speed expansion: fast movement increases pointing uncertainty radius
        speed = motion_state.magnitude

        # 4. Composite Uncertainty Radius Formula
        radius = (
            self.config.uncertainty_base_radius +
            self.config.uncertainty_noise_weight * noise_est +
            self.config.uncertainty_tremor_weight * tremor_energy +
            self.config.uncertainty_velocity_weight * (speed * self.config.base_sensitivity_x)
        )

        # Cap radius to configured maximum
        radius = min(self.config.uncertainty_max_radius, max(self.config.uncertainty_base_radius, radius))

        # 5. Confidence Score (inverse relationship to uncertainty radius)
        norm_uncertainty = (radius - self.config.uncertainty_base_radius) / (
            self.config.uncertainty_max_radius - self.config.uncertainty_base_radius + 1e-6
        )
        confidence = float(np.clip(1.0 - norm_uncertainty, 0.05, 1.0))

        return UncertaintyState(
            uncertainty_radius=radius,
            confidence_score=confidence,
            tremor_energy=tremor_energy,
            noise_estimate=noise_est
        )
