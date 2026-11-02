"""
Posture Reference & Re-Anchoring Module.
Tracks hand posture states (AIR, REST, TRANSITION), manages dual-threshold hysteresis movement detection,
re-anchors relative orientation references, and suppresses noise/drift while enabling continuous resting cursor control.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple
import numpy as np

from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample


class PostureState(Enum):
    AIR = "AIR"
    REST = "REST"
    TRANSITION = "TRANSITION"


class RestMovementClassification(Enum):
    REST_NOISE = "REST_NOISE"
    REST_INTENTIONAL_MOVEMENT = "REST_INTENTIONAL_MOVEMENT"
    AIR_MOVEMENT = "AIR_MOVEMENT"
    TRANSITION_FREEZE = "TRANSITION_FREEZE"


@dataclass
class PostureInfo:
    """
    Rich posture and re-anchoring state container.
    """
    state: PostureState
    classification: RestMovementClassification
    is_resting: bool
    is_transitioning: bool
    reference_orientation: np.ndarray      # Captured reference orientation [roll, pitch, yaw]
    effective_gyro: np.ndarray             # Gyro rates passed to cursor engine (rad/s)
    intentional_rest_motion: bool          # True if intentional movement is detected while resting
    transition_delta: Tuple[float, float]  # Discarded delta during posture transition
    transition_timestamp: float            # Timestamp of last posture transition
    raw_movement_mag: float = 0.0          # Raw sensor signal magnitude
    filtered_movement_mag: float = 0.0     # Filtered gyro magnitude (rad/s)
    velocity_mag: float = 0.0              # Estimated velocity magnitude
    uncertainty_level: float = 0.0         # Spatial uncertainty estimate (px)
    hysteresis_active: bool = False        # True if hysteresis state is currently active (latching ON)
    consecutive_above_count: int = 0       # Count of consecutive samples above threshold
    reference_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))


class PostureReferenceTracker:
    """
    Manages posture state classification (AIR, REST, TRANSITION), relative reference frame re-anchoring,
    dual-threshold hysteresis noise gating, and progressive reference tracking during resting movements.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.state = PostureState.AIR
        self.reference_orientation = np.zeros(3)
        self.last_orientation = np.zeros(3)

        # Sliding window history for posture variance analysis
        self.accel_history: List[np.ndarray] = []
        self.gyro_history: List[np.ndarray] = []

        # Transition stabilization state
        self.transition_start_time: float = 0.0
        self.last_transition_timestamp: float = 0.0

        # Dual-threshold Hysteresis & Persistence State
        self.is_moving_in_rest: bool = False
        self.consecutive_above_count: int = 0
        self.consecutive_below_count: int = 0

    def reset(self) -> None:
        """Reset posture state, hysteresis counters, and reference frames."""
        self.state = PostureState.AIR
        self.reference_orientation = np.zeros(3)
        self.last_orientation = np.zeros(3)
        self.accel_history.clear()
        self.gyro_history.clear()
        self.transition_start_time = 0.0
        self.is_moving_in_rest = False
        self.consecutive_above_count = 0
        self.consecutive_below_count = 0

    def force_state(self, state: PostureState, timestamp: float = 0.0) -> None:
        """Manually set posture state (useful for test setup)."""
        self.state = state
        if state == PostureState.TRANSITION:
            self.transition_start_time = timestamp
            self.last_transition_timestamp = timestamp

    def update(
        self,
        sample: IMUSample,
        current_orientation: np.ndarray,
        filtered_gyro: np.ndarray,
        dt: Optional[float] = None,
        uncertainty_radius: float = 15.0
    ) -> PostureInfo:
        """
        Processes current telemetry sample and returns PostureInfo state with dual-threshold hysteresis classification.
        """
        step_dt = dt if dt is not None else self.config.dt
        timestamp = sample.timestamp

        # Update sliding history windows
        self.accel_history.append(sample.accel.copy())
        self.gyro_history.append(filtered_gyro.copy())
        if len(self.accel_history) > self.config.rest_detection_window:
            self.accel_history.pop(0)
            self.gyro_history.pop(0)

        # Compute signal dynamics over window
        if len(self.accel_history) >= 3:
            accel_arr = np.array(self.accel_history)
            accel_var = float(np.mean(np.var(accel_arr, axis=0)))
        else:
            accel_var = 1.0

        raw_mag = float(np.linalg.norm(sample.gyro))
        filt_mag = float(np.linalg.norm(filtered_gyro))

        # Surface support check: accelerometer variance is low (hand supported on surface)
        is_surface_supported = (accel_var < self.config.rest_accel_var_threshold)

        # State transition machine
        if self.state == PostureState.TRANSITION:
            # Check if transition stabilization timer has completed
            if (timestamp - self.transition_start_time) >= self.config.transition_stabilization_duration:
                if is_surface_supported:
                    self.state = PostureState.REST
                else:
                    self.state = PostureState.AIR
                self.reference_orientation = current_orientation.copy()

        elif self.state == PostureState.AIR:
            if is_surface_supported and filt_mag < 0.1:
                # Trigger transition into REST
                self.state = PostureState.TRANSITION
                self.transition_start_time = timestamp
                self.last_transition_timestamp = timestamp
                self.reference_orientation = current_orientation.copy()
                self.is_moving_in_rest = False
                self.consecutive_above_count = 0
                self.consecutive_below_count = 0

        elif self.state == PostureState.REST:
            if not is_surface_supported:
                # Trigger transition out of REST back to AIR (hand lifted off surface)
                self.state = PostureState.TRANSITION
                self.transition_start_time = timestamp
                self.last_transition_timestamp = timestamp
                self.reference_orientation = current_orientation.copy()
                self.is_moving_in_rest = False
                self.consecutive_above_count = 0
                self.consecutive_below_count = 0

        # Dual-Threshold Hysteresis & Persistence Classifier (for REST state)
        if self.state == PostureState.REST:
            if filt_mag >= self.config.rest_activation_threshold:
                self.consecutive_above_count += 1
                self.consecutive_below_count = 0
                if self.consecutive_above_count >= self.config.rest_persistence_count:
                    self.is_moving_in_rest = True
            elif filt_mag <= self.config.rest_release_threshold:
                self.consecutive_below_count += 1
                self.consecutive_above_count = 0
                if self.consecutive_below_count >= self.config.rest_persistence_count:
                    self.is_moving_in_rest = False

        # Classification & Effective Gyro Determination
        effective_gyro = np.zeros(3)
        transition_delta = (0.0, 0.0)
        reference_delta = current_orientation - self.reference_orientation

        if self.state == PostureState.TRANSITION:
            classification = RestMovementClassification.TRANSITION_FREEZE
            effective_gyro = np.zeros(3)
            transition_delta = (filtered_gyro[2] * self.config.base_sensitivity_x * step_dt,
                                -filtered_gyro[1] * self.config.base_sensitivity_y * step_dt)

        elif self.state == PostureState.REST:
            if self.is_moving_in_rest:
                classification = RestMovementClassification.REST_INTENTIONAL_MOVEMENT
                effective_gyro = filtered_gyro.copy()
                # Progressive reference updating during continuous intentional resting motion
                self.reference_orientation = current_orientation.copy()
            else:
                classification = RestMovementClassification.REST_NOISE
                effective_gyro = np.zeros(3)

        elif self.state == PostureState.AIR:
            classification = RestMovementClassification.AIR_MOVEMENT
            effective_gyro = filtered_gyro.copy()

        self.last_orientation = current_orientation.copy()

        return PostureInfo(
            state=self.state,
            classification=classification,
            is_resting=(self.state == PostureState.REST),
            is_transitioning=(self.state == PostureState.TRANSITION),
            reference_orientation=self.reference_orientation.copy(),
            effective_gyro=effective_gyro,
            intentional_rest_motion=(self.state == PostureState.REST and self.is_moving_in_rest),
            transition_delta=transition_delta,
            transition_timestamp=self.last_transition_timestamp,
            raw_movement_mag=raw_mag,
            filtered_movement_mag=filt_mag,
            velocity_mag=filt_mag * self.config.base_sensitivity_x,
            uncertainty_level=uncertainty_radius,
            hysteresis_active=self.is_moving_in_rest,
            consecutive_above_count=self.consecutive_above_count,
            reference_delta=reference_delta
        )
