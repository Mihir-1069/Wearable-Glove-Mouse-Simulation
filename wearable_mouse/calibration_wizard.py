"""
6-Stage Personal Calibration Wizard Architecture.
Orchestrates guided calibration stages to measure user movement, posture, resting noise,
and flex gestures, producing a validated UserProfile without modifying source code.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.profiles.profile_models import UserProfile
from wearable_mouse.profiles.user_profile_manager import UserProfileManager


class WizardStage(Enum):
    STAGE_A_STATIONARY = auto()
    STAGE_B_NEUTRAL_POSTURE = auto()
    STAGE_C_SENSITIVITY = auto()
    STAGE_D_AIR_REST = auto()
    STAGE_E_GESTURE = auto()
    STAGE_F_CONFIRMATION = auto()
    COMPLETED = auto()


@dataclass
class StageResult:
    stage: WizardStage
    success: bool
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class CalibrationWizard:
    """
    State machine executing 6-stage guided personal calibration.
    """

    def __init__(self, profile_name: str = "calibrated_user"):
        self.profile_name = profile_name
        self.current_stage: WizardStage = WizardStage.STAGE_A_STATIONARY
        self.profile_manager = UserProfileManager()

        # Calibration Buffers
        self._stationary_samples: List[IMUSample] = []
        self._neutral_samples: List[IMUSample] = []
        self._sweep_gyro_max: float = 0.0
        self._rest_accel_vars: List[float] = []
        self._rest_gyro_energies: List[float] = []
        self._air_gyro_energies: List[float] = []
        self._flex_open_vals: List[float] = []
        self._flex_close_vals: List[float] = []

        # Calibrated Output Values
        self.calibrated_gyro_bias = np.zeros(3)
        self.calibrated_sensitivity_x: float = 1200.0
        self.calibrated_sensitivity_y: float = 1200.0
        self.calibrated_rest_accel_var: float = 0.08
        self.calibrated_rest_gyro_var: float = 0.04
        self.calibrated_rest_activation: float = 0.045
        self.calibrated_rest_release: float = 0.025
        self.calibrated_flex_open: float = 0.25
        self.calibrated_flex_close: float = 0.70

        self.results_history: List[StageResult] = []
        self.final_profile: Optional[UserProfile] = None

    def process_sample_stage_a(self, sample: IMUSample) -> Optional[StageResult]:
        """
        Stage A: Stationary Calibration.
        Collects stationary samples to compute gyro bias and noise floor.
        """
        self._stationary_samples.append(sample)
        if len(self._stationary_samples) >= 100:
            gyros = np.array([s.gyro for s in self._stationary_samples])
            self.calibrated_gyro_bias = np.mean(gyros, axis=0)
            noise_std = float(np.std(gyros))

            res = StageResult(
                stage=WizardStage.STAGE_A_STATIONARY,
                success=True,
                summary=f"Stationary gyro bias estimated: {np.round(self.calibrated_gyro_bias, 4)}, noise_std={noise_std:.4f}",
                metrics={"gyro_bias": self.calibrated_gyro_bias.tolist(), "noise_std": noise_std}
            )
            self.results_history.append(res)
            self.current_stage = WizardStage.STAGE_B_NEUTRAL_POSTURE
            return res
        return None

    def process_sample_stage_b(self, sample: IMUSample) -> Optional[StageResult]:
        """
        Stage B: Neutral Hand Position.
        Captures baseline accelerometer gravity vector for posture reference.
        """
        self._neutral_samples.append(sample)
        if len(self._neutral_samples) >= 50:
            accels = np.array([s.accel for s in self._neutral_samples])
            mean_gravity = np.mean(accels, axis=0)
            norm_g = float(np.linalg.norm(mean_gravity))

            res = StageResult(
                stage=WizardStage.STAGE_B_NEUTRAL_POSTURE,
                success=True,
                summary=f"Neutral posture gravity captured: norm={norm_g:.2f} m/s^2",
                metrics={"mean_accel": mean_gravity.tolist(), "gravity_norm": norm_g}
            )
            self.results_history.append(res)
            self.current_stage = WizardStage.STAGE_C_SENSITIVITY
            return res
        return None

    def process_sample_stage_c(self, gyro_magnitude_rads: float) -> Optional[StageResult]:
        """
        Stage C: Cursor Sensitivity Calibration.
        Samples maximum comfortable wrist/arm angular velocity to tune sensitivity gain.
        """
        if gyro_magnitude_rads > self._sweep_gyro_max:
            self._sweep_gyro_max = gyro_magnitude_rads

        # After capturing range of motion sweep
        if self._sweep_gyro_max > 0.1:
            # Map user max motion (~1.5-3.0 rad/s) to comfortable target speed (~1800 px/s)
            target_px_speed = 1800.0
            computed_sens = max(600.0, min(3000.0, target_px_speed / self._sweep_gyro_max))
            self.calibrated_sensitivity_x = computed_sens
            self.calibrated_sensitivity_y = computed_sens

            res = StageResult(
                stage=WizardStage.STAGE_C_SENSITIVITY,
                success=True,
                summary=f"Sensitivity tuned: peak_gyro={self._sweep_gyro_max:.2f} rad/s -> sensitivity={computed_sens:.1f}",
                metrics={"peak_gyro": self._sweep_gyro_max, "sensitivity": computed_sens}
            )
            self.results_history.append(res)
            self.current_stage = WizardStage.STAGE_D_AIR_REST
            return res
        return None

    def process_sample_stage_d(
        self,
        rest_accel_var: float,
        rest_gyro_energy: float,
        air_gyro_energy: float
    ) -> Optional[StageResult]:
        """
        Stage D: Air/Rest Calibration.
        Measures resting surface noise floor vs air movement to compute thresholds.
        """
        self._rest_accel_vars.append(rest_accel_var)
        self._rest_gyro_energies.append(rest_gyro_energy)
        self._air_gyro_energies.append(air_gyro_energy)

        if len(self._rest_accel_vars) >= 50:
            mean_rest_accel_var = float(np.mean(self._rest_accel_vars))
            mean_rest_gyro = float(np.mean(self._rest_gyro_energies))
            mean_air_gyro = float(np.mean(self._air_gyro_energies))

            # Set activation cutoff above resting noise, release threshold below activation
            self.calibrated_rest_accel_var = max(0.02, min(0.30, mean_rest_accel_var * 2.0))
            self.calibrated_rest_gyro_var = max(0.01, min(0.15, mean_rest_gyro * 2.0))

            rest_floor = max(0.015, mean_rest_gyro * 1.5)
            self.calibrated_rest_activation = max(0.03, min(0.15, rest_floor * 1.8))
            self.calibrated_rest_release = max(0.015, self.calibrated_rest_activation * 0.5)

            res = StageResult(
                stage=WizardStage.STAGE_D_AIR_REST,
                success=True,
                summary=f"Air/Rest thresholds set: activation={self.calibrated_rest_activation:.3f}, release={self.calibrated_rest_release:.3f}",
                metrics={
                    "accel_var_thresh": self.calibrated_rest_accel_var,
                    "activation_thresh": self.calibrated_rest_activation,
                    "release_thresh": self.calibrated_rest_release
                }
            )
            self.results_history.append(res)
            self.current_stage = WizardStage.STAGE_E_GESTURE
            return res
        return None

    def process_sample_stage_e(self, flex_open: float, flex_close: float) -> Optional[StageResult]:
        """
        Stage E: Gesture Calibration.
        Samples flex sensor open vs closed values.
        """
        self._flex_open_vals.append(flex_open)
        self._flex_close_vals.append(flex_close)

        if len(self._flex_open_vals) >= 20:
            mean_open = float(np.mean(self._flex_open_vals))
            mean_close = float(np.mean(self._flex_close_vals))

            self.calibrated_flex_open = max(0.05, min(0.40, mean_open))
            self.calibrated_flex_close = max(0.50, min(0.95, mean_close))

            res = StageResult(
                stage=WizardStage.STAGE_E_GESTURE,
                success=True,
                summary=f"Flex thresholds calibrated: open={self.calibrated_flex_open:.2f}, close={self.calibrated_flex_close:.2f}",
                metrics={"flex_open": self.calibrated_flex_open, "flex_close": self.calibrated_flex_close}
            )
            self.results_history.append(res)
            self.current_stage = WizardStage.STAGE_F_CONFIRMATION
            return res
        return None

    def process_stage_f_confirmation(self) -> Tuple[UserProfile, str]:
        """
        Stage F: Confirmation & Profile Persistence.
        Constructs UserProfile from calibrated metrics and saves JSON config.
        """
        profile = UserProfile(
            name=self.profile_name,
            sensitivity=1.0,
            sensitivity_x=self.calibrated_sensitivity_x,
            sensitivity_y=self.calibrated_sensitivity_y,
            rest_accel_var_threshold=self.calibrated_rest_accel_var,
            rest_gyro_var_threshold=self.calibrated_rest_gyro_var,
            rest_activation_threshold=self.calibrated_rest_activation,
            rest_release_threshold=self.calibrated_rest_release
        )

        save_path = self.profile_manager.save_profile(profile)
        self.final_profile = profile
        self.current_stage = WizardStage.COMPLETED

        res = StageResult(
            stage=WizardStage.STAGE_F_CONFIRMATION,
            success=True,
            summary=f"Profile '{self.profile_name}' saved to {save_path}",
            metrics={"save_path": save_path, "profile": profile.to_dict()}
        )
        self.results_history.append(res)
        return profile, save_path
