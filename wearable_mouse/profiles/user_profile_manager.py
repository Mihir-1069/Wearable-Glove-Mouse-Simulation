"""
User Profile Manager & Validation Engine.
Handles loading, saving, validation, clamping, and application of user configuration profiles.
"""

import json
import os
from typing import Dict, Any, Tuple, List, Optional
from wearable_mouse.config import Config
from wearable_mouse.profiles.profile_models import UserProfile, DefaultProfile, CustomProfile


class UserProfileManager:
    """
    Manager for persisting, validating, and applying user configuration profiles.
    """

    DEFAULT_PROFILES_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "user_profiles_store"
    )

    def __init__(self, profiles_dir: Optional[str] = None):
        self.profiles_dir = profiles_dir or self.DEFAULT_PROFILES_DIR
        os.makedirs(self.profiles_dir, exist_ok=True)
        self.active_profile: UserProfile = DefaultProfile()

    def create_profile(self, name: str, **kwargs) -> UserProfile:
        """Create a new CustomProfile with validated parameters."""
        default_dict = DefaultProfile().to_dict()
        default_dict.update(kwargs)
        default_dict["name"] = name
        _, _, validated_prof = self.validate_profile(default_dict)
        return validated_prof

    def validate_profile(self, data: Dict[str, Any]) -> Tuple[bool, List[str], UserProfile]:
        """
        Validate profile fields against acceptable boundaries.
        Auto-fills missing fields and repairs out-of-range values.
        """
        errors: List[str] = []
        is_valid = True
        default_profile = DefaultProfile()
        repaired_data = default_profile.to_dict()

        # Merge incoming data
        for k, v in data.items():
            if k in repaired_data:
                repaired_data[k] = v

        # Range Validations & Clamping
        name = str(repaired_data.get("name", "custom_user"))
        
        # Sensitivity
        sens = float(repaired_data.get("sensitivity", 1.0))
        if sens <= 0.0 or sens > 10.0:
            errors.append(f"sensitivity {sens} out of range (0, 10]; clamped to 1.0")
            sens = max(0.1, min(10.0, sens))
        repaired_data["sensitivity"] = sens

        sens_x = float(repaired_data.get("sensitivity_x", 1200.0))
        if sens_x <= 0.0 or sens_x > 10000.0:
            errors.append(f"sensitivity_x {sens_x} out of range (0, 10000]; clamped")
            sens_x = max(100.0, min(10000.0, sens_x))
        repaired_data["sensitivity_x"] = sens_x

        sens_y = float(repaired_data.get("sensitivity_y", 1200.0))
        if sens_y <= 0.0 or sens_y > 10000.0:
            errors.append(f"sensitivity_y {sens_y} out of range (0, 10000]; clamped")
            sens_y = max(100.0, min(10000.0, sens_y))
        repaired_data["sensitivity_y"] = sens_y

        # Deadzone
        dz = float(repaired_data.get("deadzone", 0.02))
        if dz < 0.0 or dz > 0.5:
            errors.append(f"deadzone {dz} out of range [0.0, 0.5]; clamped")
            dz = max(0.0, min(0.5, dz))
        repaired_data["deadzone"] = dz

        # Rest thresholds
        r_accel = float(repaired_data.get("rest_accel_var_threshold", 0.08))
        if r_accel <= 0.0 or r_accel > 2.0:
            errors.append(f"rest_accel_var_threshold {r_accel} invalid; reset to 0.08")
            r_accel = 0.08
        repaired_data["rest_accel_var_threshold"] = r_accel

        r_act = float(repaired_data.get("rest_activation_threshold", 0.045))
        r_rel = float(repaired_data.get("rest_release_threshold", 0.025))

        if r_rel >= r_act or r_rel <= 0.0:
            errors.append(f"rest_release_threshold ({r_rel}) must be > 0 and < activation ({r_act})")
            r_rel = max(0.005, r_act * 0.5)
            is_valid = False

        repaired_data["rest_activation_threshold"] = r_act
        repaired_data["rest_release_threshold"] = r_rel

        # Screen dimensions
        sw = int(repaired_data.get("screen_width", 1920))
        sh = int(repaired_data.get("screen_height", 1080))
        if sw <= 0 or sh <= 0:
            errors.append("screen_width and screen_height must be positive")
            sw = 1920 if sw <= 0 else sw
            sh = 1080 if sh <= 0 else sh
        repaired_data["screen_width"] = sw
        repaired_data["screen_height"] = sh

        if len(errors) > 0:
            is_valid = False

        validated_profile = UserProfile.from_dict(repaired_data)
        return is_valid, errors, validated_profile

    def save_profile(self, profile: UserProfile, path: Optional[str] = None) -> str:
        """Save UserProfile to JSON configuration file."""
        if not path:
            filename = f"{profile.name}.json"
            path = os.path.join(self.profiles_dir, filename)

        _, _, clean_profile = self.validate_profile(profile.to_dict())
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean_profile.to_dict(), f, indent=2)

        return path

    def load_profile(self, name_or_path: str) -> UserProfile:
        """
        Load UserProfile from JSON file or profile name.
        Falls back safely to DefaultProfile if file is invalid or missing.
        """
        target_path = name_or_path
        if not os.path.exists(target_path):
            target_path = os.path.join(self.profiles_dir, f"{name_or_path}.json")

        if not os.path.exists(target_path):
            return DefaultProfile()

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _, _, loaded_prof = self.validate_profile(data)
            self.active_profile = loaded_prof
            return loaded_prof
        except Exception:
            return DefaultProfile()

    def reset_to_default(self) -> UserProfile:
        """Reset active profile to DefaultProfile."""
        self.active_profile = DefaultProfile()
        return self.active_profile

    def apply_profile_to_config(self, profile: UserProfile, config: Optional[Config] = None) -> Config:
        """
        Map UserProfile parameters onto a Config object.
        """
        cfg = config or Config()

        cfg.base_sensitivity_x = profile.sensitivity_x * profile.sensitivity
        cfg.base_sensitivity_y = profile.sensitivity_y * profile.sensitivity
        cfg.enable_acceleration = profile.acceleration
        cfg.deadzone_threshold = profile.deadzone
        cfg.rest_accel_var_threshold = profile.rest_accel_var_threshold
        cfg.rest_gyro_energy_threshold = profile.rest_gyro_var_threshold
        cfg.rest_activation_threshold = profile.rest_activation_threshold
        cfg.rest_release_threshold = profile.rest_release_threshold
        cfg.gesture_debounce_duration = profile.gesture_debounce_duration
        cfg.gesture_min_stable_duration = profile.gesture_min_stable_duration
        cfg.double_click_max_interval = profile.double_click_interval
        cfg.gesture_max_uncertainty_threshold = profile.uncertainty_tolerance
        cfg.screen_width = profile.screen_width
        cfg.screen_height = profile.screen_height

        return cfg

    def get_diagnostic_display(self, profile: Optional[UserProfile] = None) -> Dict[str, Any]:
        """
        Return diagnostic summary dictionary for HUD telemetry rendering.
        """
        prof = profile or self.active_profile
        is_valid, errors, _ = self.validate_profile(prof.to_dict())

        return {
            "profile_name": prof.name,
            "status": "VALID" if is_valid else "CLAMPED_WITH_WARNINGS",
            "warnings": errors,
            "motion_parameters": {
                "sensitivity_master": prof.sensitivity,
                "sensitivity_x": prof.sensitivity_x,
                "sensitivity_y": prof.sensitivity_y,
                "acceleration": prof.acceleration,
                "deadzone": prof.deadzone,
                "friction_strength": prof.friction_strength,
            },
            "rest_parameters": {
                "accel_var_thresh": prof.rest_accel_var_threshold,
                "gyro_var_thresh": prof.rest_gyro_var_threshold,
                "activation_thresh": prof.rest_activation_threshold,
                "release_thresh": prof.rest_release_threshold,
            },
            "gesture_parameters": {
                "debounce_s": prof.gesture_debounce_duration,
                "stable_hold_s": prof.gesture_min_stable_duration,
                "double_click_s": prof.double_click_interval,
                "uncertainty_tol_px": prof.uncertainty_tolerance,
            },
            "screen": {
                "width": prof.screen_width,
                "height": prof.screen_height,
            }
        }
