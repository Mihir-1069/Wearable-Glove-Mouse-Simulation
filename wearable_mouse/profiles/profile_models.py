"""
User Profile Models & Schema.
Defines UserProfile dataclass and DefaultProfile baseline generator supporting all 13 user parameters.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional


@dataclass
class UserProfile:
    """
    Persistent User Configuration Profile container.
    """
    name: str = "default_user"
    
    # 1-3. Movement & Sensitivity
    sensitivity: float = 1.0            # Master cursor sensitivity multiplier
    sensitivity_x: float = 1200.0      # Base X gyro sensitivity (rad/s -> px/s)
    sensitivity_y: float = 1200.0      # Base Y gyro sensitivity (rad/s -> px/s)
    acceleration: bool = True          # Enable non-linear acceleration
    friction_strength: float = 1.0     # Dynamic friction strength multiplier

    # 4-5. Filtering & Deadzone
    deadzone: float = 0.02             # Angular velocity deadzone cutoff (rad/s)

    # 6. Air/Rest Thresholds
    rest_accel_var_threshold: float = 0.08      # Accel variance for REST posture
    rest_gyro_var_threshold: float = 0.04       # Gyro energy threshold for REST posture
    rest_activation_threshold: float = 0.045    # Rest intentional motion activation threshold
    rest_release_threshold: float = 0.025       # Rest intentional motion release threshold

    # 7-11. Gesture Timings & Sensitivity
    gesture_debounce_duration: float = 0.04     # Finger state transition debouncing (s)
    gesture_min_stable_duration: float = 0.08   # Minimum stable pose hold duration (s)
    double_click_interval: float = 0.35         # Max inter-click time for double click (s)
    scroll_sensitivity: float = 1.0            # Vertical scroll velocity multiplier
    zoom_sensitivity: float = 1.0              # Zoom scaling multiplier

    # 12-13. Uncertainty & Display Bounds
    uncertainty_tolerance: float = 80.0         # Max uncertainty radius (px) before gesture inhibition
    screen_width: int = 1920                    # Screen width in pixels
    screen_height: int = 1080                   # Screen height in pixels

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Deserialize dictionary into UserProfile instance with safe type conversions."""
        kwargs = {}
        for field_name in cls.__dataclass_fields__:
            if field_name in data:
                val = data[field_name]
                field_type = cls.__dataclass_fields__[field_name].type
                if field_type == int:
                    kwargs[field_name] = int(val)
                elif field_type == float:
                    kwargs[field_name] = float(val)
                elif field_type == bool:
                    kwargs[field_name] = bool(val)
                else:
                    kwargs[field_name] = str(val)
        return cls(**kwargs)


class DefaultProfile(UserProfile):
    """
    Factory baseline UserProfile representing factory default system parameters.
    """
    def __init__(self):
        super().__init__(name="default_user")


class CustomProfile(UserProfile):
    """
    User-configured custom profile instance.
    """
    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
