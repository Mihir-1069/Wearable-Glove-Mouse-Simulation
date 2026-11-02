"""
Context Models & Data Structures.
Defines AppCategory, SensitivityProfile, ContextProfile, and Context.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


class AppCategory(str, Enum):
    GENERAL = "GENERAL"
    BROWSER = "BROWSER"
    PRESENTATION = "PRESENTATION"
    CREATIVE = "CREATIVE"
    PRODUCTIVITY = "PRODUCTIVITY"
    MEDIA = "MEDIA"


@dataclass
class SensitivityProfile:
    """
    Sensitivity multipliers associated with a context profile.
    """
    speed_multiplier: float = 1.0
    precision_multiplier: float = 1.0
    scroll_sensitivity: float = 1.0


@dataclass
class ContextProfile:
    """
    Representation of an application interaction profile.
    """
    name: str
    category: AppCategory
    sensitivity: SensitivityProfile = field(default_factory=SensitivityProfile)
    gesture_mappings: Dict[str, str] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Context:
    """
    Active system context state container.
    """
    active_app: str
    category: AppCategory
    profile_name: str
    interaction_mode: str = "DEFAULT"
    sensitivity: SensitivityProfile = field(default_factory=SensitivityProfile)
    is_user_override: bool = False
