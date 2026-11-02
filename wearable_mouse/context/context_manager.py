"""
Context Profile Manager.
Resolves interaction profiles based on 4-tier priority hierarchy and maps gesture actions dynamically.
"""

import json
import os
from typing import Dict, Any, Optional, List
from wearable_mouse.context.context_models import (
    AppCategory,
    SensitivityProfile,
    ContextProfile,
    Context
)
from wearable_mouse.context.app_detector import ApplicationDetector


class ContextProfileManager:
    """
    Context Manager resolving active profiles, gesture re-mappings, and sensitivity parameters.
    """

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "config", "context_profiles.json"
    )

    def __init__(
        self,
        config_path: Optional[str] = None,
        app_detector: Optional[ApplicationDetector] = None
    ):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.app_detector = app_detector or ApplicationDetector()

        self.profiles: Dict[str, ContextProfile] = {}
        self.app_mappings: Dict[str, str] = {}
        self.category_defaults: Dict[AppCategory, str] = {}
        self.default_profile_name: str = "general_default"
        self.default_category: AppCategory = AppCategory.GENERAL

        self.user_override_profile: Optional[str] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load profile configuration from JSON file."""
        if not os.path.exists(self.config_path):
            self._apply_builtin_fallback()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.default_profile_name = data.get("default_profile", "general_default")
            def_cat_str = data.get("default_category", "GENERAL")
            self.default_category = AppCategory(def_cat_str) if def_cat_str in AppCategory.__members__ else AppCategory.GENERAL

            self.app_mappings = data.get("app_mappings", {})

            cat_defs = data.get("category_defaults", {})
            for cat_str, prof_name in cat_defs.items():
                if cat_str in AppCategory.__members__:
                    self.category_defaults[AppCategory(cat_str)] = prof_name

            raw_profiles = data.get("profiles", {})
            for prof_name, pdata in raw_profiles.items():
                cat_val = pdata.get("category", "GENERAL")
                category = AppCategory(cat_val) if cat_val in AppCategory.__members__ else AppCategory.GENERAL
                sens_data = pdata.get("sensitivity", {})
                sens = SensitivityProfile(
                    speed_multiplier=float(sens_data.get("speed_multiplier", 1.0)),
                    precision_multiplier=float(sens_data.get("precision_multiplier", 1.0)),
                    scroll_sensitivity=float(sens_data.get("scroll_sensitivity", 1.0))
                )
                mappings = pdata.get("gesture_mappings", {})
                self.profiles[prof_name] = ContextProfile(
                    name=prof_name,
                    category=category,
                    sensitivity=sens,
                    gesture_mappings=mappings
                )
        except Exception:
            self._apply_builtin_fallback()

        if self.default_profile_name not in self.profiles:
            self._apply_builtin_fallback()

    def _apply_builtin_fallback(self) -> None:
        """Apply emergency built-in fallback profiles."""
        default_prof = ContextProfile(
            name="general_default",
            category=AppCategory.GENERAL,
            sensitivity=SensitivityProfile(1.0, 1.0, 1.0),
            gesture_mappings={
                "LEFT_CLICK": "LEFT_CLICK",
                "RIGHT_CLICK": "RIGHT_CLICK",
                "DOUBLE_CLICK": "DOUBLE_CLICK",
                "DRAG_START": "DRAG_START",
                "SCROLL_UP": "SCROLL_UP",
                "SCROLL_DOWN": "SCROLL_DOWN",
                "ZOOM_IN": "ZOOM_IN",
                "ZOOM_OUT": "ZOOM_OUT"
            }
        )
        self.profiles = {"general_default": default_prof}
        self.app_mappings = {}
        self.category_defaults = {AppCategory.GENERAL: "general_default"}
        self.default_profile_name = "general_default"
        self.default_category = AppCategory.GENERAL

    def set_user_override(self, profile_name: Optional[str]) -> bool:
        """
        Set or clear explicit user profile override.
        Returns True if profile is valid or cleared, False if profile_name is invalid.
        """
        if profile_name is None:
            self.user_override_profile = None
            return True

        if profile_name in self.profiles:
            self.user_override_profile = profile_name
            return True
        return False

    def clear_user_override(self) -> None:
        """Clear user override."""
        self.user_override_profile = None

    def resolve_context(self, interaction_mode: str = "DEFAULT") -> Context:
        """
        Resolve active Context according to 4-tier Priority Hierarchy:
        1. User Override
        2. Application-Specific Match
        3. Category Default Match
        4. Global Default (GENERAL)
        """
        active_app = self.app_detector.get_active_app().lower()
        profile: Optional[ContextProfile] = None
        is_override = False

        # Tier 1: User Override
        if self.user_override_profile and self.user_override_profile in self.profiles:
            profile = self.profiles[self.user_override_profile]
            is_override = True

        # Tier 2: Application-Specific Matching
        if profile is None and active_app in self.app_mappings:
            target_prof = self.app_mappings[active_app]
            if target_prof in self.profiles:
                profile = self.profiles[target_prof]

        # Tier 3: Category Matching (if app maps to category or category defaults)
        if profile is None:
            for cat, default_prof in self.category_defaults.items():
                if cat.value.lower() in active_app and default_prof in self.profiles:
                    profile = self.profiles[default_prof]
                    break

        # Tier 4: Global Default Fallback
        if profile is None:
            profile = self.profiles.get(self.default_profile_name, list(self.profiles.values())[0])

        return Context(
            active_app=active_app,
            category=profile.category,
            profile_name=profile.name,
            interaction_mode=interaction_mode,
            sensitivity=profile.sensitivity,
            is_user_override=is_override
        )

    def map_gesture(self, gesture_name: str, context: Optional[Context] = None) -> Optional[str]:
        """
        Map incoming gesture string (e.g. 'LEFT_CLICK', 'SCROLL_UP') to context target mapping.
        """
        ctx = context or self.resolve_context()
        profile = self.profiles.get(ctx.profile_name)
        if not profile:
            profile = self.profiles.get(self.default_profile_name, list(self.profiles.values())[0])

        return profile.gesture_mappings.get(gesture_name, gesture_name)

    def get_diagnostic_info(self, current_state: str = "AIR") -> Dict[str, Any]:
        """
        Return structured diagnostic dictionary for HUD rendering and debugging.
        """
        ctx = self.resolve_context(interaction_mode=current_state)
        profile = self.profiles.get(ctx.profile_name)
        mappings = profile.gesture_mappings if profile else {}

        return {
            "active_app": ctx.active_app,
            "active_profile": ctx.profile_name,
            "category": ctx.category.value,
            "is_override": ctx.is_user_override,
            "sensitivity": {
                "speed_multiplier": ctx.sensitivity.speed_multiplier,
                "precision_multiplier": ctx.sensitivity.precision_multiplier,
                "scroll_sensitivity": ctx.sensitivity.scroll_sensitivity,
            },
            "current_mapping": dict(mappings),
            "current_interaction_state": current_state
        }
