"""
Cursor package initializer.
"""
from wearable_mouse.cursor.sensitivity import SensitivityScaler
from wearable_mouse.cursor.cursor_engine import VirtualCursor, CursorEngine
from wearable_mouse.cursor.target_assistance import Target, TargetAssistanceEngine, AcquisitionMode, AssistanceResult
from wearable_mouse.cursor.dynamic_friction import DynamicSensitivitySuppression, FrictionState

__all__ = [
    "SensitivityScaler",
    "VirtualCursor", "CursorEngine",
    "Target", "TargetAssistanceEngine", "AcquisitionMode", "AssistanceResult",
    "DynamicSensitivitySuppression", "FrictionState"
]
