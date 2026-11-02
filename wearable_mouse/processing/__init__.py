"""
Processing package initializer.
"""
from wearable_mouse.processing.calibration import StationaryDetector, Calibrator, CalibrationState
from wearable_mouse.processing.filters import BaseFilter, LowPassFilter, DeadZoneFilter, MovingAverageFilter, ExponentialSmoothingFilter, FilterPipeline
from wearable_mouse.processing.orientation import OrientationEstimator
from wearable_mouse.processing.motion_detector import MotionDetector, MotionState, MovementType
from wearable_mouse.processing.uncertainty import UncertaintyEstimator, UncertaintyState

__all__ = [
    "StationaryDetector", "Calibrator", "CalibrationState",
    "BaseFilter", "LowPassFilter", "DeadZoneFilter", "MovingAverageFilter", "ExponentialSmoothingFilter", "FilterPipeline",
    "OrientationEstimator",
    "MotionDetector", "MotionState", "MovementType",
    "UncertaintyEstimator", "UncertaintyState"
]
