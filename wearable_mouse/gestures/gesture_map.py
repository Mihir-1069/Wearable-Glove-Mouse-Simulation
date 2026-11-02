"""
Gesture mapping enums and event container objects.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict


class FingerName(Enum):
    THUMB = "THUMB"
    INDEX = "INDEX"
    MIDDLE = "MIDDLE"
    RING = "RING"
    LITTLE = "LITTLE"


class FingerState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    TOUCHING = "TOUCHING"
    UNKNOWN = "UNKNOWN"


@dataclass
class FingerPose:
    """
    Unified representation of all five fingers' physical state.
    """
    finger_states: Dict[FingerName, FingerState] = field(default_factory=dict)
    raw_flex: Dict[FingerName, float] = field(default_factory=dict)
    touch_contacts: Dict[str, bool] = field(default_factory=dict)
    stability_score: float = 1.0


class GestureType(Enum):
    LEFT_CLICK = "Left Click"
    RIGHT_CLICK = "Right Click"
    DOUBLE_CLICK = "Double Click"
    DRAG_START = "Drag Start"
    DRAG_MOVE = "Drag Move"
    DRAG_END = "Drag End"
    DRAG_TOGGLE = "Drag Toggle"
    SCROLL_UP = "Scroll Up"
    SCROLL_DOWN = "Scroll Down"
    ZOOM_IN = "Zoom In"
    ZOOM_OUT = "Zoom Out"
    GESTURE_CANCEL = "Gesture Cancel"
    NO_GESTURE = "No Gesture"
    PAUSE_TOGGLE = "Pause Toggle"


@dataclass
class GestureEvent:
    gesture: GestureType
    timestamp: float
    confidence: float = 1.0
    payload: Optional[str] = None


@dataclass
class GestureDiagnosticState:
    """
    Diagnostic state container for real-time telemetry rendering.
    """
    finger_states: Dict[str, str]
    candidate_gesture: str
    confirmed_gesture: str
    confidence: float
    cooldown_remaining: float
    system_state: str
    interaction_state: str

