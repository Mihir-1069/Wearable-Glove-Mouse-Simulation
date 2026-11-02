"""
Unified Interaction State definitions & StateTransitionLog container.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class UnifiedState(Enum):
    """
    Conceptual interaction states of the Wearable Computer Controller system.
    """
    IDLE = "IDLE"               # Uncalibrated or stationary baseline
    AIR = "AIR"                 # In-air relative cursor control flight
    REST = "REST"               # Resting on surface, stationary
    REST_MOVING = "REST_MOVING" # Resting on surface, intentional sliding movement
    LEFT_CLICK = "LEFT_CLICK"   # Single left click action
    RIGHT_CLICK = "RIGHT_CLICK" # Single right click action
    DOUBLE_CLICK = "DOUBLE_CLICK"# Double left click action
    DRAG = "DRAG"               # Active drag hold state
    SCROLL = "SCROLL"           # Active 2-finger scroll state
    ZOOM = "ZOOM"               # Active 3-finger zoom state
    PRECISION = "PRECISION"     # Dynamic sensitivity suppression / friction active near target
    PAUSED = "PAUSED"           # System paused (globally authoritative)


@dataclass
class StateTransitionLog:
    """
    Structured telemetry record for every deterministic state transition.
    """
    timestamp: float
    old_state: UnifiedState
    event: str
    new_state: UnifiedState
    uncertainty_radius: float
    cursor_x: float
    cursor_y: float
    payload: Optional[str] = None
