"""
Unit tests for gesture interpreter and system pause state machine.
"""

import pytest
from wearable_mouse.config import Config
from wearable_mouse.gestures.gesture_engine import GestureEngine, SystemState
from wearable_mouse.gestures.gesture_map import GestureType


def test_pause_state_machine():
    engine = GestureEngine()

    assert engine.system_state == SystemState.ACTIVE

    # Trigger click while ACTIVE
    evt1 = engine.process_gesture(GestureType.LEFT_CLICK, timestamp=1.0)
    assert evt1 is not None
    assert evt1.gesture == GestureType.LEFT_CLICK

    # Toggle Pause
    evt_pause = engine.process_gesture(GestureType.PAUSE_TOGGLE, timestamp=2.0)
    assert evt_pause is not None
    assert engine.system_state == SystemState.PAUSED

    # Trigger click while PAUSED (must return None)
    evt_blocked = engine.process_gesture(GestureType.LEFT_CLICK, timestamp=3.0)
    assert evt_blocked is None

    # Toggle Unpause
    evt_unpause = engine.process_gesture(GestureType.PAUSE_TOGGLE, timestamp=4.0)
    assert evt_unpause is not None
    assert engine.system_state == SystemState.ACTIVE
