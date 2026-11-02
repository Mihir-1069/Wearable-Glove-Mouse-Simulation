"""
Gesture simulator for keyboard and programmatic flex sensor event triggers.
"""

from typing import Optional, Callable, Dict
from wearable_mouse.gestures.gesture_map import GestureType
from wearable_mouse.config import Config


class GestureSimulator:
    """
    Simulates flex sensor gesture events using keyboard keys or manual software triggers.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.key_map: Dict[str, GestureType] = {
            self.config.key_left_click: GestureType.LEFT_CLICK,
            self.config.key_right_click: GestureType.RIGHT_CLICK,
            self.config.key_double_click: GestureType.DOUBLE_CLICK,
            self.config.key_scroll: GestureType.SCROLL_UP,
            self.config.key_drag: GestureType.DRAG_TOGGLE,
            self.config.key_zoom: GestureType.ZOOM_IN,
            self.config.key_pause.lower(): GestureType.PAUSE_TOGGLE,
        }
        self._pending_gesture: Optional[GestureType] = None

    def trigger_key(self, char: str) -> Optional[GestureType]:
        """
        Handle keypress char and return mapped GestureType if valid.
        """
        char_lower = char.lower()
        if char_lower in self.key_map:
            gesture = self.key_map[char_lower]
            self._pending_gesture = gesture
            return gesture
        return None

    def trigger_gesture(self, gesture: GestureType) -> None:
        """Programmatically queue a gesture event."""
        self._pending_gesture = gesture

    def pop_gesture(self) -> Optional[GestureType]:
        """Fetch and clear pending simulated gesture."""
        g = self._pending_gesture
        self._pending_gesture = None
        return g
