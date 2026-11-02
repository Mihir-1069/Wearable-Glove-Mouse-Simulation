"""
Haptic Event Simulator & Hardware Dispatcher.
Translates system interaction events (edge bumps, clicks, pause state changes)
into haptic vibration patterns.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Callable


class HapticPattern(Enum):
    SHORT_PULSE = "Short Pulse (30ms)"
    DOUBLE_PULSE = "Double Pulse (30ms-20ms-30ms)"
    LONG_PULSE = "Long Pulse (150ms)"
    EDGE_BUMP = "Edge Bump (60ms)"
    ERROR_ALERT = "Error Alert (300ms)"


@dataclass
class HapticEvent:
    event_name: str
    pattern: HapticPattern
    intensity: float  # 0.0 to 1.0 (PWM duty cycle)
    timestamp: float


class HapticManager:
    """
    Manages haptic feedback queuing, simulation logging, and hardware notification callbacks.
    """

    def __init__(self):
        self.history: List[HapticEvent] = []
        self.listeners: List[Callable[[HapticEvent], None]] = []

    def add_listener(self, callback: Callable[[HapticEvent], None]) -> None:
        self.listeners.append(callback)

    def trigger(self, event_name: str, pattern: HapticPattern, intensity: float = 0.8, timestamp: float = 0.0) -> HapticEvent:
        """
        Trigger a haptic feedback vibration event.
        """
        evt = HapticEvent(
            event_name=event_name,
            pattern=pattern,
            intensity=intensity,
            timestamp=timestamp
        )
        self.history.append(evt)
        # Limit history log size
        if len(self.history) > 100:
            self.history.pop(0)

        # Notify registered callbacks (GUI dashboard log / future hardware serial driver)
        for listener in self.listeners:
            listener(evt)

        return evt

    def trigger_edge_bump(self, edge_name: str, timestamp: float = 0.0) -> HapticEvent:
        return self.trigger(f"EDGE_{edge_name}", HapticPattern.EDGE_BUMP, intensity=0.7, timestamp=timestamp)

    def trigger_click(self, timestamp: float = 0.0) -> HapticEvent:
        return self.trigger("CLICK", HapticPattern.SHORT_PULSE, intensity=0.9, timestamp=timestamp)

    def trigger_double_click(self, timestamp: float = 0.0) -> HapticEvent:
        return self.trigger("DOUBLE_CLICK", HapticPattern.DOUBLE_PULSE, intensity=1.0, timestamp=timestamp)

    def trigger_pause_change(self, is_paused: bool, timestamp: float = 0.0) -> HapticEvent:
        name = "PAUSED" if is_paused else "RESUMED"
        return self.trigger(name, HapticPattern.LONG_PULSE, intensity=0.5, timestamp=timestamp)
