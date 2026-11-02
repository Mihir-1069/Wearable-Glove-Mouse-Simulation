"""
Mouse Controller module.
Provides VirtualMouseController for simulation visualization and OSMouseController for real host OS control.
Real OS mouse control is strictly disabled by default.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
from wearable_mouse.config import Config
from wearable_mouse.gestures.gesture_map import GestureEvent, GestureType


class BaseMouseController(ABC):
    """Abstract interface for mouse control execution."""

    @abstractmethod
    def move_to(self, x: float, y: float) -> None:
        pass

    @abstractmethod
    def execute_gesture(self, event: GestureEvent) -> None:
        pass


class VirtualMouseController(BaseMouseController):
    """
    In-memory virtual mouse controller.
    Logs actions without affecting host computer operating system mouse.
    """

    def __init__(self):
        self.current_pos = (0.0, 0.0)
        self.last_action: Optional[str] = None

    def move_to(self, x: float, y: float) -> None:
        self.current_pos = (x, y)

    def execute_gesture(self, event: GestureEvent) -> None:
        self.last_action = f"Virtual {event.gesture.value} at {self.current_pos}"


class OSMouseController(BaseMouseController):
    """
    Optional OS Mouse Controller using pyautogui/pynput.
    Requires explicit config.enable_os_mouse = True safety flag.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        if not self.config.enable_os_mouse:
            raise PermissionError("OS Mouse control is disabled in config. Set enable_os_mouse=True to enable.")

        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            self.pyautogui = pyautogui
        except ImportError:
            raise ImportError("pyautogui module is required for OSMouseController. Install via pip install pyautogui.")

    def move_to(self, x: float, y: float) -> None:
        self.pyautogui.moveTo(int(x), int(y))

    def execute_gesture(self, event: GestureEvent) -> None:
        g = event.gesture
        if g == GestureType.LEFT_CLICK:
            self.pyautogui.click(button='left')
        elif g == GestureType.RIGHT_CLICK:
            self.pyautogui.click(button='right')
        elif g == GestureType.DOUBLE_CLICK:
            self.pyautogui.doubleClick()
        elif g == GestureType.SCROLL_UP:
            self.pyautogui.scroll(100)
        elif g == GestureType.SCROLL_DOWN:
            self.pyautogui.scroll(-100)
