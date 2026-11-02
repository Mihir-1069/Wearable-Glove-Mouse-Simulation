"""
Base Output Adapter Abstract Interface.
Defines the standard contract for computer interaction output targets:
virtual cursor, host OS mouse, BLE HID, test logger, and diagnostic debug output.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseOutputAdapter(ABC):
    """
    Abstract Base Class for all Wearable Glove Mouse output destinations.
    Decouples interaction engine logic from computer output execution.
    """

    @abstractmethod
    def move(self, dx: float, dy: float) -> None:
        """Move cursor relative by (dx, dy) pixels."""
        pass

    @abstractmethod
    def left_click(self) -> None:
        """Trigger single left click."""
        pass

    @abstractmethod
    def right_click(self) -> None:
        """Trigger single right click."""
        pass

    @abstractmethod
    def double_click(self) -> None:
        """Trigger double click."""
        pass

    @abstractmethod
    def button_down(self, button: str = "left") -> None:
        """Press and hold mouse button ('left', 'right', 'middle')."""
        pass

    @abstractmethod
    def button_up(self, button: str = "left") -> None:
        """Release mouse button ('left', 'right', 'middle')."""
        pass

    @abstractmethod
    def scroll(self, dx: float = 0.0, dy: float = 0.0) -> None:
        """Scroll wheel by (dx, dy) units."""
        pass

    @abstractmethod
    def zoom(self, value: float) -> None:
        """Zoom in (value > 0) or zoom out (value < 0)."""
        pass

    @abstractmethod
    def pause(self) -> None:
        """Handle system pause event."""
        pass

    @abstractmethod
    def resume(self) -> None:
        """Handle system resume event."""
        pass
