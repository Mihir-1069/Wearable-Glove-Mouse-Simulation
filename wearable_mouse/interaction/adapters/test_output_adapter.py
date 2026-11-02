"""
Test Output Adapter.
Records all output operations into structured call logs for automated testing assertions.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from wearable_mouse.interaction.base_output_adapter import BaseOutputAdapter


@dataclass
class OutputLogEntry:
    operation: str
    dx: float = 0.0
    dy: float = 0.0
    button: Optional[str] = None
    value: float = 0.0


class TestOutputAdapter(BaseOutputAdapter):
    """
    Test recording output adapter.
    Logs every operation to log_entries for unit and integration testing verification.
    """
    __test__ = False

    def __init__(self):
        self.log_entries: List[OutputLogEntry] = []
        self.is_paused = False
        self.current_x: float = 960.0
        self.current_y: float = 540.0

    def move(self, dx: float, dy: float) -> None:
        if self.is_paused:
            return
        self.current_x += dx
        self.current_y += dy
        self.log_entries.append(OutputLogEntry(operation="move", dx=dx, dy=dy))

    def left_click(self) -> None:
        if self.is_paused:
            return
        self.log_entries.append(OutputLogEntry(operation="left_click"))

    def right_click(self) -> None:
        if self.is_paused:
            return
        self.log_entries.append(OutputLogEntry(operation="right_click"))

    def double_click(self) -> None:
        if self.is_paused:
            return
        self.log_entries.append(OutputLogEntry(operation="double_click"))

    def button_down(self, button: str = "left") -> None:
        if self.is_paused:
            return
        self.log_entries.append(OutputLogEntry(operation="button_down", button=button))

    def button_up(self, button: str = "left") -> None:
        if self.is_paused:
            return
        self.log_entries.append(OutputLogEntry(operation="button_up", button=button))

    def scroll(self, dx: float = 0.0, dy: float = 0.0) -> None:
        if self.is_paused:
            return
        self.log_entries.append(OutputLogEntry(operation="scroll", dx=dx, dy=dy))

    def zoom(self, value: float) -> None:
        if self.is_paused:
            return
        self.log_entries.append(OutputLogEntry(operation="zoom", value=value))

    def pause(self) -> None:
        self.is_paused = True
        self.log_entries.append(OutputLogEntry(operation="pause"))

    def resume(self) -> None:
        self.is_paused = False
        self.log_entries.append(OutputLogEntry(operation="resume"))

    def clear(self) -> None:
        self.log_entries.clear()
