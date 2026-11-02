"""
Debug Output Adapter.
Formats and prints computer output telemetry logs to stdout or diagnostic stream.
"""

import sys
from typing import TextIO
from wearable_mouse.interaction.base_output_adapter import BaseOutputAdapter


class DebugOutputAdapter(BaseOutputAdapter):
    """
    Diagnostic debug output adapter.
    Prints formatted event strings to specified stream (default sys.stdout).
    """

    def __init__(self, stream: TextIO = sys.stdout):
        self.stream = stream
        self.is_paused = False

    def _log(self, text: str) -> None:
        if not self.is_paused:
            self.stream.write(f"[DEBUG_OUTPUT] {text}\n")
            self.stream.flush()

    def move(self, dx: float, dy: float) -> None:
        self._log(f"MOVE dx={dx:.2f}, dy={dy:.2f}")

    def left_click(self) -> None:
        self._log("LEFT_CLICK")

    def right_click(self) -> None:
        self._log("RIGHT_CLICK")

    def double_click(self) -> None:
        self._log("DOUBLE_CLICK")

    def button_down(self, button: str = "left") -> None:
        self._log(f"BUTTON_DOWN ({button})")

    def button_up(self, button: str = "left") -> None:
        self._log(f"BUTTON_UP ({button})")

    def scroll(self, dx: float = 0.0, dy: float = 0.0) -> None:
        self._log(f"SCROLL dx={dx:.2f}, dy={dy:.2f}")

    def zoom(self, value: float) -> None:
        self._log(f"ZOOM val={value:.2f}")

    def pause(self) -> None:
        self.stream.write("[DEBUG_OUTPUT] PAUSE\n")
        self.stream.flush()
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False
        self.stream.write("[DEBUG_OUTPUT] RESUME\n")
        self.stream.flush()
