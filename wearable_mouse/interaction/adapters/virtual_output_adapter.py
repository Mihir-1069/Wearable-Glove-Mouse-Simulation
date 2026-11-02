"""
Virtual Output Adapter.
Maintains in-memory virtual cursor coordinates (x, y) and click state
for visualizer dashboard rendering and headless simulation.
"""

from typing import Tuple, Optional
from wearable_mouse.interaction.base_output_adapter import BaseOutputAdapter


class VirtualOutputAdapter(BaseOutputAdapter):
    """
    Virtual cursor output adapter.
    Tracks virtual screen coordinates (x, y) bounded within screen dimensions.
    """

    def __init__(self, screen_width: float = 1920.0, screen_height: float = 1080.0, initial_x: float = 960.0, initial_y: float = 540.0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = initial_x
        self.y = initial_y
        self.is_paused = False
        self.buttons_held = {"left": False, "right": False, "middle": False}
        self.last_action: Optional[str] = None
        self.click_count = 0

    def move(self, dx: float, dy: float) -> None:
        if self.is_paused:
            return
        self.x = max(0.0, min(self.screen_width, self.x + dx))
        self.y = max(0.0, min(self.screen_height, self.y + dy))

    def left_click(self) -> None:
        if self.is_paused:
            return
        self.click_count += 1
        self.last_action = f"LEFT_CLICK at ({int(self.x)}, {int(self.y)})"

    def right_click(self) -> None:
        if self.is_paused:
            return
        self.click_count += 1
        self.last_action = f"RIGHT_CLICK at ({int(self.x)}, {int(self.y)})"

    def double_click(self) -> None:
        if self.is_paused:
            return
        self.click_count += 2
        self.last_action = f"DOUBLE_CLICK at ({int(self.x)}, {int(self.y)})"

    def button_down(self, button: str = "left") -> None:
        if self.is_paused:
            return
        self.buttons_held[button] = True
        self.last_action = f"BUTTON_DOWN ({button}) at ({int(self.x)}, {int(self.y)})"

    def button_up(self, button: str = "left") -> None:
        if self.is_paused:
            return
        self.buttons_held[button] = False
        self.last_action = f"BUTTON_UP ({button}) at ({int(self.x)}, {int(self.y)})"

    def scroll(self, dx: float = 0.0, dy: float = 0.0) -> None:
        if self.is_paused:
            return
        self.last_action = f"SCROLL (dx={dx}, dy={dy})"

    def zoom(self, value: float) -> None:
        if self.is_paused:
            return
        self.last_action = f"ZOOM ({value})"

    def pause(self) -> None:
        self.is_paused = True
        self.last_action = "PAUSE"

    def resume(self) -> None:
        self.is_paused = False
        self.last_action = "RESUME"
