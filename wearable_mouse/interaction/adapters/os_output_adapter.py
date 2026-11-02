"""
Host Operating System Mouse Output Adapter.
Executes real OS cursor and mouse events via pyautogui or pynput.
Strictly disabled by default for safety unless enable_os_mouse flag is explicitly enabled.
"""

from typing import Optional
from wearable_mouse.config import Config
from wearable_mouse.interaction.base_output_adapter import BaseOutputAdapter


class OSOutputAdapter(BaseOutputAdapter):
    """
    Host OS Mouse Output Adapter.
    Forwards interaction events to operating system APIs.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.is_paused = False

        if not self.config.enable_os_mouse:
            raise PermissionError("OS Mouse control is disabled in config. Set enable_os_mouse=True to enable.")

        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            self.pyautogui = pyautogui
        except ImportError:
            raise ImportError("pyautogui module is required for OSOutputAdapter. Install via pip install pyautogui.")

    def move(self, dx: float, dy: float) -> None:
        if self.is_paused:
            return
        self.pyautogui.moveRel(int(dx), int(dy))

    def left_click(self) -> None:
        if self.is_paused:
            return
        self.pyautogui.click(button='left')

    def right_click(self) -> None:
        if self.is_paused:
            return
        self.pyautogui.click(button='right')

    def double_click(self) -> None:
        if self.is_paused:
            return
        self.pyautogui.doubleClick()

    def button_down(self, button: str = "left") -> None:
        if self.is_paused:
            return
        self.pyautogui.mouseDown(button=button)

    def button_up(self, button: str = "left") -> None:
        if self.is_paused:
            return
        self.pyautogui.mouseUp(button=button)

    def scroll(self, dx: float = 0.0, dy: float = 0.0) -> None:
        if self.is_paused:
            return
        self.pyautogui.scroll(int(dy * 100))

    def zoom(self, value: float) -> None:
        if self.is_paused:
            return
        # Zoom simulated via Ctrl+Scroll or hotkey
        key = 'num+' if value > 0 else 'num-'
        self.pyautogui.hotkey('ctrl', key)

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False
