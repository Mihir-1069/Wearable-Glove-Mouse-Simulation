"""
BLE HID Mouse Output Adapter.
Formats interaction events into standard 4-byte BLE HID Mouse Reports
and dispatches them over a Bluetooth LE Transport interface.
"""

from dataclasses import dataclass
from typing import Optional, List, Callable
import struct
from wearable_mouse.interaction.base_output_adapter import BaseOutputAdapter


@dataclass
class BLEHIDReport:
    """
    Standard 4-byte BLE HID Mouse Input Report structure.
    Byte 0: Buttons bitmask (Bit 0: Left, Bit 1: Right, Bit 2: Middle)
    Byte 1: X displacement (-127 to +127 signed int8)
    Byte 2: Y displacement (-127 to +127 signed int8)
    Byte 3: Vertical Scroll Wheel (-127 to +127 signed int8)
    """
    buttons: int = 0
    dx: int = 0
    dy: int = 0
    wheel: int = 0

    def encode(self) -> bytes:
        """Pack 4-byte report: uint8 buttons, int8 dx, int8 dy, int8 wheel."""
        b_dx = max(-127, min(127, int(self.dx)))
        b_dy = max(-127, min(127, int(self.dy)))
        b_wheel = max(-127, min(127, int(self.wheel)))
        return struct.pack("<Bbbb", self.buttons & 0x07, b_dx, b_dy, b_wheel)

    @classmethod
    def decode(cls, data: bytes) -> 'BLEHIDReport':
        """Unpack 4-byte BLE HID report."""
        if len(data) != 4:
            raise ValueError(f"BLE HID report must be exactly 4 bytes, received {len(data)}")
        buttons, dx, dy, wheel = struct.unpack("<Bbbb", data)
        return cls(buttons=buttons, dx=dx, dy=dy, wheel=wheel)


class BLEHIDOutputAdapter(BaseOutputAdapter):
    """
    Platform abstraction layer formatting mouse events into BLE HID reports.
    Emits raw BLE HID report bytes to registered BLE transport callbacks.
    """

    def __init__(self, transport_callback: Optional[Callable[[bytes], None]] = None):
        self.transport_callback = transport_callback
        self.report_history: List[BLEHIDReport] = []
        self.is_paused = False
        self.current_buttons = 0  # Bit 0: Left, Bit 1: Right, Bit 2: Middle

    def _send_report(self, report: BLEHIDReport) -> None:
        if self.is_paused:
            return
        self.report_history.append(report)
        encoded = report.encode()
        if self.transport_callback:
            self.transport_callback(encoded)

    def move(self, dx: float, dy: float) -> None:
        b_dx = max(-127, min(127, int(round(dx))))
        b_dy = max(-127, min(127, int(round(dy))))
        report = BLEHIDReport(buttons=self.current_buttons, dx=b_dx, dy=b_dy, wheel=0)
        self._send_report(report)

    def left_click(self) -> None:
        # Press left button
        self.button_down("left")
        # Release left button
        self.button_up("left")

    def right_click(self) -> None:
        self.button_down("right")
        self.button_up("right")

    def double_click(self) -> None:
        self.left_click()
        self.left_click()

    def button_down(self, button: str = "left") -> None:
        mask = 0x01 if button == "left" else (0x02 if button == "right" else 0x04)
        self.current_buttons |= mask
        report = BLEHIDReport(buttons=self.current_buttons, dx=0, dy=0, wheel=0)
        self._send_report(report)

    def button_up(self, button: str = "left") -> None:
        mask = 0x01 if button == "left" else (0x02 if button == "right" else 0x04)
        self.current_buttons &= ~mask
        report = BLEHIDReport(buttons=self.current_buttons, dx=0, dy=0, wheel=0)
        self._send_report(report)

    def scroll(self, dx: float = 0.0, dy: float = 0.0) -> None:
        wheel_val = max(-127, min(127, int(round(dy))))
        report = BLEHIDReport(buttons=self.current_buttons, dx=0, dy=0, wheel=wheel_val)
        self._send_report(report)

    def zoom(self, value: float) -> None:
        # Simulated as wheel + modifier
        wheel_val = 5 if value > 0 else -5
        report = BLEHIDReport(buttons=self.current_buttons, dx=0, dy=0, wheel=wheel_val)
        self._send_report(report)

    def pause(self) -> None:
        self.is_paused = True

    def resume(self) -> None:
        self.is_paused = False
