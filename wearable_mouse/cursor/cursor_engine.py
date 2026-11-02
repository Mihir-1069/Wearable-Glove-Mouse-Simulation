"""
Virtual Cursor Engine module.
Tracks 2D virtual screen cursor coordinates, applies relative displacement,
and enforces screen edge boundaries.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from wearable_mouse.config import Config
from wearable_mouse.cursor.sensitivity import SensitivityScaler


@dataclass
class VirtualCursor:
    x: float
    y: float
    dx: float = 0.0
    dy: float = 0.0
    vx: float = 0.0
    vy: float = 0.0

    @property
    def pos(self) -> Tuple[float, float]:
        return self.x, self.y

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


class CursorEngine:
    """
    Core relative cursor positioning engine.
    Applies hand movement deltas to VirtualCursor state and bounds within virtual screen boundaries.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.scaler = SensitivityScaler(self.config)

        # Start cursor at screen center
        self.cursor = VirtualCursor(
            x=self.config.screen_width / 2.0,
            y=self.config.screen_height / 2.0
        )

    def set_position(self, x: float, y: float) -> None:
        """Manually force cursor position (useful for scenario resets)."""
        self.cursor.x = float(np.clip(x, 0, self.config.screen_width))
        self.cursor.y = float(np.clip(y, 0, self.config.screen_height))
        self.cursor.dx = 0.0
        self.cursor.dy = 0.0
        self.cursor.vx = 0.0
        self.cursor.vy = 0.0

    def update(
        self,
        filtered_gyro: np.ndarray,
        dt: Optional[float] = None,
        is_paused: bool = False,
        posture_info: Optional['PostureInfo'] = None
    ) -> VirtualCursor:
        """
        Process filtered gyro input and update VirtualCursor state.

        Args:
            filtered_gyro: [gx, gy, gz] filtered angular rates in rad/s.
            dt: Time delta step.
            is_paused: If True, cursor movement is disabled.
            posture_info: Posture state container (AIR, REST, TRANSITION).
        """
        step_dt = dt if dt is not None else self.config.dt

        if is_paused:
            self.cursor.dx = 0.0
            self.cursor.dy = 0.0
            self.cursor.vx = 0.0
            self.cursor.vy = 0.0
            return self.cursor

        # Use posture-gated effective gyro if posture_info is provided
        input_gyro = posture_info.effective_gyro if posture_info is not None else filtered_gyro

        # Calculate relative delta displacement
        dx, dy = self.scaler.scale_motion(input_gyro, step_dt)

        # New proposed coordinates
        new_x = self.cursor.x + dx
        new_y = self.cursor.y + dy

        # Clamp within virtual screen coordinates
        clamped_x = float(np.clip(new_x, 0, self.config.screen_width))
        clamped_y = float(np.clip(new_y, 0, self.config.screen_height))

        # Compute instant velocity
        vx = (clamped_x - self.cursor.x) / step_dt if step_dt > 0 else 0.0
        vy = (clamped_y - self.cursor.y) / step_dt if step_dt > 0 else 0.0

        self.cursor.x = clamped_x
        self.cursor.y = clamped_y
        self.cursor.dx = dx
        self.cursor.dy = dy
        self.cursor.vx = vx
        self.cursor.vy = vy

        return self.cursor
