"""
Unit tests for VirtualCursor engine and sensitivity mapping.
"""

import numpy as np
import pytest
from wearable_mouse.config import Config
from wearable_mouse.cursor.cursor_engine import CursorEngine


def test_cursor_relative_movement():
    config = Config(screen_width=1920, screen_height=1080)
    engine = CursorEngine(config)

    # Initial position at screen center (960, 540)
    assert engine.cursor.pos == (960.0, 540.0)

    # Positive Yaw (gz) -> move right (dx > 0)
    # Negative Pitch (gy) -> pitch down -> move down (dy > 0)
    gyro_input = np.array([0.0, -0.2, 0.3])
    cursor = engine.update(gyro_input, dt=0.01)

    assert cursor.x > 960.0
    assert cursor.y > 540.0


def test_cursor_screen_bounding():
    config = Config(screen_width=1000, screen_height=1000)
    engine = CursorEngine(config)
    engine.set_position(990.0, 990.0)

    # Move heavily beyond screen right boundary
    gyro_input = np.array([0.0, 0.0, 5.0])
    cursor = engine.update(gyro_input, dt=1.0)

    assert cursor.x == 1000.0  # Clamped at max screen width


def test_cursor_pause_inhibition():
    config = Config()
    engine = CursorEngine(config)
    start_pos = engine.cursor.pos

    gyro_input = np.array([0.0, 0.5, 0.5])
    cursor = engine.update(gyro_input, dt=0.1, is_paused=True)

    assert cursor.pos == start_pos
    assert cursor.dx == 0.0 and cursor.dy == 0.0
