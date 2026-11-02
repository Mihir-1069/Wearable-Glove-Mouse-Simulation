"""
Output Adapter and Semantic Interaction Events Router.
Decouples interaction state machine intent from computer output execution using polymorphic BaseOutputAdapters.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List, Callable, Tuple

from wearable_mouse.interaction.interaction_state import UnifiedState
from wearable_mouse.interaction.base_output_adapter import BaseOutputAdapter
from wearable_mouse.interaction.adapters.virtual_output_adapter import VirtualOutputAdapter
from wearable_mouse.interaction.adapters.os_output_adapter import OSOutputAdapter


class SemanticEventType(Enum):
    CURSOR_MOVE = "CURSOR_MOVE"
    LEFT_CLICK = "LEFT_CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    DRAG_START = "DRAG_START"
    DRAG_MOVE = "DRAG_MOVE"
    DRAG_END = "DRAG_END"
    SCROLL = "SCROLL"
    ZOOM = "ZOOM"
    PRECISION_TOGGLE = "PRECISION_TOGGLE"
    PAUSE_TOGGLE = "PAUSE_TOGGLE"
    STATE_CHANGE = "STATE_CHANGE"


@dataclass
class SemanticEvent:
    event_type: SemanticEventType
    timestamp: float
    cursor_x: float
    cursor_y: float
    delta_x: float = 0.0
    delta_y: float = 0.0
    state: UnifiedState = UnifiedState.AIR
    payload: Optional[str] = None

    def __str__(self) -> str:
        return f"[{self.timestamp:.2f}s] {self.event_type.value} @ ({int(self.cursor_x)}, {int(self.cursor_y)}) [{self.state.value}]"


class OutputAdapter:
    """
    Polymorphic Output Router.
    Dispatches high-level SemanticEvents to registered BaseOutputAdapter targets
    (VirtualOutputAdapter, OSOutputAdapter, BLEHIDOutputAdapter, TestOutputAdapter, DebugOutputAdapter).
    """

    def __init__(self, enable_os_mouse: bool = False, primary_adapter: Optional[BaseOutputAdapter] = None):
        self.adapters: List[BaseOutputAdapter] = []

        if primary_adapter is not None:
            self.adapters.append(primary_adapter)
        else:
            self.virtual_adapter = VirtualOutputAdapter()
            self.adapters.append(self.virtual_adapter)

            if enable_os_mouse:
                try:
                    self.adapters.append(OSOutputAdapter())
                except Exception:
                    pass

        self.listeners: List[Callable[[SemanticEvent], None]] = []
        self.event_history: List[SemanticEvent] = []
        self.last_cursor_pos: Tuple[float, float] = (960.0, 540.0)

    def add_adapter(self, adapter: BaseOutputAdapter) -> None:
        """Register a new output adapter target."""
        if adapter not in self.adapters:
            self.adapters.append(adapter)

    def remove_adapter(self, adapter: BaseOutputAdapter) -> None:
        """Remove an output adapter target."""
        if adapter in self.adapters:
            self.adapters.remove(adapter)

    def add_listener(self, callback: Callable[[SemanticEvent], None]) -> None:
        """Register a callback listener for semantic events."""
        self.listeners.append(callback)

    def dispatch(self, event: SemanticEvent) -> None:
        """
        Record and dispatch a semantic interaction event across all registered output adapters.
        """
        self.event_history.append(event)
        self.last_cursor_pos = (event.cursor_x, event.cursor_y)

        # Notify registered listener callbacks
        for listener in self.listeners:
            try:
                listener(event)
            except Exception:
                pass

        # Dispatch polymorphically to registered BaseOutputAdapters
        for adapter in self.adapters:
            try:
                if event.event_type == SemanticEventType.CURSOR_MOVE:
                    adapter.move(event.delta_x, event.delta_y)
                elif event.event_type == SemanticEventType.LEFT_CLICK:
                    adapter.left_click()
                elif event.event_type == SemanticEventType.RIGHT_CLICK:
                    adapter.right_click()
                elif event.event_type == SemanticEventType.DOUBLE_CLICK:
                    adapter.double_click()
                elif event.event_type == SemanticEventType.DRAG_START:
                    adapter.button_down("left")
                elif event.event_type == SemanticEventType.DRAG_MOVE:
                    adapter.move(event.delta_x, event.delta_y)
                elif event.event_type == SemanticEventType.DRAG_END:
                    adapter.button_up("left")
                elif event.event_type == SemanticEventType.SCROLL:
                    adapter.scroll(event.delta_x, event.delta_y)
                elif event.event_type == SemanticEventType.ZOOM:
                    adapter.zoom(event.delta_y)
                elif event.event_type == SemanticEventType.PAUSE_TOGGLE:
                    if event.state == UnifiedState.PAUSED:
                        adapter.pause()
                    else:
                        adapter.resume()
            except Exception:
                pass

    def clear_history(self) -> None:
        """Clear recorded event history."""
        self.event_history.clear()
