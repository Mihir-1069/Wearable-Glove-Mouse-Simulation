"""
Interaction package initializer.
"""
from wearable_mouse.interaction.mouse_controller import BaseMouseController, VirtualMouseController, OSMouseController
from wearable_mouse.interaction.edge_feedback import EdgeDetector, ScreenEdge
from wearable_mouse.interaction.haptic_events import HapticManager, HapticPattern
from wearable_mouse.interaction.interaction_state import UnifiedState, StateTransitionLog
from wearable_mouse.interaction.interaction_state_machine import InteractionStateMachine
from wearable_mouse.interaction.base_output_adapter import BaseOutputAdapter
from wearable_mouse.interaction.adapters.virtual_output_adapter import VirtualOutputAdapter
from wearable_mouse.interaction.adapters.os_output_adapter import OSOutputAdapter
from wearable_mouse.interaction.adapters.ble_hid_adapter import BLEHIDOutputAdapter, BLEHIDReport
from wearable_mouse.interaction.adapters.test_output_adapter import TestOutputAdapter, OutputLogEntry
from wearable_mouse.interaction.adapters.debug_output_adapter import DebugOutputAdapter
from wearable_mouse.interaction.output_adapter import SemanticEventType, SemanticEvent, OutputAdapter

__all__ = [
    "BaseMouseController", "VirtualMouseController", "OSMouseController",
    "EdgeDetector", "ScreenEdge",
    "HapticManager", "HapticPattern",
    "UnifiedState", "StateTransitionLog", "InteractionStateMachine",
    "BaseOutputAdapter", "VirtualOutputAdapter", "OSOutputAdapter",
    "BLEHIDOutputAdapter", "BLEHIDReport", "TestOutputAdapter", "OutputLogEntry", "DebugOutputAdapter",
    "SemanticEventType", "SemanticEvent", "OutputAdapter"
]
