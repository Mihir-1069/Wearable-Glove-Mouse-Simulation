"""
Unified Interaction State Machine.
Coordinates cursor movement, Air/Rest postures, gestures, system pause authority,
uncertainty gating, dynamic friction precision mode, and structured transition logging.
"""

from typing import Optional, List
from wearable_mouse.config import Config
from wearable_mouse.interaction.interaction_state import UnifiedState, StateTransitionLog
from wearable_mouse.processing.calibration import CalibrationState
from wearable_mouse.processing.posture_reference import PostureInfo, PostureState
from wearable_mouse.gestures.gesture_map import GestureEvent, GestureType
from wearable_mouse.cursor.target_assistance import AssistanceResult
from wearable_mouse.processing.uncertainty import UncertaintyState
from wearable_mouse.cursor.cursor_engine import VirtualCursor


class InteractionStateMachine:
    """
    Globally authoritative, deterministic interaction state machine.
    Decoupled from direct OS APIs and underlying processing modules.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.current_state: UnifiedState = UnifiedState.AIR
        self.previous_active_state: UnifiedState = UnifiedState.AIR
        self.transition_logs: List[StateTransitionLog] = []

    def reset(self, initial_state: UnifiedState = UnifiedState.AIR) -> None:
        """Reset state machine to initial state."""
        self.current_state = initial_state
        self.previous_active_state = initial_state
        self.transition_logs.clear()

    def update(
        self,
        timestamp: float,
        calib_state: CalibrationState,
        posture_info: Optional[PostureInfo] = None,
        gesture_event: Optional[GestureEvent] = None,
        assist_result: Optional[AssistanceResult] = None,
        uncertainty: Optional[UncertaintyState] = None,
        cursor: Optional[VirtualCursor] = None
    ) -> UnifiedState:
        """
        Evaluate inputs and determine next UnifiedState deterministically.
        Logs every valid state transition.
        """
        uncert_r = uncertainty.uncertainty_radius if uncertainty else 15.0
        cur_x = cursor.x if cursor else 960.0
        cur_y = cursor.y if cursor else 540.0
        event_str = gesture_event.gesture.value if gesture_event else "FRAME_TICK"

        # ---------------------------------------------------------------------
        # 1. Authoritative PAUSED State Handling
        # ---------------------------------------------------------------------
        if gesture_event and gesture_event.gesture == GestureType.PAUSE_TOGGLE:
            if self.current_state != UnifiedState.PAUSED:
                self.previous_active_state = self.current_state
                self._transition_to(
                    timestamp,
                    UnifiedState.PAUSED,
                    "PAUSE_TOGGLE",
                    uncert_r,
                    cur_x,
                    cur_y,
                    payload="System Paused"
                )
                return self.current_state
            else:
                # Unpause resume
                resume_state = self.previous_active_state if self.previous_active_state != UnifiedState.PAUSED else UnifiedState.AIR
                if posture_info:
                    if posture_info.is_resting:
                        resume_state = UnifiedState.REST_MOVING if posture_info.intentional_rest_motion else UnifiedState.REST
                    elif posture_info.state == PostureState.AIR:
                        resume_state = UnifiedState.AIR
                self._transition_to(
                    timestamp,
                    resume_state,
                    "UNPAUSE_TOGGLE",
                    uncert_r,
                    cur_x,
                    cur_y,
                    payload="System Resumed"
                )
                return self.current_state

        if self.current_state == UnifiedState.PAUSED:
            return UnifiedState.PAUSED

        # ---------------------------------------------------------------------
        # 2. Calibration Baseline Check
        # ---------------------------------------------------------------------
        if calib_state != CalibrationState.CALIBRATED:
            self._transition_to(timestamp, UnifiedState.IDLE, "UNCALIBRATED", uncert_r, cur_x, cur_y)
            return self.current_state

        # ---------------------------------------------------------------------
        # 3. Determine Base Posture / Movement State
        # ---------------------------------------------------------------------
        base_state = UnifiedState.AIR
        if posture_info:
            if posture_info.is_transitioning and self.config.gesture_inhibit_on_transition:
                # Transition freeze window
                base_state = UnifiedState.AIR if posture_info.state == PostureState.AIR else UnifiedState.REST
            elif posture_info.is_resting:
                base_state = UnifiedState.REST_MOVING if posture_info.intentional_rest_motion else UnifiedState.REST
            else:
                base_state = UnifiedState.AIR

        # Precision mode check (dynamic friction active near target)
        if assist_result and assist_result.friction_gain < 1.0 and base_state in (UnifiedState.AIR, UnifiedState.REST_MOVING):
            base_state = UnifiedState.PRECISION

        # ---------------------------------------------------------------------
        # 4. Uncertainty & Transition Safety Gating
        # ---------------------------------------------------------------------
        is_high_uncertainty = (uncert_r > self.config.gesture_max_uncertainty_threshold)
        is_transition_frozen = (posture_info is not None and posture_info.is_transitioning and self.config.gesture_inhibit_on_transition)
        is_inhibited = is_high_uncertainty or is_transition_frozen

        # ---------------------------------------------------------------------
        # 5. Process Gesture Actions & Transitions
        # ---------------------------------------------------------------------
        target_state = base_state

        if gesture_event and not is_inhibited:
            g = gesture_event.gesture
            if g == GestureType.LEFT_CLICK:
                target_state = UnifiedState.LEFT_CLICK
            elif g == GestureType.RIGHT_CLICK:
                target_state = UnifiedState.RIGHT_CLICK
            elif g == GestureType.DOUBLE_CLICK:
                target_state = UnifiedState.DOUBLE_CLICK
            elif g in (GestureType.DRAG_START, GestureType.DRAG_MOVE, GestureType.DRAG_TOGGLE):
                target_state = UnifiedState.DRAG
            elif g == GestureType.DRAG_END:
                target_state = base_state
            elif g in (GestureType.SCROLL_UP, GestureType.SCROLL_DOWN):
                target_state = UnifiedState.SCROLL
            elif g in (GestureType.ZOOM_IN, GestureType.ZOOM_OUT):
                target_state = UnifiedState.ZOOM
            elif g == GestureType.GESTURE_CANCEL:
                target_state = base_state
        else:
            # Maintain active DRAG until explicit DRAG_END or posture change
            if self.current_state == UnifiedState.DRAG and gesture_event and gesture_event.gesture != GestureType.DRAG_END:
                target_state = UnifiedState.DRAG
            elif self.current_state in (UnifiedState.LEFT_CLICK, UnifiedState.RIGHT_CLICK, UnifiedState.DOUBLE_CLICK, UnifiedState.SCROLL, UnifiedState.ZOOM):
                target_state = base_state

        self._transition_to(timestamp, target_state, event_str, uncert_r, cur_x, cur_y)
        return self.current_state

    def _transition_to(
        self,
        timestamp: float,
        new_state: UnifiedState,
        event: str,
        uncertainty_radius: float,
        cursor_x: float,
        cursor_y: float,
        payload: Optional[str] = None
    ) -> None:
        """Helper to execute state transition and append structured transition log."""
        if new_state != self.current_state:
            log_entry = StateTransitionLog(
                timestamp=timestamp,
                old_state=self.current_state,
                event=event,
                new_state=new_state,
                uncertainty_radius=uncertainty_radius,
                cursor_x=cursor_x,
                cursor_y=cursor_y,
                payload=payload
            )
            self.transition_logs.append(log_entry)
            self.current_state = new_state
