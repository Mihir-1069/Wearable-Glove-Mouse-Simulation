"""
Gesture Engine & Global Pause State Machine.
Enforces system ACTIVE vs PAUSED states, evaluates finger poses with temporal stability,
resolves priority conflicts, enforces cancellation rules, and emits clean semantic GestureEvents.
"""

from enum import Enum, auto
from typing import Optional, List, Callable, Dict
from wearable_mouse.config import Config
from wearable_mouse.gestures.gesture_map import (
    GestureType,
    GestureEvent,
    FingerName,
    FingerState,
    FingerPose,
    GestureDiagnosticState,
)
from wearable_mouse.gestures.finger_tracker import FingerTracker
from wearable_mouse.sensors.imu_data import SensorData
from wearable_mouse.processing.posture_reference import PostureInfo, PostureState
from wearable_mouse.processing.uncertainty import UncertaintyState


class SystemState(Enum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"


class InteractionState(Enum):
    NEUTRAL = "NEUTRAL"
    PINCH_CANDIDATE = "PINCH_CANDIDATE"
    DRAGGING = "DRAGGING"
    SCROLLING = "SCROLLING"
    ZOOMING = "ZOOMING"
    COOLDOWN = "COOLDOWN"
    INHIBITED = "INHIBITED"


class GestureEngine:
    """
    Complete, decoupled gesture interaction engine with temporal validation,
    deterministic priority arbitration, cancellation rules, and diagnostic telemetry.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.system_state = SystemState.ACTIVE
        self.interaction_state = InteractionState.NEUTRAL
        self.is_dragging = False

        # Finger tracker
        self.finger_tracker = FingerTracker(self.config)

        # Temporal tracking state
        self.candidate_gesture = GestureType.NO_GESTURE
        self.candidate_start_time: float = 0.0
        self.last_action_time: float = -100.0
        self.last_click_time: float = -100.0
        self.last_click_type: Optional[GestureType] = None
        self.last_emitted_event: Optional[GestureEvent] = None

        # Confirmed pose history
        self.current_pose: Optional[FingerPose] = None

    def toggle_pause(self) -> SystemState:
        """Toggle system between ACTIVE and PAUSED states."""
        if self.system_state == SystemState.ACTIVE:
            self.system_state = SystemState.PAUSED
            self.is_dragging = False
            self.interaction_state = InteractionState.NEUTRAL
        else:
            self.system_state = SystemState.ACTIVE
            self.interaction_state = InteractionState.NEUTRAL
        return self.system_state

    def process_gesture(self, gesture: GestureType, timestamp: float) -> Optional[GestureEvent]:
        """
        Process raw manual/simulated gesture trigger (keyboard keypresses) for backward compatibility.
        """
        if gesture == GestureType.PAUSE_TOGGLE:
            new_state = self.toggle_pause()
            evt = GestureEvent(
                gesture=GestureType.PAUSE_TOGGLE,
                timestamp=timestamp,
                payload=f"System set to {new_state.value}"
            )
            self.last_emitted_event = evt
            return evt

        if self.system_state == SystemState.PAUSED:
            return None

        if gesture == GestureType.DRAG_TOGGLE:
            self.is_dragging = not self.is_dragging
            payload = "DRAG_ACTIVE" if self.is_dragging else "DRAG_RELEASED"
            evt = GestureEvent(gesture=gesture, timestamp=timestamp, payload=payload)
            self.last_emitted_event = evt
            return evt

        evt = GestureEvent(gesture=gesture, timestamp=timestamp)
        self.last_emitted_event = evt
        return evt

    def process_frame(
        self,
        sensor_data: SensorData,
        timestamp: float,
        posture_info: Optional[PostureInfo] = None,
        uncertainty: Optional[UncertaintyState] = None
    ) -> Optional[GestureEvent]:
        """
        Process unified SensorData frame (IMU + flex/touch sensors), evaluate finger pose,
        apply posture/uncertainty inhibition, resolve priority hierarchy, and emit semantic GestureEvents.
        """
        # 1. Update finger tracker
        self.current_pose = self.finger_tracker.update(sensor_data.flex_sensors, timestamp)
        pose = self.current_pose

        # 2. Check PAUSE state
        if self.system_state == SystemState.PAUSED:
            if self.is_dragging:
                self.is_dragging = False
                self.interaction_state = InteractionState.NEUTRAL
                evt = GestureEvent(gesture=GestureType.DRAG_END, timestamp=timestamp, payload="Paused during drag")
                self.last_emitted_event = evt
                return evt
            self.interaction_state = InteractionState.NEUTRAL
            return None

        # 3. Check Posture & Uncertainty Inhibition Rules
        is_inhibited = False
        inhibit_reason = ""

        if posture_info is not None:
            # Inhibit during AIR <-> REST posture transition stabilization freeze
            if posture_info.is_transitioning and self.config.gesture_inhibit_on_transition:
                is_inhibited = True
                inhibit_reason = "Posture transition freeze"

        if uncertainty is not None:
            # Inhibit under high spatial uncertainty (high tremor / rapid motion)
            if uncertainty.uncertainty_radius > self.config.gesture_max_uncertainty_threshold:
                is_inhibited = True
                inhibit_reason = f"High uncertainty ({int(uncertainty.uncertainty_radius)}px)"

        if is_inhibited:
            # If an active drag or candidate gesture was in progress, cancel it cleanly
            if self.is_dragging:
                self.is_dragging = False
                self.interaction_state = InteractionState.INHIBITED
                evt = GestureEvent(gesture=GestureType.DRAG_END, timestamp=timestamp, payload=f"Cancelled: {inhibit_reason}")
                self.last_emitted_event = evt
                return evt
            elif self.candidate_gesture != GestureType.NO_GESTURE:
                self.candidate_gesture = GestureType.NO_GESTURE
                self.interaction_state = InteractionState.INHIBITED
                evt = GestureEvent(gesture=GestureType.GESTURE_CANCEL, timestamp=timestamp, payload=f"Cancelled: {inhibit_reason}")
                self.last_emitted_event = evt
                return evt
            self.interaction_state = InteractionState.INHIBITED
            return None

        # 4. Classify Candidate Gesture Intent from FingerPose
        raw_candidate = self._classify_pose_to_gesture(pose)

        # 5. Candidate Gesture State Machine & Stability Validation
        if raw_candidate != self.candidate_gesture:
            prev_candidate = self.candidate_gesture
            self.candidate_gesture = raw_candidate
            self.candidate_start_time = timestamp

            # Handle cancellation if transitioning away from active pinch/hold pose
            if prev_candidate != GestureType.NO_GESTURE and raw_candidate == GestureType.NO_GESTURE:
                if self.is_dragging:
                    self.is_dragging = False
                    self.interaction_state = InteractionState.NEUTRAL
                    evt = GestureEvent(gesture=GestureType.DRAG_END, timestamp=timestamp, payload="Pinch released")
                    self.last_emitted_event = evt
                    return evt
                elif (timestamp - self.candidate_start_time) < self.config.gesture_min_stable_duration:
                    # Premature release before min_stable_duration -> emit GESTURE_CANCEL
                    self.interaction_state = InteractionState.NEUTRAL
                    evt = GestureEvent(gesture=GestureType.GESTURE_CANCEL, timestamp=timestamp, payload="Accidental short contact cancelled")
                    self.last_emitted_event = evt
                    return evt

        candidate_stable_duration = timestamp - self.candidate_start_time

        # 6. Priority Arbitration & Action Emission
        # Handle active DRAG state progression
        if self.is_dragging:
            if self.candidate_gesture in (GestureType.LEFT_CLICK, GestureType.DRAG_START, GestureType.DRAG_MOVE):
                # Continuous drag holding
                self.interaction_state = InteractionState.DRAGGING
                evt = GestureEvent(gesture=GestureType.DRAG_MOVE, timestamp=timestamp, payload="DRAG_HOLDING")
                self.last_emitted_event = evt
                return evt
            else:
                # Drag released or interrupted by conflicting pose
                self.is_dragging = False
                self.interaction_state = InteractionState.NEUTRAL
                evt = GestureEvent(gesture=GestureType.DRAG_END, timestamp=timestamp, payload="DRAG_RELEASED")
                self.last_emitted_event = evt
                return evt

        # Check if candidate pose has achieved minimum stable duration
        if candidate_stable_duration >= self.config.gesture_min_stable_duration:
            # --- Priority 1: SCROLL / ZOOM ---
            if self.candidate_gesture in (GestureType.SCROLL_UP, GestureType.SCROLL_DOWN):
                if (timestamp - self.last_action_time) < self.config.gesture_cooldown_duration:
                    self.interaction_state = InteractionState.COOLDOWN
                    return None
                self.interaction_state = InteractionState.SCROLLING
                self.last_action_time = timestamp
                evt = GestureEvent(gesture=self.candidate_gesture, timestamp=timestamp, confidence=pose.stability_score)
                self.last_emitted_event = evt
                return evt

            elif self.candidate_gesture in (GestureType.ZOOM_IN, GestureType.ZOOM_OUT):
                if (timestamp - self.last_action_time) < self.config.gesture_cooldown_duration:
                    self.interaction_state = InteractionState.COOLDOWN
                    return None
                self.interaction_state = InteractionState.ZOOMING
                self.last_action_time = timestamp
                evt = GestureEvent(gesture=self.candidate_gesture, timestamp=timestamp, confidence=pose.stability_score)
                self.last_emitted_event = evt
                return evt

            # --- Priority 2: RIGHT_CLICK ---
            elif self.candidate_gesture == GestureType.RIGHT_CLICK:
                if (timestamp - self.last_action_time) < self.config.gesture_cooldown_duration:
                    self.interaction_state = InteractionState.COOLDOWN
                    return None
                self.interaction_state = InteractionState.NEUTRAL
                self.last_action_time = timestamp
                self.candidate_gesture = GestureType.NO_GESTURE
                evt = GestureEvent(gesture=GestureType.RIGHT_CLICK, timestamp=timestamp, confidence=pose.stability_score)
                self.last_emitted_event = evt
                return evt

            # --- Priority 3: LEFT_CLICK / DOUBLE_CLICK / DRAG_START ---
            elif self.candidate_gesture == GestureType.LEFT_CLICK:
                # Check for sustained pinch -> DRAG_START
                if candidate_stable_duration >= self.config.drag_hold_duration:
                    self.is_dragging = True
                    self.interaction_state = InteractionState.DRAGGING
                    self.last_action_time = timestamp
                    evt = GestureEvent(gesture=GestureType.DRAG_START, timestamp=timestamp, confidence=pose.stability_score, payload="DRAG_INITIATED")
                    self.last_emitted_event = evt
                    return evt

                # Check for DOUBLE_CLICK timing window
                time_since_last_click = timestamp - self.last_click_time
                if (
                    self.last_click_type == GestureType.LEFT_CLICK
                    and 0.04 <= time_since_last_click <= self.config.double_click_max_interval
                    and self.candidate_start_time > self.last_click_time
                ):
                    self.interaction_state = InteractionState.NEUTRAL
                    self.last_action_time = timestamp
                    self.last_click_time = timestamp
                    self.last_click_type = GestureType.DOUBLE_CLICK
                    self.candidate_gesture = GestureType.NO_GESTURE
                    evt = GestureEvent(gesture=GestureType.DOUBLE_CLICK, timestamp=timestamp, confidence=pose.stability_score)
                    self.last_emitted_event = evt
                    return evt

                # Otherwise emit SINGLE LEFT_CLICK if not emitted yet and not in cooldown
                if self.last_click_time != self.candidate_start_time:
                    if (timestamp - self.last_action_time) < self.config.gesture_cooldown_duration and self.candidate_start_time > self.last_action_time:
                        self.interaction_state = InteractionState.COOLDOWN
                        return None
                    self.interaction_state = InteractionState.NEUTRAL
                    self.last_action_time = timestamp
                    self.last_click_time = self.candidate_start_time
                    self.last_click_type = GestureType.LEFT_CLICK
                    evt = GestureEvent(gesture=GestureType.LEFT_CLICK, timestamp=timestamp, confidence=pose.stability_score)
                    self.last_emitted_event = evt
                    return evt

        self.interaction_state = InteractionState.NEUTRAL
        return None

        self.interaction_state = InteractionState.NEUTRAL
        return None

    def _classify_pose_to_gesture(self, pose: FingerPose) -> GestureType:
        """
        Maps finger pose configuration to candidate gesture intent.
        """
        states = pose.finger_states
        thumb = states.get(FingerName.THUMB, FingerState.OPEN)
        index = states.get(FingerName.INDEX, FingerState.OPEN)
        middle = states.get(FingerName.MIDDLE, FingerState.OPEN)
        ring = states.get(FingerName.RING, FingerState.OPEN)
        little = states.get(FingerName.LITTLE, FingerState.OPEN)

        # 3-finger pose -> ZOOM
        if (
            index in (FingerState.CLOSED, FingerState.TOUCHING)
            and middle in (FingerState.CLOSED, FingerState.TOUCHING)
            and ring in (FingerState.CLOSED, FingerState.TOUCHING)
            and little == FingerState.OPEN
        ):
            return GestureType.ZOOM_IN

        # 2-finger pose -> SCROLL
        if (
            index in (FingerState.CLOSED, FingerState.TOUCHING)
            and middle in (FingerState.CLOSED, FingerState.TOUCHING)
            and ring == FingerState.OPEN
            and little == FingerState.OPEN
            and thumb == FingerState.OPEN
        ):
            return GestureType.SCROLL_UP

        # Thumb + Middle pinch -> RIGHT_CLICK
        if (
            (thumb in (FingerState.TOUCHING, FingerState.CLOSED) and middle in (FingerState.TOUCHING, FingerState.CLOSED))
            and index == FingerState.OPEN
        ):
            return GestureType.RIGHT_CLICK

        # Thumb + Index pinch -> LEFT_CLICK / DRAG
        if (
            thumb in (FingerState.TOUCHING, FingerState.CLOSED)
            and index in (FingerState.TOUCHING, FingerState.CLOSED)
            and middle == FingerState.OPEN
        ):
            return GestureType.LEFT_CLICK

        return GestureType.NO_GESTURE

    def get_diagnostic_state(self) -> GestureDiagnosticState:
        """
        Returns full diagnostic state object for real-time telemetry rendering.
        """
        finger_dict: Dict[str, str] = {}
        if self.current_pose:
            for f, st in self.current_pose.finger_states.items():
                finger_dict[f.value] = st.value
        else:
            for f in FingerName:
                finger_dict[f.value] = FingerState.OPEN.value

        cooldown_rem = max(0.0, self.config.gesture_cooldown_duration - (self.candidate_start_time - self.last_action_time))

        confirmed_str = self.last_emitted_event.gesture.value if self.last_emitted_event else "None"

        return GestureDiagnosticState(
            finger_states=finger_dict,
            candidate_gesture=self.candidate_gesture.value,
            confirmed_gesture=confirmed_str,
            confidence=self.current_pose.stability_score if self.current_pose else 1.0,
            cooldown_remaining=round(cooldown_rem, 3),
            system_state=self.system_state.value,
            interaction_state=self.interaction_state.value
        )
