"""
Wearable Glove Mouse Master End-to-End Pipeline Manager.
Orchestrates all 13 software stages in strict deterministic order per frame tick.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

from wearable_mouse.config import Config
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.processing.calibration import Calibrator, CalibrationState
from wearable_mouse.processing.filters import FilterPipeline, LowPassFilter, DeadZoneFilter, ExponentialSmoothingFilter
from wearable_mouse.processing.orientation import OrientationEstimator
from wearable_mouse.processing.motion_detector import MotionDetector, MotionState
from wearable_mouse.processing.uncertainty import UncertaintyEstimator, UncertaintyState
from wearable_mouse.processing.posture_reference import PostureReferenceTracker, PostureInfo
from wearable_mouse.cursor.cursor_engine import CursorEngine, VirtualCursor
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode, AssistanceResult
from wearable_mouse.gestures.gesture_engine import GestureEngine, SystemState
from wearable_mouse.gestures.gesture_map import GestureEvent, GestureType
from wearable_mouse.interaction.interaction_state import UnifiedState
from wearable_mouse.interaction.interaction_state_machine import InteractionStateMachine
from wearable_mouse.interaction.output_adapter import SemanticEventType, SemanticEvent, OutputAdapter
from wearable_mouse.context.context_manager import ContextProfileManager
from wearable_mouse.context.context_models import Context


@dataclass
class PipelineFrameResult:
    """
    Per-frame diagnostic container holding outputs from all software pipeline stages.
    """
    timestamp: float
    calib_state: CalibrationState
    filtered_gyro: np.ndarray
    orientation: np.ndarray
    motion_state: MotionState
    uncertainty: UncertaintyState
    posture_info: PostureInfo
    cursor: VirtualCursor
    assist_result: AssistanceResult
    gesture_event: Optional[GestureEvent]
    interaction_state: UnifiedState
    semantic_events: List[SemanticEvent]
    active_context: Optional[Context] = None


class WearableMousePipeline:
    """
    Master 13-stage deterministic end-to-end wearable mouse interaction engine.
    """

    def __init__(self, config: Optional[Config] = None, enable_os_mouse: bool = False):
        self.config = config or Config()

        # 1-3. Sensor, Calibrator, Filters
        self.calibrator = Calibrator(self.config)
        self.filters = FilterPipeline([
            LowPassFilter(self.config.lowpass_cutoff, self.config.sample_rate),
            DeadZoneFilter(self.config.deadzone_threshold),
            ExponentialSmoothingFilter(self.config.exp_smoothing_alpha)
        ])

        # 4-6. Orientation, Motion, Uncertainty, Posture
        self.orientation_est = OrientationEstimator(self.config)
        self.motion_detector = MotionDetector(self.config)
        self.uncertainty_est = UncertaintyEstimator(self.config)
        self.posture_tracker = PostureReferenceTracker(self.config)

        # 7-8. Cursor Engine & Target Assistance (Dynamic Friction)
        self.cursor_engine = CursorEngine(self.config)
        self.target_engine = TargetAssistanceEngine(self.config)

        # 9-10. Gesture Recognition & Temporal Arbitration Engine
        self.gesture_engine = GestureEngine(self.config)

        # 11. Unified Interaction State Machine
        self.interaction_sm = InteractionStateMachine(self.config)

        # 12-13. Semantic Events & Output Adapter
        self.output_adapter = OutputAdapter(enable_os_mouse=enable_os_mouse)

        # Context Profile Manager
        self.context_manager = ContextProfileManager()

        # Internal Pipeline State
        self.current_mode = AcquisitionMode.D
        self.enable_filtering = True
        self.previous_cursor_pos: Tuple[float, float] = (self.cursor_engine.cursor.x, self.cursor_engine.cursor.y)
        self.previous_interaction_state: UnifiedState = UnifiedState.AIR

    def set_targets(self, targets: List[Target]) -> None:
        """Register interactive UI targets with target assistance engine."""
        self.target_engine.set_targets(targets)

    def set_mode(self, mode: AcquisitionMode) -> None:
        """Set target acquisition assistance mode (A, B, C, D, or E)."""
        self.current_mode = mode

    def reset(self, initial_x: float = 960.0, initial_y: float = 540.0) -> None:
        """Reset pipeline state and cursor coordinates."""
        self.calibrator.start_calibration()
        self.filters.reset()
        self.orientation_est.reset()
        self.posture_tracker.reset()
        self.cursor_engine.set_position(initial_x, initial_y)
        self.interaction_sm.reset()
        self.output_adapter.clear_history()
        self.previous_cursor_pos = (initial_x, initial_y)
        self.previous_interaction_state = UnifiedState.AIR

    def process_frame(
        self,
        imu_sample: IMUSample,
        gesture_input: Optional[GestureType] = None
    ) -> PipelineFrameResult:
        """
        Execute full 13-stage interaction pipeline for a single sensor frame.
        """
        timestamp = imu_sample.timestamp

        # ---------------------------------------------------------------------
        # Stage 1-2: Calibration
        # ---------------------------------------------------------------------
        calib_state = self.calibrator.update(imu_sample)
        calib_sample = self.calibrator.apply(imu_sample)

        # ---------------------------------------------------------------------
        # Stage 3: Filtering
        # ---------------------------------------------------------------------
        filt_gyro = self.filters.process(calib_sample.gyro) if self.enable_filtering else calib_sample.gyro

        # ---------------------------------------------------------------------
        # Stage 4: Motion & Orientation Estimation
        # ---------------------------------------------------------------------
        orientation, _ = self.orientation_est.update(calib_sample)
        is_stat = (calib_state == CalibrationState.CALIBRATING or calib_state == CalibrationState.UNCALIBRATED)
        motion_state = self.motion_detector.update(filt_gyro, is_stationary_calib=is_stat)

        # ---------------------------------------------------------------------
        # Stage 5: Uncertainty Model
        # ---------------------------------------------------------------------
        uncertainty = self.uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)

        # ---------------------------------------------------------------------
        # Stage 6: Air / Rest Posture Reference Tracking
        # ---------------------------------------------------------------------
        posture_info = self.posture_tracker.update(calib_sample, orientation, filt_gyro)

        # ---------------------------------------------------------------------
        # Stage 7: Cursor Engine (Decoupled from gesture inputs)
        # ---------------------------------------------------------------------
        is_paused = (self.interaction_sm.current_state == UnifiedState.PAUSED)
        cursor = self.cursor_engine.update(filt_gyro, is_paused=is_paused, posture_info=posture_info)

        # Calculate Cursor Motion Delta
        prev_x, prev_y = self.previous_cursor_pos
        dx = cursor.x - prev_x
        dy = cursor.y - prev_y
        self.previous_cursor_pos = (cursor.x, cursor.y)

        # ---------------------------------------------------------------------
        # Stage 8: Dynamic Friction & Target Assistance
        # ---------------------------------------------------------------------
        assist_result = self.target_engine.process(cursor, uncertainty, mode=self.current_mode)
        if assist_result.snapped_position != (cursor.x, cursor.y):
            self.cursor_engine.cursor.x, self.cursor_engine.cursor.y = assist_result.snapped_position
            cursor = self.cursor_engine.cursor

        # ---------------------------------------------------------------------
        # Stage 9-10: Gesture Recognition & Priority Arbitration
        # ---------------------------------------------------------------------
        gesture_event: Optional[GestureEvent] = None
        if gesture_input is not None:
            gesture_event = self.gesture_engine.process_gesture(gesture_input, timestamp)

        # ---------------------------------------------------------------------
        # Stage 11: Unified Interaction State Machine
        # ---------------------------------------------------------------------
        current_state = self.interaction_sm.update(
            timestamp=timestamp,
            calib_state=calib_state,
            posture_info=posture_info,
            gesture_event=gesture_event,
            assist_result=assist_result,
            uncertainty=uncertainty,
            cursor=cursor
        )

        # ---------------------------------------------------------------------
        # Context Resolution
        # ---------------------------------------------------------------------
        active_context = self.context_manager.resolve_context(interaction_mode=current_state.value)

        # ---------------------------------------------------------------------
        # Stage 12: Semantic Interaction Event Formulation
        # ---------------------------------------------------------------------
        semantic_events: List[SemanticEvent] = []

        # State transition event check
        if current_state != self.previous_interaction_state:
            semantic_events.append(
                SemanticEvent(
                    event_type=SemanticEventType.STATE_CHANGE,
                    timestamp=timestamp,
                    cursor_x=cursor.x,
                    cursor_y=cursor.y,
                    delta_x=dx,
                    delta_y=dy,
                    state=current_state,
                    payload=f"{self.previous_interaction_state.value} -> {current_state.value}"
                )
            )

        # Cursor movement semantic event (if unpaused and non-zero motion)
        if current_state != UnifiedState.PAUSED and (abs(dx) > 1e-4 or abs(dy) > 1e-4):
            semantic_events.append(
                SemanticEvent(
                    event_type=SemanticEventType.CURSOR_MOVE,
                    timestamp=timestamp,
                    cursor_x=cursor.x,
                    cursor_y=cursor.y,
                    delta_x=dx,
                    delta_y=dy,
                    state=current_state
                )
            )

        # Specific action semantic events based on UnifiedState triggers
        if gesture_event and current_state != UnifiedState.PAUSED:
            gt = gesture_event.gesture
            if gt == GestureType.LEFT_CLICK and current_state == UnifiedState.LEFT_CLICK:
                semantic_events.append(
                    SemanticEvent(SemanticEventType.LEFT_CLICK, timestamp, cursor.x, cursor.y, state=current_state)
                )
            elif gt == GestureType.RIGHT_CLICK and current_state == UnifiedState.RIGHT_CLICK:
                semantic_events.append(
                    SemanticEvent(SemanticEventType.RIGHT_CLICK, timestamp, cursor.x, cursor.y, state=current_state)
                )
            elif gt == GestureType.DOUBLE_CLICK and current_state == UnifiedState.DOUBLE_CLICK:
                semantic_events.append(
                    SemanticEvent(SemanticEventType.DOUBLE_CLICK, timestamp, cursor.x, cursor.y, state=current_state)
                )
            elif gt in (GestureType.DRAG_START, GestureType.DRAG_TOGGLE) and current_state == UnifiedState.DRAG:
                semantic_events.append(
                    SemanticEvent(SemanticEventType.DRAG_START, timestamp, cursor.x, cursor.y, state=current_state)
                )
            elif gt == GestureType.DRAG_END and self.previous_interaction_state == UnifiedState.DRAG:
                semantic_events.append(
                    SemanticEvent(SemanticEventType.DRAG_END, timestamp, cursor.x, cursor.y, state=current_state)
                )
            elif gt in (GestureType.SCROLL_UP, GestureType.SCROLL_DOWN) and current_state == UnifiedState.SCROLL:
                scroll_delta = (1.0 if gt == GestureType.SCROLL_UP else -1.0) * active_context.sensitivity.scroll_sensitivity
                semantic_events.append(
                    SemanticEvent(SemanticEventType.SCROLL, timestamp, cursor.x, cursor.y, delta_y=scroll_delta, state=current_state)
                )
            elif gt in (GestureType.ZOOM_IN, GestureType.ZOOM_OUT) and current_state == UnifiedState.ZOOM:
                zoom_delta = 1.0 if gt == GestureType.ZOOM_IN else -1.0
                semantic_events.append(
                    SemanticEvent(SemanticEventType.ZOOM, timestamp, cursor.x, cursor.y, delta_y=zoom_delta, state=current_state)
                )

        if gesture_event and gesture_event.gesture == GestureType.PAUSE_TOGGLE:
            semantic_events.append(
                SemanticEvent(SemanticEventType.PAUSE_TOGGLE, timestamp, cursor.x, cursor.y, state=current_state)
            )

        # Update previous interaction state tracker
        self.previous_interaction_state = current_state

        # ---------------------------------------------------------------------
        # Stage 13: Output Adapter Dispatch
        # ---------------------------------------------------------------------
        for sem_evt in semantic_events:
            self.output_adapter.dispatch(sem_evt)

        return PipelineFrameResult(
            timestamp=timestamp,
            calib_state=calib_state,
            filtered_gyro=filt_gyro,
            orientation=orientation,
            motion_state=motion_state,
            uncertainty=uncertainty,
            posture_info=posture_info,
            cursor=cursor,
            assist_result=assist_result,
            gesture_event=gesture_event,
            interaction_state=current_state,
            semantic_events=semantic_events,
            active_context=active_context
        )
