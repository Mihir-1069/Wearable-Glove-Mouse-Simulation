"""
Tkinter Visual Dashboard GUI Application.
Provides real-time interactive rendering of the virtual screen, cursor position,
uncertainty circle, targets, telemetry meters, scenario controls, gesture triggers, and haptic logs.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List
import numpy as np
import os
from PIL import Image

from wearable_mouse.config import Config
from wearable_mouse.simulation.imu_simulator import IMUSimulator
from wearable_mouse.simulation.scenarios import MotionScenario, ScenarioGenerator
from wearable_mouse.simulation.gesture_simulator import GestureSimulator
from wearable_mouse.sensors.imu_data import IMUSample
from wearable_mouse.processing.calibration import Calibrator, CalibrationState
from wearable_mouse.processing.filters import FilterPipeline, LowPassFilter, DeadZoneFilter, ExponentialSmoothingFilter
from wearable_mouse.processing.motion_detector import MotionDetector, MotionState
from wearable_mouse.processing.uncertainty import UncertaintyEstimator, UncertaintyState
from wearable_mouse.cursor.cursor_engine import CursorEngine, VirtualCursor
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode, AssistanceResult
from wearable_mouse.gestures.gesture_engine import GestureEngine, SystemState
from wearable_mouse.gestures.gesture_map import GestureType, GestureEvent
from wearable_mouse.interaction.mouse_controller import VirtualMouseController
from wearable_mouse.interaction.edge_feedback import EdgeDetector
from wearable_mouse.interaction.haptic_events import HapticManager, HapticEvent
from wearable_mouse.processing.orientation import OrientationEstimator
from wearable_mouse.processing.posture_reference import PostureReferenceTracker, PostureInfo
from wearable_mouse.telemetry_logging.data_logger import DataLogger, LogFrame


from wearable_mouse.sensors.sensor_models import AbstractSensor, SimulatedIMUSensor
from wearable_mouse.sensors.recorder_replayer import SensorDataRecorder


class DashboardApp:
    """
    Main Tkinter Real-Time Visualization Window.
    """

    def __init__(
        self,
        root: tk.Tk,
        config: Optional[Config] = None,
        sensor_source: Optional[AbstractSensor] = None,
        record_filepath: Optional[str] = None
    ):
        self.root = root
        self.config = config or Config()
        self.root.title("Wearable Glove Mouse — Real-Time Simulation Dashboard")
        self.root.geometry("1280x760")
        self.root.configure(bg="#1e1e2e")

        # Core Engines Initialization
        self.simulator = IMUSimulator(self.config)
        self.scenario_gen = ScenarioGenerator(self.simulator)
        self.gesture_sim = GestureSimulator(self.config)

        # Telemetry Sensor Source Setup
        base_source = sensor_source if sensor_source is not None else SimulatedIMUSensor(self.simulator)
        if record_filepath:
            self.sensor_source = SensorDataRecorder(base_source, record_filepath)
        else:
            self.sensor_source = base_source

        self.calibrator = Calibrator(self.config)
        self.filters = FilterPipeline([
            LowPassFilter(self.config.lowpass_cutoff, self.config.sample_rate),
            DeadZoneFilter(self.config.deadzone_threshold),
            ExponentialSmoothingFilter(self.config.exp_smoothing_alpha)
        ])
        from wearable_mouse.processing.orientation import OrientationEstimator
        from wearable_mouse.processing.posture_reference import PostureReferenceTracker
        self.orientation_est = OrientationEstimator(self.config)
        self.posture_tracker = PostureReferenceTracker(self.config)
        self.motion_detector = MotionDetector(self.config)
        self.uncertainty_est = UncertaintyEstimator(self.config)
        self.cursor_engine = CursorEngine(self.config)
        self.target_engine = TargetAssistanceEngine(self.config)
        self.gesture_engine = GestureEngine(self.config)
        from wearable_mouse.interaction.interaction_state_machine import InteractionStateMachine
        self.interaction_sm = InteractionStateMachine(self.config)
        self.mouse_ctrl = VirtualMouseController()
        self.edge_detector = EdgeDetector(self.config)
        self.haptic_manager = HapticManager()
        self.logger = DataLogger()

        # Dashboard State
        self.is_running = True
        self.step_counter = 0
        self.current_scenario = MotionScenario.TARGET_CLICK_SEQUENCE
        self.current_mode = AcquisitionMode.D
        self.last_gesture_str = "None"

        # Setup standard targets
        self.targets = [
            Target(id="t1", x=200, y=200, width=160, height=100, label="Target A"),
            Target(id="t2", x=850, y=250, width=180, height=110, label="Target B"),
            Target(id="t3", x=1450, y=200, width=160, height=100, label="Target C"),
            Target(id="t4", x=450, y=700, width=200, height=120, label="Target D"),
            Target(id="t5", x=1250, y=720, width=180, height=110, label="Target E"),
            # Close pair to test ambiguity:
            Target(id="t6_a", x=800, y=550, width=100, height=80, label="Pair X"),
            Target(id="t6_b", x=920, y=550, width=100, height=80, label="Pair Y"),
        ]
        self.target_engine.set_targets(self.targets)

        # Wire up haptic logger callback
        self.haptic_manager.add_listener(self._on_haptic_event)

        # Build UI layout
        self._build_ui()

        # Register keybindings for gesture simulation
        self.root.bind("<Key>", self._on_keypress)

        # Start simulation loop (10ms tick ~ 100Hz)
        self.root.after(10, self._simulation_loop)

    def _build_ui(self) -> None:
        # Main layout: Left Canvas Viewport, Right Telemetry Control Panel
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------------------
        # Left Panel: Virtual Screen Canvas
        # ---------------------------------------------------------------------
        canvas_frame = ttk.LabelFrame(main_frame, text=" Virtual Display Viewport (1920x1080) ", padding=5)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scaled canvas width:height = 16:9 (960x540 pixels display)
        self.canvas_w = 920
        self.canvas_h = 517
        self.scale_x = self.canvas_w / self.config.screen_width
        self.scale_y = self.canvas_h / self.config.screen_height

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_w,
            height=self.canvas_h,
            bg="#11111b",
            highlightthickness=2,
            highlightbackground="#45475a"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Canvas Control Strip
        ctrl_strip = ttk.Frame(canvas_frame)
        ctrl_strip.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl_strip, text="Scenario:").pack(side=tk.LEFT, padx=5)
        self.scenario_var = tk.StringVar(value=self.current_scenario.value)
        scenario_combo = ttk.Combobox(
            ctrl_strip,
            textvariable=self.scenario_var,
            values=[s.value for s in MotionScenario],
            state="readonly",
            width=25
        )
        scenario_combo.pack(side=tk.LEFT, padx=5)
        scenario_combo.bind("<<ComboboxSelected>>", self._on_scenario_changed)

        self.enable_filtering = True
        self.btn_toggle_filter = ttk.Button(ctrl_strip, text="Filter: ON", command=self._toggle_filter)
        self.btn_toggle_filter.pack(side=tk.LEFT, padx=5)

        self.btn_recalib = ttk.Button(ctrl_strip, text="Recalibrate", command=self._trigger_recalib)
        self.btn_recalib.pack(side=tk.LEFT, padx=5)

        # ---------------------------------------------------------------------
        # Right Panel: Controls & Telemetry
        # ---------------------------------------------------------------------
        right_panel = ttk.Frame(main_frame, width=320, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # 1. System State & Mode Box
        state_box = ttk.LabelFrame(right_panel, text=" System Status & Mode ", padding=10)
        state_box.pack(fill=tk.X, pady=5)

        self.lbl_system_state = tk.Label(state_box, text="ACTIVE", font=("Helvetica", 14, "bold"), fg="#a6e3a1", bg="#1e1e2e")
        self.lbl_system_state.pack(fill=tk.X, pady=2)

        ttk.Label(state_box, text="Assistance Mode:").pack(anchor=tk.W, pady=(5, 2))
        self.mode_var = tk.StringVar(value=self.current_mode.value)
        mode_combo = ttk.Combobox(
            state_box,
            textvariable=self.mode_var,
            values=[m.value for m in AcquisitionMode],
            state="readonly"
        )
        mode_combo.pack(fill=tk.X, pady=2)
        mode_combo.bind("<<ComboboxSelected>>", self._on_mode_changed)

        # 2. Telemetry Gauges
        telem_box = ttk.LabelFrame(right_panel, text=" Real-Time Telemetry ", padding=10)
        telem_box.pack(fill=tk.X, pady=5)

        self.lbl_hand_state = ttk.Label(telem_box, text="HAND STATE: AIR", font=("Consolas", 10, "bold"), foreground="#89b4fa")
        self.lbl_hand_state.pack(anchor=tk.W, pady=2)

        self.lbl_class = ttk.Label(telem_box, text="CLASSIFICATION: AIR_MOVEMENT", font=("Consolas", 9), foreground="#a6e3a1")
        self.lbl_class.pack(anchor=tk.W, pady=2)

        self.lbl_imu = ttk.Label(telem_box, text="IMU Gyro Mag: 0.000 rad/s", font=("Consolas", 9))
        self.lbl_imu.pack(anchor=tk.W, pady=2)

        self.lbl_ref = ttk.Label(telem_box, text="REFERENCE: R:0.0° P:0.0° Y:0.0°", font=("Consolas", 9))
        self.lbl_ref.pack(anchor=tk.W, pady=2)

        self.lbl_motion = ttk.Label(telem_box, text="MOVEMENT: STATIONARY", font=("Consolas", 9))
        self.lbl_motion.pack(anchor=tk.W, pady=2)

        self.lbl_cursor = ttk.Label(telem_box, text="CURSOR: (960, 540) @ 0 px/s", font=("Consolas", 9))
        self.lbl_cursor.pack(anchor=tk.W, pady=2)

        self.lbl_uncert = ttk.Label(telem_box, text="UNCERTAINTY: R=15px (100%)", font=("Consolas", 9))
        self.lbl_uncert.pack(anchor=tk.W, pady=2)

        self.lbl_friction = ttk.Label(telem_box, text="FRICTION GAIN: 1.00", font=("Consolas", 9), foreground="#a6e3a1")
        self.lbl_friction.pack(anchor=tk.W, pady=2)

        self.lbl_inter_state = ttk.Label(telem_box, text="INTERACTION STATE: AIR", font=("Consolas", 9, "bold"), foreground="#cba6f7")
        self.lbl_inter_state.pack(anchor=tk.W, pady=2)

        self.lbl_semantic_evt = ttk.Label(telem_box, text="Semantic Evt: None", font=("Consolas", 8), foreground="#f5e0dc")
        self.lbl_semantic_evt.pack(anchor=tk.W, pady=2)

        self.lbl_calib = ttk.Label(telem_box, text="Calib: UNCALIBRATED", font=("Consolas", 9))
        self.lbl_calib.pack(anchor=tk.W, pady=2)

        self.lbl_gesture = ttk.Label(telem_box, text="Last Gesture: None", font=("Consolas", 9, "bold"), foreground="#cba6f7")
        self.lbl_gesture.pack(anchor=tk.W, pady=2)

        self.lbl_gesture_diag = ttk.Label(telem_box, text="Gesture Diag: NEUTRAL (Cooldown: 0.0s)", font=("Consolas", 8), foreground="#a6adc8")
        self.lbl_gesture_diag.pack(anchor=tk.W, pady=2)

        # 3. Gesture Simulator Controls & Help
        help_box = ttk.LabelFrame(right_panel, text=" Keyboard Gesture Simulator ", padding=10)
        help_box.pack(fill=tk.X, pady=5)

        help_text = (
            "Keys:\n"
            "  [1] Left Click      [2] Right Click\n"
            "  [3] Double Click    [4] Scroll Up\n"
            "  [5] Drag Start/Stop [6] Zoom In\n"
            "  [P] Pause/Unpause System"
        )
        ttk.Label(help_box, text=help_text, justify=tk.LEFT, font=("Consolas", 9)).pack(anchor=tk.W)

        # 4. Haptic Feedback Event Log Feed
        haptic_box = ttk.LabelFrame(right_panel, text=" Haptic Event Log ", padding=5)
        haptic_box.pack(fill=tk.BOTH, expand=True, pady=5)

        self.haptic_listbox = tk.Listbox(haptic_box, height=8, bg="#181825", fg="#f5e0dc", font=("Consolas", 9), selectbackground="#45475a")
        self.haptic_listbox.pack(fill=tk.BOTH, expand=True)

    def _on_scenario_changed(self, event=None) -> None:
        val = self.scenario_var.get()
        for s in MotionScenario:
            if s.value == val:
                self.current_scenario = s
                self.step_counter = 0
                self.cursor_engine.set_position(960, 540)
                break

    def _on_mode_changed(self, event=None) -> None:
        val = self.mode_var.get()
        for m in AcquisitionMode:
            if m.value == val:
                self.current_mode = m
                break

    def _toggle_sim_play(self) -> None:
        self.is_running = not self.is_running
        self.btn_pause_sim.config(text="Resume Sim" if not self.is_running else "Pause Sim")

    def _toggle_filter(self) -> None:
        self.enable_filtering = not self.enable_filtering
        self.btn_toggle_filter.config(text="Filter: ON" if self.enable_filtering else "Filter: OFF")

    def _trigger_recalib(self) -> None:
        self.calibrator.start_calibration()

    def _on_keypress(self, event: tk.Event) -> None:
        char = event.char
        gesture = self.gesture_sim.trigger_key(char)
        if gesture:
            evt = self.gesture_engine.process_gesture(gesture, self.simulator.current_time)
            if evt:
                self.last_gesture_str = evt.gesture.value
                self.haptic_manager.trigger_click(self.simulator.current_time)

    def _on_haptic_event(self, evt: HapticEvent) -> None:
        log_str = f"[{evt.timestamp:.2f}s] {evt.event_name} -> {evt.pattern.value}"
        self.haptic_listbox.insert(0, log_str)
        if self.haptic_listbox.size() > 50:
            self.haptic_listbox.delete(50, tk.END)

    def _simulation_loop(self) -> None:
        if self.is_running:
            self.step_counter += 1

            # 1. Fetch Sensor Telemetry Frame
            if isinstance(self.sensor_source, SimulatedIMUSensor):
                sample = self.scenario_gen.run_scenario_step(self.current_scenario, self.step_counter)
            else:
                data = self.sensor_source.read_sample()
                sample = data.imu

            # 2. Calibration
            calib_state = self.calibrator.update(sample)
            calib_sample = self.calibrator.apply(sample)

            # 3. Filtering
            filt_gyro = self.filters.process(calib_sample.gyro) if self.enable_filtering else calib_sample.gyro

            # 4. Orientation & Posture Reference Tracking
            current_orient, _ = self.orientation_est.update(calib_sample)
            posture_info = self.posture_tracker.update(calib_sample, current_orient, filt_gyro)

            # 5. Motion Detection
            is_stat = (calib_state == CalibrationState.CALIBRATING or calib_state == CalibrationState.UNCALIBRATED)
            motion_state = self.motion_detector.update(filt_gyro, is_stationary_calib=is_stat)

            # 6. Uncertainty Model
            uncertainty = self.uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)

            # 7. Cursor Mapping with Posture Continuity
            is_paused = (self.gesture_engine.system_state == SystemState.PAUSED)
            cursor = self.cursor_engine.update(filt_gyro, is_paused=is_paused, posture_info=posture_info)

            # 7. Target Acquisition Assistance
            assist = self.target_engine.process(cursor, uncertainty, mode=self.current_mode)

            # Apply soft snapping if active
            if assist.snapped_position != (cursor.x, cursor.y):
                self.cursor_engine.cursor.x, self.cursor_engine.cursor.y = assist.snapped_position

            # 8. Edge Detection
            new_edges = self.edge_detector.update(cursor)
            for edge in new_edges:
                self.haptic_manager.trigger_edge_bump(edge.value, sample.timestamp)

            # 9. Unified Interaction State Machine Update
            self.interaction_sm.update(
                timestamp=sample.timestamp,
                calib_state=calib_state,
                posture_info=posture_info,
                gesture_event=None,
                assist_result=assist,
                uncertainty=uncertainty,
                cursor=cursor
            )

            # 10. Update Telemetry UI
            self._update_telemetry_ui(calib_state, motion_state, cursor, uncertainty, assist, posture_info)

            # 10. Draw Canvas Viewport
            self._render_canvas(cursor, uncertainty, assist)

        # Queue next frame loop (~100Hz)
        self.root.after(10, self._simulation_loop)

    def _update_telemetry_ui(
        self,
        calib_state: CalibrationState,
        motion: MotionState,
        cursor: VirtualCursor,
        uncertainty: UncertaintyState,
        assist: AssistanceResult,
        posture_info: Optional[PostureInfo] = None
    ) -> None:
        # Pause state badge
        if self.gesture_engine.system_state == SystemState.PAUSED:
            self.lbl_system_state.config(text="PAUSED", fg="#f38ba8")
        else:
            self.lbl_system_state.config(text="ACTIVE", fg="#a6e3a1")

        if posture_info:
            state_str = posture_info.state.name
            state_fg = "#89b4fa" if state_str == "AIR" else ("#f9e2af" if state_str == "REST" else "#f38ba8")
            self.lbl_hand_state.config(text=f"HAND STATE: {state_str}", foreground=state_fg)

            class_str = posture_info.classification.name
            class_fg = "#a6e3a1" if "INTENTIONAL" in class_str or "AIR" in class_str else "#89dceb"
            self.lbl_class.config(text=f"CLASSIFICATION: {class_str}", foreground=class_fg)

            self.lbl_imu.config(text=f"IMU Gyro Mag: {posture_info.filtered_movement_mag:.3f} rad/s")
            ref_deg = np.degrees(posture_info.reference_orientation)
            self.lbl_ref.config(text=f"REFERENCE: R:{ref_deg[0]:.1f}° P:{ref_deg[1]:.1f}° Y:{ref_deg[2]:.1f}°")
            self.lbl_motion.config(text=f"MOVEMENT: {motion.movement_type.name} ({posture_info.filtered_movement_mag:.2f} r/s)")
        else:
            self.lbl_hand_state.config(text="HAND STATE: AIR")
            self.lbl_class.config(text="CLASSIFICATION: AIR_MOVEMENT")
            self.lbl_imu.config(text="IMU Gyro Mag: 0.000 rad/s")
            self.lbl_ref.config(text="REFERENCE: R:0.0° P:0.0° Y:0.0°")
            self.lbl_motion.config(text=f"MOVEMENT: {motion.movement_type.name}")

        self.lbl_cursor.config(text=f"CURSOR: ({int(cursor.x)}, {int(cursor.y)}) @ {int(cursor.speed)} px/s")
        self.lbl_uncert.config(text=f"UNCERTAINTY: R={int(uncertainty.uncertainty_radius)}px ({int(uncertainty.confidence_score * 100)}%)")

        friction = getattr(assist, 'friction_gain', 1.0)
        self.lbl_friction.config(text=f"FRICTION GAIN: {friction:.2f}", foreground="#f9e2af" if friction < 1.0 else "#a6e3a1")

        inter_state = self.interaction_sm.current_state.name
        self.lbl_inter_state.config(text=f"INTERACTION STATE: {inter_state}")

        last_evt = self.interaction_sm.transition_logs[-1].event if self.interaction_sm.transition_logs else "None"
        self.lbl_semantic_evt.config(text=f"Semantic Evt: {last_evt}")

        self.lbl_calib.config(text=f"Calib: {calib_state.name}")
        self.lbl_gesture.config(text=f"Last Gesture: {self.last_gesture_str}")

        diag = self.gesture_engine.get_diagnostic_state()
        self.lbl_gesture_diag.config(text=f"Diag: {diag.interaction_state} | Cand: {diag.candidate_gesture}")

    def _render_canvas(
        self,
        cursor: VirtualCursor,
        uncertainty: UncertaintyState,
        assist: AssistanceResult
    ) -> None:
        self.canvas.delete("all")

        # Render Targets
        for target in self.targets:
            tx = target.x * self.scale_x
            ty = target.y * self.scale_y
            tw = target.width * self.scale_x
            th = target.height * self.scale_y

            is_selected = (assist.selected_target and assist.selected_target.id == target.id)
            is_ambig = (assist.is_ambiguous and target in assist.ambiguous_targets)

            if is_selected:
                fill_color = "#313244"
                outline_color = "#a6e3a1"
                width = 3
            elif is_ambig:
                fill_color = "#45475a"
                outline_color = "#f38ba8"  # Red warning for ambiguous target
                width = 3
            else:
                fill_color = "#1e1e2e"
                outline_color = "#89b4fa"
                width = 1

            self.canvas.create_rectangle(tx, ty, tx + tw, ty + th, fill=fill_color, outline=outline_color, width=width)
            self.canvas.create_text(tx + tw / 2, ty + th / 2, text=target.label, fill="#cdd6f4", font=("Consolas", 10, "bold"))

        # Render Uncertainty Circle
        cx = cursor.x * self.scale_x
        cy = cursor.y * self.scale_y
        r_uncert = uncertainty.uncertainty_radius * self.scale_x

        circle_color = "#f38ba8" if assist.is_ambiguous else "#cba6f7"
        self.canvas.create_oval(
            cx - r_uncert, cy - r_uncert,
            cx + r_uncert, cy + r_uncert,
            outline=circle_color,
            width=2,
            dash=(4, 4)
        )

        # Render Virtual Cursor Dot
        r_cursor = 5
        self.canvas.create_oval(
            cx - r_cursor, cy - r_cursor,
            cx + r_cursor, cy + r_cursor,
            fill="#f9e2af",
            outline="#f5e0dc"
        )

        # Ambiguity Warning Overlay Banner
        if assist.is_ambiguous:
            self.canvas.create_rectangle(
                self.canvas_w / 2 - 140, 10,
                self.canvas_w / 2 + 140, 35,
                fill="#f38ba8", outline="#11111b"
            )
            self.canvas.create_text(
                self.canvas_w / 2, 22,
                text="⚠️ AMBIGUOUS TARGET DETECTED",
                fill="#11111b",
                font=("Helvetica", 10, "bold")
            )
def _run_headless_scenarios(app: DashboardApp, steps_per_scenario: int = 200, capture_interval: int = 20, output_dir: str = "headless_scenarios") -> None:
    """Run each predefined MotionScenario headlessly and capture canvas screenshots.

    Args:
        app: Instance of DashboardApp.
        steps_per_scenario: Number of simulation steps per scenario.
        capture_interval: Interval of steps at which to capture a screenshot.
        output_dir: Directory to save screenshot PNG files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for scenario in MotionScenario:
        app.current_scenario = scenario
        app.step_counter = 0
        # Reset cursor to screen centre
        app.cursor_engine.set_position(app.config.screen_width / 2, app.config.screen_height / 2)
        for step in range(steps_per_scenario):
            sample = app.scenario_gen.run_scenario_step(scenario, step + 1)
            calib_state = app.calibrator.update(sample)
            calib_sample = app.calibrator.apply(sample)
            filt_gyro = app.filters.process(calib_sample.gyro) if app.enable_filtering else calib_sample.gyro
            is_stat = (calib_state == CalibrationState.CALIBRATING or calib_state == CalibrationState.UNCALIBRATED)
            motion_state = app.motion_detector.update(filt_gyro, is_stationary_calib=is_stat)
            uncertainty = app.uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)
            is_paused = (app.gesture_engine.system_state == SystemState.PAUSED)
            cursor = app.cursor_engine.update(filt_gyro, is_paused=is_paused)
            assist = app.target_engine.process(cursor, uncertainty, mode=app.current_mode)
            if assist.snapped_position != (cursor.x, cursor.y):
                app.cursor_engine.cursor.x, app.cursor_engine.cursor.y = assist.snapped_position
            new_edges = app.edge_detector.update(cursor)
            for edge in new_edges:
                app.haptic_manager.trigger_edge_bump(edge.value, sample.timestamp)
            app._update_telemetry_ui(calib_state, motion_state, cursor, uncertainty, assist)
            app._render_canvas(cursor, uncertainty, assist)
            if (step + 1) % capture_interval == 0:
                ps_path = os.path.join(output_dir, f"{scenario.value}_step{step+1}.ps")
                png_path = os.path.join(output_dir, f"{scenario.value}_step{step+1}.png")
                app.canvas.postscript(file=ps_path)
                Image.open(ps_path).save(png_path)
                os.remove(ps_path)
        # Final capture for the scenario
        ps_path = os.path.join(output_dir, f"{scenario.value}_final.ps")
        png_path = os.path.join(output_dir, f"{scenario.value}_final.png")
        app.canvas.postscript(file=ps_path)
        Image.open(ps_path).save(png_path)
        os.remove(ps_path)
