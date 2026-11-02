"""
Human Test Mode Interactive GUI Application (python app.py --human-test).
Allows a human operator to control the virtual cursor via WASD / Arrow keys,
test control conditions interactively, and complete subjective feedback rating surveys (1-5 scales).
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, List
import numpy as np

from wearable_mouse.config import Config
from wearable_mouse.simulation.imu_simulator import IMUSimulator
from wearable_mouse.simulation.keyboard_imu_adapter import KeyboardIMUAdapter
from wearable_mouse.processing.calibration import Calibrator, CalibrationState
from wearable_mouse.processing.filters import FilterPipeline, LowPassFilter, DeadZoneFilter, ExponentialSmoothingFilter
from wearable_mouse.processing.motion_detector import MotionDetector, MotionState
from wearable_mouse.processing.uncertainty import UncertaintyEstimator, UncertaintyState
from wearable_mouse.cursor.cursor_engine import CursorEngine, VirtualCursor
from wearable_mouse.cursor.target_assistance import TargetAssistanceEngine, Target, AcquisitionMode, AssistanceResult
from wearable_mouse.cursor.dynamic_friction import DynamicSensitivitySuppression, FrictionState
from wearable_mouse.interaction.human_test_manager import HumanTestManager, ConditionInfo, TargetLayout
from wearable_mouse.interaction.edge_feedback import EdgeDetector
from wearable_mouse.interaction.haptic_events import HapticManager, HapticEvent


class RatingSurveyModal(tk.Toplevel):
    """
    Modal survey dialog prompting user for subjective ratings (1-5 scales) at end of trial block.
    """

    def __init__(self, parent: tk.Tk, block_num: int, condition_label: str):
        super().__init__(parent)
        self.title(f"Subjective Feedback Survey — Block {block_num}")
        self.geometry("450x420")
        self.configure(bg="#1e1e2e")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.ratings = {
            "difficulty": 3,
            "stability": 3,
            "confidence": 3,
            "fatigue": 1
        }
        self.submitted = False

        ttk.Label(
            self,
            text=f"Block {block_num} Complete ({condition_label})",
            font=("Helvetica", 12, "bold"),
            foreground="#a6e3a1"
        ).pack(pady=(15, 10))

        ttk.Label(
            self,
            text="Please rate your experience during this block (1 to 5):",
            font=("Helvetica", 10)
        ).pack(pady=(0, 15))

        # 1. Difficulty
        self.diff_var = self._create_rating_row("1. Perceived Control Difficulty (1: Easy -> 5: Hard):", 3)
        # 2. Stability
        self.stab_var = self._create_rating_row("2. Cursor Stability (1: Jittery -> 5: Extremely Stable):", 3)
        # 3. Confidence
        self.conf_var = self._create_rating_row("3. Confidence in Clicking (1: Low -> 5: High):", 3)
        # 4. Fatigue
        self.fatig_var = self._create_rating_row("4. Perceived Effort / Fatigue (1: Low -> 5: High):", 1)

        btn_submit = ttk.Button(self, text="Submit Ratings & Continue", command=self._on_submit)
        btn_submit.pack(pady=20)

    def _create_rating_row(self, label_text: str, default_val: int) -> tk.IntVar:
        frame = ttk.Frame(self, padding=5)
        frame.pack(fill=tk.X, padx=15, pady=5)
        ttk.Label(frame, text=label_text, font=("Consolas", 9, "bold")).pack(anchor=tk.W)

        var = tk.IntVar(value=default_val)
        sub_frame = ttk.Frame(frame)
        sub_frame.pack(anchor=tk.W, pady=2)
        for i in range(1, 6):
            rb = ttk.Radiobutton(sub_frame, text=str(i), value=i, variable=var)
            rb.pack(side=tk.LEFT, padx=8)
        return var

    def _on_submit(self) -> None:
        self.ratings = {
            "difficulty": self.diff_var.get(),
            "stability": self.stab_var.get(),
            "confidence": self.conf_var.get(),
            "fatigue": self.fatig_var.get()
        }
        self.submitted = True
        self.destroy()


class HumanTestGUI:
    """
    Main Interactive GUI Application for python app.py --human-test.
    """

    def __init__(self, root: tk.Tk, config: Optional[Config] = None):
        self.root = root
        self.config = config or Config()
        self.root.title("Wearable Glove Mouse — Interactive Human Test Mode")
        self.root.geometry("1280x780")
        self.root.configure(bg="#1e1e2e")

        # Pipeline Engines
        self.simulator = IMUSimulator(self.config)
        self.key_adapter = KeyboardIMUAdapter(self.simulator, self.config)
        self.calibrator = Calibrator(self.config)
        self.filters = FilterPipeline([
            LowPassFilter(self.config.lowpass_cutoff, self.config.sample_rate),
            DeadZoneFilter(self.config.deadzone_threshold),
            ExponentialSmoothingFilter(self.config.exp_smoothing_alpha)
        ])
        self.motion_detector = MotionDetector(self.config)
        self.uncertainty_est = UncertaintyEstimator(self.config)
        self.cursor_engine = CursorEngine(self.config)
        self.target_engine = TargetAssistanceEngine(self.config)
        self.friction_engine = DynamicSensitivitySuppression(self.config)
        self.test_manager = HumanTestManager(self.config, is_blinded=True)
        self.edge_detector = EdgeDetector(self.config)
        self.haptic_manager = HapticManager()

        # Telemetry State
        self.last_outcome_str = "None"
        self.haptic_manager.add_listener(self._on_haptic_event)

        # Build UI layout
        self._build_ui()

        # Bind Keypresses for WASD / Arrow Movement & Space / 1 Click
        self.root.bind("<KeyPress>", self._on_key_down)
        self.root.bind("<KeyRelease>", self._on_key_up)

        # Reset trial timer & cursor
        self.cursor_engine.set_position(960, 540)
        self.test_manager.reset_trial_metrics((960, 540), self.simulator.current_time)

        # Start 100Hz simulation tick loop
        self.root.after(10, self._simulation_loop)

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ---------------------------------------------------------------------
        # Left Panel: Virtual Viewport Canvas
        # ---------------------------------------------------------------------
        canvas_frame = ttk.LabelFrame(main_frame, text=" Interactive Virtual Screen (1920x1080) ", padding=5)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

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

        # Control Bar
        ctrl_bar = ttk.Frame(canvas_frame)
        ctrl_bar.pack(fill=tk.X, padx=5, pady=5)

        self.btn_blinded = ttk.Checkbutton(
            ctrl_bar, text="Blinded Mode (Hide Condition Names)", command=self._toggle_blinded
        )
        self.btn_blinded.pack(side=tk.LEFT, padx=5)
        self.btn_blinded.state(['selected'])

        ttk.Button(ctrl_bar, text="Trigger Click (Space)", command=self._trigger_click).pack(side=tk.RIGHT, padx=5)

        # ---------------------------------------------------------------------
        # Right Panel: Telemetry & Controls
        # ---------------------------------------------------------------------
        right_panel = ttk.Frame(main_frame, width=320, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)

        # 1. Trial Progress Box
        prog_box = ttk.LabelFrame(right_panel, text=" Test Progress ", padding=10)
        prog_box.pack(fill=tk.X, pady=5)

        self.lbl_cond = tk.Label(prog_box, text="Condition 1", font=("Helvetica", 14, "bold"), fg="#89b4fa", bg="#1e1e2e")
        self.lbl_cond.pack(fill=tk.X, pady=2)

        self.lbl_block_info = ttk.Label(prog_box, text="Block 1 — Trial 1 / 5", font=("Consolas", 10, "bold"))
        self.lbl_block_info.pack(anchor=tk.W, pady=2)

        self.lbl_layout_info = ttk.Label(prog_box, text="Layout: ISOLATED (120px)", font=("Consolas", 9))
        self.lbl_layout_info.pack(anchor=tk.W, pady=2)

        # 2. Real-Time Telemetry Box
        telem_box = ttk.LabelFrame(right_panel, text=" Live Telemetry ", padding=10)
        telem_box.pack(fill=tk.X, pady=5)

        self.lbl_speed = ttk.Label(telem_box, text="Cursor Speed: 0 px/s", font=("Consolas", 10))
        self.lbl_speed.pack(anchor=tk.W, pady=2)

        self.lbl_uncert = ttk.Label(telem_box, text="Uncertainty R: 15 px", font=("Consolas", 10))
        self.lbl_uncert.pack(anchor=tk.W, pady=2)

        self.lbl_friction = ttk.Label(telem_box, text="Friction Factor: 0.00", font=("Consolas", 10))
        self.lbl_friction.pack(anchor=tk.W, pady=2)

        self.lbl_outcome = ttk.Label(telem_box, text="Last Outcome: None", font=("Consolas", 10, "bold"), foreground="#f9e2af")
        self.lbl_outcome.pack(anchor=tk.W, pady=2)

        # 3. Legend / Keyboard Help Box
        help_box = ttk.LabelFrame(right_panel, text=" Operator Controls ", padding=10)
        help_box.pack(fill=tk.X, pady=5)

        help_text = (
            "Movement Keys:\n"
            "  [W] / [Up]    -> Pitch Up (Move Up)\n"
            "  [S] / [Down]  -> Pitch Down (Move Down)\n"
            "  [A] / [Left]  -> Yaw Left (Move Left)\n"
            "  [D] / [Right] -> Yaw Right (Move Right)\n\n"
            "Action Keys:\n"
            "  [Space] / [1] -> Trigger Target Click\n"
            "  [P]          -> Pause / Resume Control"
        )
        ttk.Label(help_box, text=help_text, justify=tk.LEFT, font=("Consolas", 9)).pack(anchor=tk.W)

        # 4. Haptic Log Box
        haptic_box = ttk.LabelFrame(right_panel, text=" Edge & Click Haptics Feed ", padding=5)
        haptic_box.pack(fill=tk.BOTH, expand=True, pady=5)

        self.haptic_listbox = tk.Listbox(haptic_box, height=6, bg="#181825", fg="#f5e0dc", font=("Consolas", 9))
        self.haptic_listbox.pack(fill=tk.BOTH, expand=True)

    def _toggle_blinded(self) -> None:
        self.test_manager.is_blinded = not self.test_manager.is_blinded

    def _on_key_down(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in ('w', 'a', 's', 'd', 'up', 'down', 'left', 'right'):
            self.key_adapter.key_down(key)
        elif key in ('space', '1'):
            self._trigger_click()

    def _on_key_up(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key in ('w', 'a', 's', 'd', 'up', 'down', 'left', 'right'):
            self.key_adapter.key_up(key)

    def _trigger_click(self) -> None:
        cursor_pos = (self.cursor_engine.cursor.x, self.cursor_engine.cursor.y)
        cond = self.test_manager.current_condition

        # Evaluate target click assistance
        uncert = self.uncertainty_est.update(np.zeros(3), np.zeros(3), self.motion_detector.update(np.zeros(3)))
        assist = self.target_engine.process(self.cursor_engine.cursor, uncert, mode=cond.acq_mode)

        outcome, is_block_done = self.test_manager.evaluate_click(
            cursor_pos=cursor_pos,
            timestamp=self.simulator.current_time,
            selected_target=assist.selected_target,
            uncertainty_radius=uncert.uncertainty_radius,
            noise_level=self.config.gyro_noise_std,
            tremor_level=self.config.tremor_amplitude
        )

        self.last_outcome_str = outcome
        self.lbl_outcome.config(text=f"Last Outcome: {outcome}")

        # Haptic feedback
        if outcome == "SUCCESS":
            self.haptic_manager.trigger_click(self.simulator.current_time)
        else:
            self.haptic_manager.trigger("MISSED_CLICK", HapticManager().trigger_click().pattern, intensity=0.3)

        if is_block_done:
            self._show_survey_modal()
        else:
            # Advance to next target and reset trial timer
            self.cursor_engine.set_position(960, 540)
            self.test_manager.reset_trial_metrics((960, 540), self.simulator.current_time)

    def _show_survey_modal(self) -> None:
        self.key_adapter.clear_keys()
        cond_label = self.test_manager.current_condition.blinded_label if self.test_manager.is_blinded else self.test_manager.current_condition.name

        modal = RatingSurveyModal(self.root, self.test_manager.block_index, cond_label)
        self.root.wait_window(modal)

        if modal.submitted:
            r = modal.ratings
            self.test_manager.logger.attach_block_ratings(
                block_index=self.test_manager.block_index,
                difficulty=r["difficulty"],
                stability=r["stability"],
                confidence=r["confidence"],
                fatigue=r["fatigue"]
            )
            # Save CSV report automatically after block rating
            csv_file = "human_test_results.csv"
            self.test_manager.logger.export_csv(csv_file)

        # Advance to next block and condition
        self.test_manager.advance_block()
        self.cursor_engine.set_position(960, 540)
        self.test_manager.reset_trial_metrics((960, 540), self.simulator.current_time)

    def _on_haptic_event(self, evt: HapticEvent) -> None:
        log_str = f"[{evt.timestamp:.1f}s] {evt.event_name}"
        self.haptic_listbox.insert(0, log_str)
        if self.haptic_listbox.size() > 30:
            self.haptic_listbox.delete(30, tk.END)

    def _simulation_loop(self) -> None:
        cond = self.test_manager.current_condition

        # 1. Step Keyboard IMU Adapter (runs keypress inputs through physics, noise, bias, tremor)
        sample = self.key_adapter.update_step()

        # 2. Calibration & Filtering
        calib_state = self.calibrator.update(sample)
        calib_sample = self.calibrator.apply(sample)
        filt_gyro = self.filters.process(calib_sample.gyro)

        # 3. Motion Detection & Uncertainty Model
        motion_state = self.motion_detector.update(filt_gyro)
        uncertainty = self.uncertainty_est.update(calib_sample.gyro, filt_gyro, motion_state)

        # 4. Target Assistance & Condition E Dynamic Sensitivity Suppression
        targets = self.test_manager.targets
        self.target_engine.set_targets(targets)

        friction_factor = 0.0
        if cond.use_friction:
            f_state = self.friction_engine.compute_friction(self.cursor_engine.cursor, targets, uncertainty)
            friction_factor = f_state.friction_factor
            filt_gyro *= f_state.gain

        # 5. Virtual Cursor Engine
        cursor = self.cursor_engine.update(filt_gyro)
        self.test_manager.update_frame((cursor.x, cursor.y))

        assist = self.target_engine.process(cursor, uncertainty, mode=cond.acq_mode)

        # 6. Screen Edge Detection
        new_edges = self.edge_detector.update(cursor)
        for edge in new_edges:
            self.haptic_manager.trigger_edge_bump(edge.value, sample.timestamp)

        # 7. Update Telemetry UI
        cond_text = cond.blinded_label if self.test_manager.is_blinded else cond.name
        self.lbl_cond.config(text=cond_text)
        self.lbl_block_info.config(
            text=f"Block {self.test_manager.block_index} — Trial {self.test_manager.trial_in_block + 1} / {self.test_manager.trials_per_block}"
        )
        self.lbl_layout_info.config(
            text=f"Layout: {self.test_manager.current_layout.value} ({self.test_manager.current_target_size}px)"
        )
        self.lbl_speed.config(text=f"Cursor Speed: {int(cursor.speed)} px/s")
        self.lbl_uncert.config(text=f"Uncertainty R: {int(uncertainty.uncertainty_radius)} px")
        self.lbl_friction.config(text=f"Friction Factor: {friction_factor:.2f}")

        # 8. Render Canvas
        self._render_canvas(cursor, uncertainty, assist)

        # Loop next tick (~100Hz)
        self.root.after(10, self._simulation_loop)

    def _render_canvas(
        self,
        cursor: VirtualCursor,
        uncertainty: UncertaintyState,
        assist: AssistanceResult
    ) -> None:
        self.canvas.delete("all")

        # Render Targets
        for target in self.test_manager.targets:
            tx = target.x * self.scale_x
            ty = target.y * self.scale_y
            tw = target.width * self.scale_x
            th = target.height * self.scale_y

            is_active = (self.test_manager.active_target and self.test_manager.active_target.id == target.id)
            is_hovered = (assist.selected_target and assist.selected_target.id == target.id)

            if is_active:
                fill_color = "#313244"
                outline_color = "#f9e2af"  # Gold boundary for active target to reach
                width = 3
            elif is_hovered:
                fill_color = "#181825"
                outline_color = "#a6e3a1"
                width = 2
            else:
                fill_color = "#1e1e2e"
                outline_color = "#45475a"
                width = 1

            self.canvas.create_rectangle(tx, ty, tx + tw, ty + th, fill=fill_color, outline=outline_color, width=width)
            self.canvas.create_text(tx + tw / 2, ty + th / 2, text=target.label, fill="#cdd6f4", font=("Consolas", 9, "bold"))

        # Render Uncertainty Circle
        cx = cursor.x * self.scale_x
        cy = cursor.y * self.scale_y
        r_uncert = uncertainty.uncertainty_radius * self.scale_x

        self.canvas.create_oval(
            cx - r_uncert, cy - r_uncert,
            cx + r_uncert, cy + r_uncert,
            outline="#cba6f7",
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
