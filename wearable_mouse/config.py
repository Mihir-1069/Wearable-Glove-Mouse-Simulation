"""
Configuration settings for the Wearable Computer Controller / Glove Mouse Simulation.
All tunable parameters for sensors, filters, cursor, uncertainty, target assistance,
gestures, edge feedback, and visualization are centralized here.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # -------------------------------------------------------------------------
    # Simulation Parameters
    # -------------------------------------------------------------------------
    sample_rate: float = 100.0          # IMU sampling frequency in Hz
    dt: float = 1.0 / 100.0             # Time delta per sample (seconds)
    random_seed: int = 42              # Seed for reproducible random simulation

    # -------------------------------------------------------------------------
    # IMU Noise & Imperfection Model
    # -------------------------------------------------------------------------
    gyro_bias_std: float = 0.02         # Gyro static offset std dev (rad/s)
    gyro_noise_std: float = 0.015       # Gyro additive white Gaussian noise std dev (rad/s)
    accel_noise_std: float = 0.05       # Accelerometer noise std dev (m/s^2)
    gyro_drift_rate: float = 0.0001     # Gyro linear bias drift rate (rad/s per second)
    tremor_amplitude: float = 0.03      # Involuntary hand tremor amplitude (rad/s)
    tremor_frequency: float = 10.0      # Hand tremor frequency in Hz (~8-12 Hz typical)

    # -------------------------------------------------------------------------
    # Calibration & Stationary Detection
    # -------------------------------------------------------------------------
    calibration_duration: float = 2.0   # Seconds of stationary state required for calibration
    stationary_gyro_threshold: float = 0.05  # Gyro magnitude threshold for stationary detection (rad/s)
    stationary_accel_threshold: float = 0.2  # Accel variance threshold for stationary detection (m/s^2)
    stationary_window_size: int = 20    # Sliding window size (samples) for variance check

    # -------------------------------------------------------------------------
    # Filtering & Motion Processing
    # -------------------------------------------------------------------------
    lowpass_cutoff: float = 12.0        # Low-pass filter cutoff frequency (Hz)
    deadzone_threshold: float = 0.02    # Angular velocity deadzone cutoff (rad/s)
    exp_smoothing_alpha: float = 0.3    # Exponential filter smoothing factor (0.0 to 1.0)
    moving_average_window: int = 5      # Moving average window size

    # -------------------------------------------------------------------------
    # Orientation Estimation & Posture Reference State
    # -------------------------------------------------------------------------
    complementary_alpha: float = 0.98   # Weight for gyro vs accel in complementary filter
    rest_accel_var_threshold: float = 0.08      # Accel variance threshold for REST posture detection (m/s^2)^2
    rest_gyro_energy_threshold: float = 0.04    # Gyro energy threshold for REST posture detection (rad/s)
    rest_detection_window: int = 15             # Window size for posture variance checks (samples)
    intentional_motion_threshold: float = 0.045  # Angular velocity threshold for intentional movement while RESTing (rad/s)
    rest_activation_threshold: float = 0.045     # Activation cutoff for intentional rest movement (rad/s)
    rest_release_threshold: float = 0.025        # Release cutoff (hysteresis) for intentional rest movement (rad/s)
    rest_persistence_count: int = 3              # Consecutive frames required to activate/deactivate intentional rest motion
    transition_stabilization_duration: float = 0.15 # Posture transition stabilization freeze window (seconds)


    # -------------------------------------------------------------------------
    # Cursor Engine & Sensitivity
    # -------------------------------------------------------------------------
    screen_width: int = 1920            # Virtual screen width in pixels
    screen_height: int = 1080           # Virtual screen height in pixels
    base_sensitivity_x: float = 1200.0  # Sensitivity scale factor X (gyro rad/s -> pixels/s)
    base_sensitivity_y: float = 1200.0  # Sensitivity scale factor Y (gyro rad/s -> pixels/s)
    enable_acceleration: bool = True    # Enable non-linear cursor acceleration
    acceleration_exponent: float = 1.5  # Acceleration curve power factor
    max_cursor_velocity: float = 3000.0 # Maximum allowed cursor speed (pixels/s)
    enable_os_mouse: bool = False       # Safety flag: real OS mouse control disabled by default

    # -------------------------------------------------------------------------
    # Uncertainty Model
    # -------------------------------------------------------------------------
    uncertainty_base_radius: float = 15.0      # Base uncertainty radius (pixels)
    uncertainty_tremor_weight: float = 250.0   # Scale factor for tremor energy
    uncertainty_noise_weight: float = 150.0    # Scale factor for sensor noise
    uncertainty_velocity_weight: float = 0.02  # Scale factor for cursor velocity
    uncertainty_max_radius: float = 120.0      # Max cap on uncertainty circle (pixels)

    # -------------------------------------------------------------------------
    # Target Assistance & Ambiguity
    # -------------------------------------------------------------------------
    assist_mode: str = "D"               # Active mode: "A"=Baseline, "B"=Tolerance, "C"=Snapping, "D"=Full Assistance
    target_tolerance_margin: float = 20.0# Extra pixel margin around target for acquisition
    soft_snap_strength: float = 0.3     # Strength of gravitational pull to target center (0.0 to 1.0)
    ambiguity_ratio_threshold: float = 0.85 # Ratio threshold between top 2 targets to trigger ambiguity

    # -------------------------------------------------------------------------
    # Keybindings & Interaction
    # -------------------------------------------------------------------------
    key_left_click: str = "1"
    key_right_click: str = "2"
    key_double_click: str = "3"
    key_scroll: str = "4"
    key_drag: str = "5"
    key_zoom: str = "6"
    key_pause: str = "p"

    # -------------------------------------------------------------------------
    # Gesture Interaction & Finger Tracking Parameters
    # -------------------------------------------------------------------------
    flex_open_threshold: float = 0.25           # Flex value below which finger is OPEN
    flex_close_threshold: float = 0.70          # Flex value above which finger is CLOSED
    flex_touch_threshold: float = 0.85          # Touch/pressure contact threshold
    gesture_debounce_duration: float = 0.04     # Minimum debouncing duration for finger state transitions (seconds)
    gesture_min_stable_duration: float = 0.08   # Minimum stable pose duration before candidate gesture triggers (seconds)
    gesture_cooldown_duration: float = 0.25     # Refractory cooldown period after emitting action gesture (seconds)
    double_click_max_interval: float = 0.35     # Maximum time window between pinches for DOUBLE_CLICK (seconds)
    drag_hold_duration: float = 0.35            # Pinch hold duration before click transitions to DRAG_START (seconds)
    gesture_max_uncertainty_threshold: float = 80.0  # Max uncertainty radius (px) above which gestures are inhibited
    gesture_inhibit_on_transition: bool = True  # Inhibit action gestures during AIR <-> REST posture transitions

