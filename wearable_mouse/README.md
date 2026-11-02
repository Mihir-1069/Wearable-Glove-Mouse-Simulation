# Wearable Computer Controller / Glove Mouse Simulation & Testing Environment

A modular Python simulation framework, error-tolerant target acquisition engine, interactive visual dashboard, and benchmark testing environment for camera-free wearable glove mouse systems.

---

## 1. Overview & System Concept

Human hand movements and IMU tracking are inherently imperfect:
- Accelerometers and gyroscopes suffer from **sensor noise**, **zero-rate bias**, and **slow gyro drift**.
- Human hands exhibit **involuntary tremor** (~8–12 Hz), micro-oscillations, and **cursor jitter**.
- Traditional pixel-exact point-and-click interfaces require fine motor control that is difficult to maintain during air gestures.

This project implements a complete **software simulation environment** to model, process, filter, and evaluate human motion dynamics and error-tolerant target acquisition *before* connecting real ESP32 + MPU6050 physical hardware.

---

## 2. Architecture & Pipeline

```
Raw IMU Simulator / Hardware Stream
                │
                ▼
        [ Calibration ]
   (Stationary Gyro Bias & Baseline Gravity Offset)
                │
                ▼
      [ Modular Filter Pipeline ]
 (Low-Pass IIR → Deadzone Threshold → Exponential Smoothing)
                │
                ▼
       [ Motion Classification ]
 (Stationary / Intentional / Tremor / Rapid Movement States)
                │
         ┌──────┴──────────────────────┐
         ▼                             ▼
[ Relative Cursor Engine ]     [ Uncertainty Model ]
 (Sensitivity & Acceleration) (Noise, Tremor, Speed Radius)
         │                             │
         └──────────────┬──────────────┘
                        ▼
       [ Target Assistance Engine ]
  (Tolerance Hitboxes, Ambiguity Check, Soft Snapping)
                        │
                        ▼
      [ Gesture & State Interpreter ]
         (ACTIVE vs PAUSED State Machine)
                        │
                        ▼
   [ Visual Dashboard / Edge Haptics / Logger ]
```

### Key Components

- `simulation/imu_simulator.py`: Configurable 6-axis IMU physics engine with AWGN noise, zero-rate bias, linear bias drift, and sinusoidal 3D hand tremor.
- `processing/calibration.py`: `StationaryDetector` and `Calibrator` estimating gyro bias without continuous frame-by-frame resets.
- `processing/filters.py`: Standardized `BaseFilter` pipeline (`LowPassFilter`, `DeadZoneFilter`, `MovingAverageFilter`, `ExponentialSmoothingFilter`).
- `processing/posture_reference.py`: Manages posture state classification (`AIR`, `REST`, `TRANSITION`), dual-threshold hysteresis movement detection (`REST_NOISE` vs `REST_INTENTIONAL_MOVEMENT`), and relative orientation re-anchoring.
- `processing/uncertainty.py`: Computes dynamic spatial uncertainty circle radius and motion confidence score.
- `cursor/target_assistance.py`: Supports 4 switchable modes:
  - **Mode A (Baseline)**: Exact point-and-click matching.
  - **Mode B (Tolerance)**: Expands target hit areas by uncertainty radius.
  - **Mode C (Soft Snapping)**: Gravitational pull vector toward target centroid.
  - **Mode D (Full Assistance)**: Combined tolerance hit selection + soft snapping + ambiguity suppression.
- `gestures/gesture_engine.py`: Manages global `ACTIVE` vs `PAUSED` system states and translates flex events.
- `interaction/edge_feedback.py` & `haptic_events.py`: Screen boundary detection and vibration feedback logging.
- `visualization/dashboard.py`: Real-time Tkinter GUI canvas displaying virtual screen, cursor, uncertainty region, target states, posture/reference telemetry, telemetry meters, and keyboard controls.
- `experiments/experiment_runner.py`: Automated benchmark evaluator testing Modes A, B, C, D across identical target sequences and exporting CSV reports.
- `experiments/rest_movement_experiment.py`: Automated 5-condition benchmark runner evaluating resting-surface cursor stability, noise suppression, and intentional movement responsiveness (Conditions A–E).

---

## 3. Installation

Ensure Python 3.9+ is installed. Clone or extract the project directory:

```bash
cd c:/Users/mihir/Desktop/gm/wearable_mouse
pip install -r requirements.txt
```

---

## 4. How to Run

### Interactive GUI Dashboard

Launch the real-time Tkinter dashboard:

```bash
python app.py --dashboard
```

**Dashboard Controls:**
- **Scenario Dropdown**: Select motion patterns (`Target Acquisition Sequence`, `Left-Right Sweep`, `Tremor Only`, etc.).
- **Assistance Mode**: Switch between Mode A (Baseline), Mode B (Tolerance), Mode C (Snapping), Mode D (Full Assistance), Mode E (Dynamic Friction), and Mode E+C (Hybrid).
- **Keyboard Gesture Simulator**:
  - `1`: Left Click
  - `2`: Right Click
  - `3`: Double Click
  - `4`: Scroll Up
  - `5`: Drag Start/Stop
  - `6`: Zoom In
  - `P`: Pause / Unpause System (Disables cursor movement and accidental clicks)

### Human-in-the-Loop Software Test Mode

Launch the interactive human test mode to evaluate cursor control with your own keyboard inputs before hardware integration:

```bash
python app.py --human-test
```

**Interactive Test Features:**
- **Full Sensor Pipeline**: WASD / Arrow key directional input generates synthetic 3D gyro rates, passing through complete sensor noise, tremor, bias drift, filtering, uncertainty estimation, and dynamic sensitivity suppression.
- **Controlled Layouts & Target Sizes**: Automatically cycles through 4 target sizes (120px, 80px, 40px, 20px) and 3 layouts (`ISOLATED`, `DENSE_GRID`, `TOOLBAR`).
- **Blinded Condition Evaluation**: Presents conditions under neutral blinded labels ("Condition 1", "Condition 2") to eliminate user bias when comparing Filtered (B), Uncertainty Tolerance (C), Dynamic Friction (E), and Hybrid (E+C).
- **Subjective Rating Survey**: Prompts the user with an interactive rating modal after each block of 5 trials to score Perceived Difficulty, Cursor Stability, Clicking Confidence, and Fatigue (1–5 scale).
- **CSV Exporting**: Writes complete trial performance metrics and subjective ratings to `human_test_results.csv`.

### Resting-Surface Cursor Control Experiment

To evaluate resting-surface noise suppression, zero-jump posture transitions, and intentional movement responsiveness across Conditions A–E:

```bash
python app.py --rest-experiment
```

Generates `rest_movement_experiment_results.csv` evaluating:
- **False cursor movement** while resting (px)
- **Transition position jumps** ($\Delta X, \Delta Y$ on posture change)
- **Intentional movement detection rate** (%)
- **Movement latency** (seconds)

### Running Unit & Pipeline Tests

Run pytest suite:

```bash
python -m pytest tests/ -v
```

---

## 5. Running Automated Experiments

To run the automated empirical benchmark comparing Conditions B, C, D, and E across 20 stress scenarios:

```bash
python app.py --uncertainty-test
```

This runs target acquisition trials under identical noise/tremor conditions and outputs:
- CSV metric summary: `experiment_results.csv`
- Comparative benchmark metrics: path length, acquisition time, misclicks, and friction gains.

**Metrics Recorded:**
- Target acquisition time (seconds)
- Successful selections vs wrong selections
- Misclick rate (%)
- Total cursor path length (pixels)
- Corrective movements count (direction reversals)

---

## 6. How to Modify Parameters

All tunable parameters are centralized in `config.py`:

```python
# Sample rate & simulation seed
sample_rate: float = 100.0
random_seed: int = 42

# IMU Noise & Tremor
gyro_bias_std: float = 0.02
gyro_noise_std: float = 0.015
tremor_amplitude: float = 0.03
tremor_frequency: float = 10.0

# Sensitivity & Acceleration
base_sensitivity_x: float = 1200.0
base_sensitivity_y: float = 1200.0
acceleration_exponent: float = 1.5

# Target Assistance & Ambiguity
soft_snap_strength: float = 0.3
ambiguity_ratio_threshold: float = 0.85
```

---

## 7. Phase 2: Hardware-in-the-Loop (HITL) Simulation using Wokwi

Phase 2 introduces hardware-in-the-loop simulation connecting Wokwi ESP32 + MPU6050 firmware to the existing Python processing pipeline via serial telemetry stream.

### Wokwi Project Structure & Circuit Wiring

- **Directory**: `wokwi/`
  - `diagram.json`: Wokwi canvas circuit layout.
  - `wokwi.toml`: Wokwi simulation configuration.
  - `firmware/main.cpp`: Arduino C++ ESP32 firmware.
  - `firmware/platformio.ini`: PlatformIO build config.

**I2C & Pin Mapping:**
- **MPU6050 SDA**: GPIO 21
- **MPU6050 SCL**: GPIO 22
- **MPU6050 Power / GND**: 3.3V / GND (I2C address: `0x68`)
- **Status LED**: GPIO 2 (Gradients green when active)
- **Pause Pushbutton**: GPIO 15 (Toggles `PAUSE_ENABLED` state)

### Serial Telemetry Protocol

Sampling rate: **100 Hz** (10 ms interval).

- **IMU Frame Data**:
  ```csv
  IMU,timestamp_ms,ax,ay,az,gx,gy,gz
  ```
  *Example*: `IMU,1024,0.02,0.01,9.81,0.001,-0.004,0.012`
  - **Units**: Accelerometer `ax, ay, az` in $\text{m/s}^2$; Gyroscope `gx, gy, gz` in $\text{rad/s}$.

- **Diagnostic Messages**:
  - `STATUS,MPU6050_OK`
  - `ERROR,MPU6050_NOT_FOUND`
  - `STATUS,PAUSE_ENABLED` / `STATUS,PAUSE_DISABLED`

### Launch Commands for HITL Simulation

1. **Run Python with Serial Source (Wokwi or Physical ESP32)**:
   ```bash
   python app.py --source=serial --port COM3 --baud 115200
   ```

2. **Record Raw ESP32 Sensor Telemetry to CSV**:
   ```bash
   python app.py --source=serial --port COM3 --record real_imu.csv
   ```

3. **Replay Recorded Sensor Telemetry**:
   ```bash
   python app.py --source=replay --replay real_imu.csv
   ```

---

## 8. Future Hardware Integration Roadmap (ESP32 + MPU6050)

When moving from Wokwi HITL simulation to physical hardware:

1. **Physical Serial / BLE Connection**:
   - Connect physical ESP32 via USB-to-Serial or BLE SPP.
2. **Flex Sensors**:
   - Connect analog flex sensor voltage divider pins on ESP32 ADC.
3. **Physical Haptic Feedback**:
   - Connect ERM/LRA vibration motor PWM driver on ESP32 GPIO.
4. **Zero Algorithm Rewriting**:
   - The calibration, filtering, motion classification, cursor scaling, uncertainty model, target assistance, dynamic sensitivity suppression, and pause states remain **100% untouched**.

---

## 9. Known Limitations

- **Euler Angle Gimbal Lock**: The current orientation estimator uses Euler angles (roll, pitch, yaw) suitable for standard desktop tilting ranges. Extremely steep vertical wrist flips near $\pm 90^\circ$ pitch may benefit from full quaternion integration.
- **2D Screen Projection**: 3D wrist rotation is projected onto a 2D desktop screen coordinate space. Extreme off-axis wrist twist roll can introduce slight cross-coupling between horizontal and vertical cursor velocity.
- **Fixed Display Resolution Assumption**: Virtual screen coordinates default to 1920x1080; high-DPI scaling should be calibrated per display.
