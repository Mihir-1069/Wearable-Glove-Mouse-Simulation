"""
Main Application Launcher for Wearable Glove Mouse Simulator & Testing Framework.
"""

import os
import sys

# Ensure parent directory is on sys.path so 'wearable_mouse' package imports resolve smoothly
pkg_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(pkg_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import argparse
import tkinter as tk
from wearable_mouse.config import Config
from wearable_mouse.visualization.dashboard import DashboardApp
from wearable_mouse.visualization.human_test_gui import HumanTestGUI
from wearable_mouse.experiments.experiment_runner import ExperimentRunner
from wearable_mouse.experiments.uncertainty_experiment import UncertaintyExperimentRunner
from wearable_mouse.visualization.plots import plot_experiment_results


def main():
    parser = argparse.ArgumentParser(description="Wearable Glove Mouse Simulator & Controlled Experiment Framework")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive Tkinter GUI dashboard")
    parser.add_argument("--human-test", action="store_true", help="Launch interactive Human-in-the-Loop test mode with keyboard WASD controls and subjective rating survey")
    parser.add_argument("--experiment", action="store_true", help="Run automated controlled A/B/C/D benchmark experiments and save CSV")
    parser.add_argument("--uncertainty-test", action="store_true", help="Run 20 controlled stress-test scenarios for uncertainty evaluation (Conditions B, C, D, E)")
    parser.add_argument("--rest-experiment", action="store_true", help="Run controlled A/B/C/D/E benchmark experiment for Resting-Surface Cursor Control")
    parser.add_argument("--trials", type=int, default=15, help="Number of trials per condition for experiment mode")
    parser.add_argument("--os-mouse", action="store_true", help="Enable real host OS cursor control (Safety Warning: disabled by default)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic simulation")
    parser.add_argument("--output-csv", type=str, default="experiment_results.csv", help="Path to save experiment CSV report")

    # Hardware-in-the-Loop (Phase 2) Arguments
    parser.add_argument("--source", choices=["simulation", "serial", "replay"], default="simulation", help="Telemetry input source: 'simulation', 'serial' (ESP32/Wokwi), or 'replay'")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port for ESP32/Wokwi (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--record", type=str, default=None, help="Record raw sensor stream to CSV file")
    parser.add_argument("--replay", type=str, default=None, help="Recorded CSV file path to replay")

    args = parser.parse_args()

    # If --replay is provided, set source to 'replay'
    if args.replay:
        args.source = "replay"

    # Default to --dashboard if no action flag is explicitly passed
    if not args.experiment and not args.dashboard and not args.uncertainty_test and not args.human_test and not args.rest_experiment:
        args.dashboard = True

    config = Config()
    config.random_seed = args.seed
    config.enable_os_mouse = args.os_mouse

    if args.rest_experiment:
        from wearable_mouse.experiments.rest_movement_experiment import RestMovementExperimentRunner
        print("=========================================================================================================")
        print(" Running Controlled Resting-Surface Cursor Control Experiment (Conditions A, B, C, D, E) ")
        print("=========================================================================================================")
        runner = RestMovementExperimentRunner(config)
        results = runner.run_all_conditions()
        csv_path = "rest_movement_experiment_results.csv"
        runner.save_results_csv(results, csv_path)

        print(f"\nRest Movement Experiment Results saved to CSV: {csv_path}\n")
        header = f"{'Code':<6} | {'Condition Name':<52} | {'FalseMove(px)':<14} | {'Detect%':<8} | {'Path(px)':<9} | {'Latency(s)':<10} | {'Missed':<7} | {'FalseAct':<9} | {'Jump(px)':<9}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r.condition_code:<6} | {r.condition_name:<52} | {r.false_cursor_movement_px:>14.2f} | {r.intentional_detection_rate_pct:>7.1f}% | {r.cursor_path_length_px:>9.1f} | {r.movement_latency_sec:>10.3f}s | {r.missed_intentional_movements:>7d} | {r.false_movements_count:>9d} | {r.transition_jumps_px:>9.2f}")
        print("-" * len(header))

    elif args.human_test:
        print("Launching Wearable Mouse Interactive Human-in-the-Loop Test Mode...")
        print("Use WASD or Arrow Keys to move virtual cursor. Press Space or 1 to trigger Target Click.")
        root = tk.Tk()
        app = HumanTestGUI(root, config)
        root.mainloop()

    elif args.uncertainty_test:
        print("=============================================================================================================================")
        print(" Running 20 Controlled Uncertainty Stress-Test Scenarios (Conditions B, C, D, E) ")
        print("=============================================================================================================================")
        runner = UncertaintyExperimentRunner(config)
        results = runner.run_all_scenarios()
        csv_path = "uncertainty_stress_test_results.csv"
        runner.save_results_csv(results, csv_path)

        print(f"\nStress Test Results saved to CSV: {csv_path}\n")
        header = f"{'ScenID':<6} | {'Scenario Name':<38} | {'Cond':<5} | {'Success':<8} | {'Wrong':<6} | {'Missed':<7} | {'Time(s)':<8} | {'Path(px)':<9} | {'UncertR':<8} | {'AvgFrict':<9} | {'MinGain':<8}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r.scenario_id:<6} | {r.scenario_name:<38} | {r.condition_code:<5} | {str(r.successful_acquisition):<8} | {str(r.wrong_target_acquisition):<6} | {str(r.missed_target):<7} | {r.acquisition_time_sec:>7.3f}s | {r.cursor_path_length:>8.1f} | {r.uncertainty_radius:>7.1f}p | {r.average_friction_factor:>8.3f} | {r.minimum_gain_reached:>7.3f}")
        print("-" * len(header))

    elif args.experiment:
        print("==========================================================================================")
        print(" Controlled Target Acquisition Experiment (Conditions A, B, C, D) ")
        print("==========================================================================================")
        runner = ExperimentRunner(config)
        results = runner.run_benchmark(num_trials=args.trials)
        runner.save_results_csv(results, args.output_csv)

        print(f"\nExperiment Results saved to CSV: {args.output_csv}\n")
        header = f"{'Code':<6} | {'Condition Name':<42} | {'Success%':<8} | {'Wrong%':<7} | {'Misclick%':<9} | {'Time(s)':<8} | {'Path(px)':<9} | {'FixMoves':<8}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(f"{r.condition_code:<6} | {r.condition_name:<42} | {r.success_rate:>7.1f}% | {r.wrong_target_rate:>6.1f}% | {r.misclick_rate:>8.1f}% | {r.avg_acquisition_time_sec:>7.3f}s | {r.total_path_length_pixels:>8.1f} | {r.corrective_movements_count:>8d}")
        print("-" * len(header))
        print("\nPlotting 6-metric summary comparison chart...")
        plot_experiment_results(results, save_path="experiment_chart.png")

    elif args.dashboard:
        print("Launching Wearable Mouse Tkinter Visual Dashboard...")
        sensor_source = None
        if args.source == "serial":
            from wearable_mouse.sensors.esp32_serial_adapter import ESP32SerialAdapter
            print(f"Connecting to ESP32 Serial Source on {args.port} @ {args.baud} baud...")
            sensor_source = ESP32SerialAdapter(port=args.port, baudrate=args.baud)
            sensor_source.connect()
        elif args.source == "replay":
            from wearable_mouse.sensors.recorder_replayer import SensorDataReplayer
            print(f"Replaying raw IMU sensor stream from CSV: {args.replay}...")
            sensor_source = SensorDataReplayer(filepath=args.replay, realtime_playback=True)

        if args.record:
            print(f"Recording raw IMU sensor telemetry to: {args.record}")

        root = tk.Tk()
        app = DashboardApp(root, config, sensor_source=sensor_source, record_filepath=args.record)
        root.mainloop()


if __name__ == "__main__":
    main()
