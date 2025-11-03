
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

def load_controls(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        steps: list[int] = []
        actions: list[float] = []
        for row in reader:
            steps.append(int(row['time_step']))
            actions.append(float(row['control_action']))
    return np.array(steps), np.array(actions)


def load_states(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load states CSV in either format:
    - Columns: time_step, x, x_dot, theta, theta_dot
    - Columns: time_step, state (string like "[x x_dot theta theta_dot]")
    Returns (steps, states_matrix[NumSteps x 4]).
    """
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        steps: list[int] = []
        states: list[list[float]] = []

        has_split_cols = all(name in fieldnames for name in ['x', 'x_dot', 'theta', 'theta_dot'])
        has_state_str = 'state' in fieldnames

        for row in reader:
            steps.append(int(row['time_step']))
            if has_split_cols:
                states.append([
                    float(row['x']),
                    float(row['x_dot']),
                    float(row['theta']),
                    float(row['theta_dot']),
                ])
            elif has_state_str:
                s = row['state']
                # Parse formats like "[0.1 0.2 0.3 0.4]" or "[0.1, 0.2, 0.3, 0.4]"
                s = s.strip().lstrip('[').rstrip(']')
                # Support both comma and space separated
                parts = [p for p in s.replace(',', ' ').split() if p]
                if len(parts) != 4:
                    raise ValueError(f"Invalid state format in {path}: '{row['state']}'")
                states.append([float(v) for v in parts])
            else:
                raise ValueError(f"states.csv missing required columns in {path}")

    return np.array(steps), np.array(states)

PATHS: dict[str, tuple[str, str]] = {
    # "Model (Context 1)": (
    #     "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_004/run_004_mC2.00_mP0.57_L1.48_context_1/controls.csv",
    #     "red",
    # ),
    "Model (Context 50)": (
        "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_004/run_004_mC2.00_mP0.57_L1.48_context_50/controls.csv",
        "green",
    ),
    "Given Context": (
        "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_004/run_004_mC2.00_mP0.57_L1.48/controls.csv",
        "black",
    ),
}


def controls_graph(paths: dict[str, tuple[str, str]]) -> None:
    plt.figure(figsize=(10, 5))
    for label, (path, color) in paths.items():
        steps, actions = load_controls(path)
        if label == "Given Context":
            # Only plot first 50 time steps for Given Context
            steps = steps[:50]
            actions = actions[:50]
            plt.plot(steps, actions, label=label, color=color, alpha=1)
        else:
            plt.scatter(steps, actions, label=label, color=color, s=14, alpha=0.85)

    plt.title('Control Actions Over Time (Context 50)')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('context_1_graph.tiff', dpi=150, format='tiff')
    plt.close()


def derive_states_path(controls_csv_path: str) -> str:
    directory = os.path.dirname(controls_csv_path)
    return os.path.join(directory, 'states.csv')


def states_graph(paths: dict[str, tuple[str, str]]) -> None:
    labels = ['Cart Position: x', 'Cart Velocity: x_dot', 'Pole Angle: theta', 'Pole Angular Velocity: theta_dot']
    ylabels = ['x', 'x_dot', 'theta', 'theta_dot']

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for series_label, (controls_csv, color) in paths.items():
        states_csv = derive_states_path(controls_csv)
        steps, states = load_states(states_csv)
        if series_label == 'Given Context':
            # Only plot first 50 time steps for Given Context
            steps = steps[:50]
            states = states[:50]
            for i in range(4):
                ax = axes[i]
                ax.plot(steps, states[:, i], label=series_label, color=color, alpha=1)
        else:
            for i in range(4):
                ax = axes[i]
                ax.scatter(steps, states[:, i], label=series_label, color=color, s=14, alpha=0.85)

    for i, ax in enumerate(axes):
        ax.set_title(labels[i] + ' Over Time (Context 50)')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(ylabels[i])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig('context_1_states_graph.tiff', dpi=150, format='tiff')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot controls and/or states graphs from CSVs')
    parser.add_argument('--mode', choices=['controls', 'states', 'both'], default='both')
    args = parser.parse_args()

    if args.mode in ('controls', 'both'):
        controls_graph(PATHS)
    if args.mode in ('states', 'both'):
        states_graph(PATHS)


if __name__ == '__main__':
    main()

# PATHS1: dict[str, tuple[str, str]] = {
#     "Ground Truth Run 0": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_000/run_000_mC2.00_mP0.95_L1.19/controls.csv",
#         "black",
#     ),
#     "Ground Truth Run 1": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_001/run_001_mC2.00_mP0.31_L1.30/controls.csv",
#         "red",
#     ),
#     "Ground Truth Run 2": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_002/run_002_mC2.00_mP0.75_L1.45/controls.csv",
#         "blue",
#     ),
#     "Ground Truth Run 3": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_003/run_003_mC2.00_mP0.90_L1.33/controls.csv",
#         "green",
#     ),
#     "Ground Truth Run 4": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_004/run_004_mC2.00_mP0.57_L1.48/controls.csv",
#         "yellow",
#     ),
#     "Ground Truth Run 5": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_005/run_005_mC2.00_mP0.48_L1.12/controls.csv",
#         "purple",
#     ),
#     "Ground Truth Run 6": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_006/run_006_mC2.00_mP0.31_L1.46/controls.csv",
#         "orange",
#     ),
#     "Ground Truth Run 7": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_007/run_007_mC2.00_mP0.74_L1.06/controls.csv",
#         "brown",
#     ),
#     "Ground Truth Run 8": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_008/run_008_mC2.00_mP0.87_L1.29/controls.csv",
#         "pink",
#     ),
#     "Ground Truth Run 9": (
#         "videos/cartpole_inference_gym_runs/b3725997-9aee-4578-b668-d33e7cb29c4e/step_30000/run_009/run_009_mC2.00_mP0.54_L1.42/controls.csv",
#         "gray",
#     )
# }