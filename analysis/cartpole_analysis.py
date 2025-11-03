import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

STEPS = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000]
CONTEXTS = [1, 10, 20, 30, 40, 50]
MODEL_RUN_ID = "ebonye_models"
NUM_PENDS = 10


def load_data(step):
    """Load pickled data for a given step"""
    data_path = f"cartpole/inference_run/mse_control_{step}_{MODEL_RUN_ID}/results_maxcontext50_numpends{NUM_PENDS}_indistr_alexcode.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def angle_difference(angle1, angle2):
    """
    Compute the smallest difference between two angles, handling wrapping.
    
    Args:
        angle1: First angle (radians)
        angle2: Second angle (radians)
    
    Returns:
        Smallest angular difference in range [-π, π]
    """
    diff = angle1 - angle2
    return np.arctan2(np.sin(diff), np.cos(diff))


def check_stability(states, theta_target=0.0, theta_threshold=0.3, theta_dot_threshold=1.0, window_size=50):
    """
    Check if the cartpole stabilizes to the upright position.
    Stabilizes iff θ ≈ 0 (upright) and θ_dot is small for the last window_size steps.
    
    Args:
        states: DataFrame or array with columns [time_step, x, x_dot, theta, theta_dot]
        theta_target: Target angle (0 for upright)
        theta_threshold: Threshold for considering angle stable (radians)
        theta_dot_threshold: Threshold for angular velocity (rad/s)
        window_size: Number of final steps to check for stability
    
    Returns:
        bool: True if stable, False otherwise
    """
    if isinstance(states, pd.DataFrame):
        theta = states['theta'].values
        theta_dot = states['theta_dot'].values
    else:
        theta = states[:, 2]
        theta_dot = states[:, 3]
    
    if len(theta) < window_size:
        return False
    
    # Check last window_size steps
    theta_final = theta[-window_size:]
    theta_dot_final = theta_dot[-window_size:]
    
    # Compute angular differences (handles wrapping correctly)
    theta_diffs = np.array([angle_difference(t, theta_target) for t in theta_final])
    
    # Check if all values in the window are close to target
    theta_stable = np.all(np.abs(theta_diffs) < theta_threshold)
    theta_dot_stable = np.all(np.abs(theta_dot_final) < theta_dot_threshold)
    
    return theta_stable and theta_dot_stable


def analyze_stability_rates(steps=None, contexts=None, save_dir="analysis/cartpole_results"):
    """
    Analyze stability rates as a function of checkpoint step and context length.
    
    Args:
        steps: List of checkpoint steps to analyze (default: all STEPS)
        contexts: List of context lengths to analyze (default: all CONTEXTS)
        save_dir: Directory to save results and visualizations
    """
    if steps is None:
        steps = STEPS
    if contexts is None:
        contexts = CONTEXTS
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Store stability rates: stability_rates[step][context] = rate
    stability_rates = {step: {context: 0.0 for context in contexts} for step in steps}
    stability_counts = {step: {context: 0 for context in contexts} for step in steps}
    
    print("Analyzing stability rates across all checkpoints and contexts...")
    
    for step in tqdm(steps, desc="Processing steps"):
        video_dir = f"cartpole/videos/cartpole_inference_gym_runs/{MODEL_RUN_ID}/step_{step}"
        
        if not os.path.exists(video_dir):
            print(f"⚠️  WARNING: Directory not found: {video_dir}")
            continue
        
        for run_idx in range(NUM_PENDS):
            run_dir = os.path.join(video_dir, f"run_{run_idx:03}")
            
            if not os.path.exists(run_dir):
                continue
            
            for context in contexts:
                # Find the context directory
                context_dirs = [d for d in os.listdir(run_dir) if f"context_{context}" in d]
                
                if not context_dirs:
                    continue
                
                context_dir = os.path.join(run_dir, context_dirs[0])
                states_file = os.path.join(context_dir, "states.csv")
                
                if not os.path.exists(states_file):
                    continue
                
                # Load and check stability
                states_df = pd.read_csv(states_file)
                is_stable = check_stability(states_df)
                
                if is_stable:
                    stability_counts[step][context] += 1
        
        # Calculate rates for this step
        for context in contexts:
            if NUM_PENDS > 0:
                stability_rates[step][context] = stability_counts[step][context] / NUM_PENDS
    
    # Create visualizations
    create_stability_visualizations(stability_rates, stability_counts, steps, contexts, save_dir)
    
    # Save raw data
    results_file = os.path.join(save_dir, "stability_rates.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump({
            'rates': stability_rates,
            'counts': stability_counts,
            'steps': steps,
            'contexts': contexts
        }, f)
    
    # Save as CSV
    csv_file = os.path.join(save_dir, "stability_rates.csv")
    rows = []
    for step in steps:
        for context in contexts:
            rows.append({
                'step': step,
                'context': context,
                'stability_rate': stability_rates[step][context],
                'stable_count': stability_counts[step][context],
                'total_count': NUM_PENDS
            })
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    
    print(f"\n✓ Results saved to {save_dir}")
    return stability_rates, stability_counts


def create_stability_visualizations(stability_rates, stability_counts, steps, contexts, save_dir, metric_name="Stability Rate"):
    """Create comprehensive visualizations of stability rates
    
    Args:
        stability_rates: Dictionary of rates by step and context
        stability_counts: Dictionary of counts by step and context
        steps: List of checkpoint steps
        contexts: List of context lengths
        save_dir: Directory to save plots
        metric_name: Name of the metric being visualized (default: "Stability Rate")
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare data for plotting
    rate_matrix = np.zeros((len(steps), len(contexts)))
    for i, step in enumerate(steps):
        for j, context in enumerate(contexts):
            rate_matrix[i, j] = stability_rates[step][context]
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rate_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(contexts, fontsize=12)
    ax.set_xlabel('Context Length', fontsize=14, fontweight='bold')
    
    # Show all steps on y-axis
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(steps, fontsize=11)
    ax.set_ylabel('Checkpoint Step', fontsize=14, fontweight='bold')
    
    ax.set_title(f'{metric_name}: Checkpoint Step vs Context Length\n({len(steps)} steps × {len(contexts)} contexts = {len(steps)*len(contexts)} total evaluations)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name, rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Line plot: Stability rate vs checkpoint step for each context (overlaid)
    fig, ax = plt.subplots(figsize=(14, 8))
    for context in contexts:
        rates = [stability_rates[step][context] for step in steps]
        ax.plot(steps, rates, marker='o', linewidth=2.5, markersize=6, label=f'Context {context}', alpha=0.8)
    
    ax.set_xlabel('Checkpoint Step', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_name} vs Checkpoint Step (by Context)\nAll {len(steps)} checkpoint steps shown', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_vs_step.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2b. Separate plots: Stability rate vs checkpoint step for each context
    n_contexts = len(contexts)
    n_cols = 3
    n_rows = (n_contexts + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if n_contexts > 1 else [axes]
    
    for idx, context in enumerate(contexts):
        ax = axes[idx]
        rates = [stability_rates[step][context] for step in steps]
        ax.plot(steps, rates, marker='o', linewidth=2.5, markersize=6, color='#2E86AB')
        
        ax.set_xlabel('Checkpoint Step', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f'Context {context}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    # Hide unused subplots
    for idx in range(n_contexts, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{metric_name} vs Checkpoint Step (Separate by Context)', 
                 fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_vs_step_separate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Line plot: Stability rate vs context for each checkpoint (overlaid)
    fig, ax = plt.subplots(figsize=(12, 8))
    for step in steps:
        rates = [stability_rates[step][context] for context in contexts]
        ax.plot(contexts, rates, marker='o', linewidth=2, markersize=6, label=f'Step {step}', alpha=0.8)
    
    ax.set_xlabel('Context Length', fontsize=14, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
    ax.set_title(f'{metric_name} vs Context Length (by Checkpoint)\nAll {len(steps)} checkpoint steps', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(contexts)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_vs_context.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3b. Separate plots: Stability rate vs context for each checkpoint step
    n_steps = len(steps)
    n_cols = 4
    n_rows = (n_steps + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_steps > 1 else [axes]
    
    for idx, step in enumerate(steps):
        ax = axes[idx]
        rates = [stability_rates[step][context] for context in contexts]
        ax.plot(contexts, rates, marker='o', linewidth=2, markersize=6, color='#A23B72')
        
        ax.set_xlabel('Context Length', fontsize=10, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_title(f'Step {step}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(contexts)
    
    # Hide unused subplots
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f'{metric_name} vs Context Length (Separate by Checkpoint)', 
                 fontsize=18, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_vs_context_separate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(contexts, steps)
    surf = ax.plot_surface(X, Y, rate_matrix, cmap='viridis', alpha=0.8, edgecolor='none')
    
    ax.set_xlabel('Context Length', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Checkpoint Step', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel(metric_name, fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f'3D Surface: {metric_name}\n({len(steps)} steps × {len(contexts)} contexts)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_3d_surface.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Bar chart comparison: Early vs Late checkpoints
    early_steps = steps[:len(steps)//2]
    late_steps = steps[len(steps)//2:]
    
    early_rates = [np.mean([stability_rates[step][context] for step in early_steps]) for context in contexts]
    late_rates = [np.mean([stability_rates[step][context] for step in late_steps]) for context in contexts]
    
    x = np.arange(len(contexts))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(x - width/2, early_rates, width, label='Early Checkpoints', alpha=0.8)
    bars2 = ax.bar(x + width/2, late_rates, width, label='Late Checkpoints', alpha=0.8)
    
    ax.set_xlabel('Context Length', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Average {metric_name}', fontsize=14, fontweight='bold')
    ax.set_title(f'Early vs Late Checkpoint {metric_name} (by Context)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_early_vs_late.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Simple line plot: Average stability rate vs checkpoint step (averaged across all contexts)
    fig, ax = plt.subplots(figsize=(14, 8))
    avg_rates_by_step = [np.mean([stability_rates[step][context] for context in contexts]) for step in steps]
    ax.plot(steps, avg_rates_by_step, marker='o', linewidth=2.5, markersize=8, color='#2E86AB', label=f'Average {metric_name}')
    
    ax.set_xlabel('Checkpoint Step', fontsize=16, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=16, fontweight='bold')
    ax.set_title(f'{metric_name} vs Checkpoint Step\n(Averaged Across All Contexts)', fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=14, loc='best')
    
    # Add minor gridlines
    ax.grid(True, which='minor', alpha=0.1)
    ax.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_vs_step_averaged.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Simple line plot: Average stability rate vs context (averaged across all checkpoints)
    fig, ax = plt.subplots(figsize=(12, 8))
    avg_rates_by_context = [np.mean([stability_rates[step][context] for step in steps]) for context in contexts]
    ax.plot(contexts, avg_rates_by_context, marker='o', linewidth=3, markersize=10, color='#A23B72', label=f'Average {metric_name}')
    
    ax.set_xlabel('Context Length', fontsize=16, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=16, fontweight='bold')
    ax.set_title(f'{metric_name} vs Context Length\n(Averaged Across All Checkpoint Steps)', fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linewidth=1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(contexts)
    ax.legend(fontsize=14, loc='best')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'stability_vs_context_averaged.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Generated 9 visualization plots in {save_dir}")


def check_stability_by_phase(states, controls, theta_target=0.0, theta_threshold=0.3, theta_dot_threshold=1.0, window_size=50, lqr_handoff_threshold=0.5):
    """
    Check stability separately for swing-up and LQR phases.
    
    Swing-up success: Successfully hands off to LQR (>=50% LQR commands after first LQR)
    LQR stability: Achieves equilibrium during LQR phase (θ≈0, upright)
    
    Args:
        states: DataFrame or tensor with state data (x, x_dot, theta, theta_dot)
        controls: Tensor or array with shape (timesteps, 2) where controls[:, 1] is the mode flag
        theta_target: Target angle (0 for upright)
        theta_threshold: Threshold for considering angle stable (radians)
        theta_dot_threshold: Threshold for angular velocity (rad/s)
        window_size: Number of final steps to check for LQR stability
        lqr_handoff_threshold: Minimum fraction of LQR commands after first LQR (default: 0.5)
    
    Returns:
        tuple: (swing_up_success, lqr_stable)
            - swing_up_success: bool, whether successfully handed off to LQR controller
            - lqr_stable: bool, whether stable during LQR phase (equilibrium)
    """
    # Extract control flags
    if isinstance(controls, torch.Tensor):
        flags = controls[:, 1].cpu().numpy()
    else:
        flags = controls[:, 1]
    
    # Extract states
    if isinstance(states, pd.DataFrame):
        theta = states['theta'].values
        theta_dot = states['theta_dot'].values
    elif isinstance(states, torch.Tensor):
        if states.device.type == 'cuda':
            states_cpu = states.cpu()
        else:
            states_cpu = states
        theta = states_cpu[:, 2].numpy()
        theta_dot = states_cpu[:, 3].numpy()
    else:
        theta = states[:, 2]
        theta_dot = states[:, 3]
    
    # Check swing-up success: successful handoff to LQR
    swing_up_success = False
    lqr_indices = np.where(flags == 1.0)[0]
    
    if len(lqr_indices) > 0:
        # Find first LQR command at time step t
        first_lqr_idx = lqr_indices[0]
        
        # Get all commands from t to t_max
        subsequent_flags = flags[first_lqr_idx:]
        
        # Check if >= threshold% of subsequent commands are LQR
        if len(subsequent_flags) > 0:
            lqr_fraction = np.sum(subsequent_flags == 1.0) / len(subsequent_flags)
            swing_up_success = lqr_fraction >= lqr_handoff_threshold
    
    # Check LQR phase stability using equilibrium criteria
    lqr_stable = False
    if len(lqr_indices) >= window_size:
        # Take the last window_size steps of LQR phase
        lqr_window_indices = lqr_indices[-window_size:]
        theta_lqr = theta[lqr_window_indices]
        theta_dot_lqr = theta_dot[lqr_window_indices]
        
        # Check stability using equilibrium criteria
        theta_diffs = np.array([angle_difference(t, theta_target) for t in theta_lqr])
        
        theta_stable_lqr = np.all(np.abs(theta_diffs) < theta_threshold)
        theta_dot_stable_lqr = np.all(np.abs(theta_dot_lqr) < theta_dot_threshold)
        lqr_stable = theta_stable_lqr and theta_dot_stable_lqr
    
    return swing_up_success, lqr_stable


def compare_lqr_vs_swing_up(steps=None, contexts=None, save_dir="analysis/cartpole_lqr_comparison"):
    """
    Analyze and compare performance separately for swing-up and LQR phases.
    
    Generates two sets of visualizations:
    1. Swing-up success rate: % of runs that successfully hand off to LQR (≥50% LQR commands after first LQR)
    2. LQR stability rate: % of runs that achieve equilibrium during LQR phase (θ≈0, upright)
    
    Args:
        steps: List of checkpoint steps to analyze (default: all STEPS)
        contexts: List of context lengths to analyze (default: all CONTEXTS)
        save_dir: Directory to save results and visualizations
    """
    if steps is None:
        steps = STEPS
    if contexts is None:
        contexts = CONTEXTS
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Store success/stability rates for each phase
    swing_up_success_rates = {step: {context: 0.0 for context in contexts} for step in steps}
    swing_up_success_counts = {step: {context: 0 for context in contexts} for step in steps}
    
    lqr_stability_rates = {step: {context: 0.0 for context in contexts} for step in steps}
    lqr_stability_counts = {step: {context: 0 for context in contexts} for step in steps}
    
    print("Analyzing swing-up vs LQR phase performance across all checkpoints and contexts...")
    print("Swing-up success: ≥50% LQR commands after first LQR transition")
    print("LQR stability: Equilibrium (θ≈0, upright) during LQR phase")
    
    for step in tqdm(steps, desc="Processing steps"):
        # Load the pickle data to get states and controls
        data_path = f"cartpole/inference_run/mse_control_{step}_{MODEL_RUN_ID}/results_maxcontext50_numpends{NUM_PENDS}_indistr_alexcode.pkl"
        
        if not os.path.exists(data_path):
            print(f"⚠️  WARNING: Data file not found: {data_path}")
            continue
        
        try:
            with open(data_path, 'rb') as f:
                cartmasses, polemasses, polelengths, phase_data, controls_data, data_and_controls, pends, folder_name = pickle.load(f)
        except Exception as e:
            print(f"⚠️  WARNING: Failed to load {data_path}: {e}")
            continue
        
        for context in contexts:
            if context not in controls_data or context not in phase_data:
                continue
            
            for run_idx in range(min(NUM_PENDS, len(controls_data[context]))):
                controls = controls_data[context][run_idx]
                states = phase_data[context][run_idx]
                
                # Check swing-up success (handoff) and LQR stability (equilibrium)
                swing_up_success, lqr_stable = check_stability_by_phase(states, controls)
                
                if swing_up_success:
                    swing_up_success_counts[step][context] += 1
                
                if lqr_stable:
                    lqr_stability_counts[step][context] += 1
        
        # Calculate rates for this step
        for context in contexts:
            if NUM_PENDS > 0:
                swing_up_success_rates[step][context] = swing_up_success_counts[step][context] / NUM_PENDS
                lqr_stability_rates[step][context] = lqr_stability_counts[step][context] / NUM_PENDS
    
    # Create visualizations for both phases
    print("\nGenerating swing-up success rate visualizations...")
    create_stability_visualizations(swing_up_success_rates, swing_up_success_counts, steps, contexts, 
                                   os.path.join(save_dir, "swing_up"), metric_name="Swing-up Success Rate")
    
    print("\nGenerating LQR stability rate visualizations...")
    create_stability_visualizations(lqr_stability_rates, lqr_stability_counts, steps, contexts, 
                                   os.path.join(save_dir, "lqr_stability"), metric_name="LQR Stability Rate")
    
    # Save combined raw data
    results_file = os.path.join(save_dir, "lqr_comparison_data.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump({
            'swing_up_success_rates': swing_up_success_rates,
            'swing_up_success_counts': swing_up_success_counts,
            'lqr_stability_rates': lqr_stability_rates,
            'lqr_stability_counts': lqr_stability_counts,
            'steps': steps,
            'contexts': contexts
        }, f)
    
    # Save as CSV
    csv_file = os.path.join(save_dir, "lqr_comparison_data.csv")
    rows = []
    for step in steps:
        for context in contexts:
            rows.append({
                'step': step,
                'context': context,
                'swing_up_success_rate': swing_up_success_rates[step][context],
                'swing_up_success_count': swing_up_success_counts[step][context],
                'lqr_stability_rate': lqr_stability_rates[step][context],
                'lqr_stable_count': lqr_stability_counts[step][context],
                'total_runs': NUM_PENDS
            })
    pd.DataFrame(rows).to_csv(csv_file, index=False)
    
    print(f"\n✓ LQR comparison results saved to {save_dir}")
    return swing_up_success_rates, lqr_stability_rates


def main():
    parser = argparse.ArgumentParser(description='Cartpole Analysis Tool')
    parser.add_argument('--sr', '--stability-rates', action='store_true', 
                        dest='stability_rates',
                        help='Analyze and visualize stability rates')
    parser.add_argument('--lqr', '--lqr-comparison', action='store_true',
                        dest='lqr_comparison',
                        help='Compare LQR vs swing-up performance')
    parser.add_argument('--steps', type=int, nargs='+', 
                        help='Specific checkpoint steps to analyze (default: all)')
    parser.add_argument('--contexts', type=int, nargs='+',
                        help='Specific context lengths to analyze (default: all)')
    parser.add_argument('--save-dir', type=str, default='analysis/cartpole_results',
                        help='Directory to save results (default: analysis/cartpole_results)')
    
    args = parser.parse_args()
    
    steps = args.steps if args.steps else STEPS
    contexts = args.contexts if args.contexts else CONTEXTS
    
    if args.stability_rates:
        analyze_stability_rates(steps=steps, contexts=contexts, save_dir=args.save_dir)
    elif args.lqr_comparison:
        compare_lqr_vs_swing_up(steps=steps, contexts=contexts, save_dir=args.save_dir)
    else:
        print("No analysis mode selected.")
        print("Available options:")
        print("  --sr   : Analyze overall stability rates")
        print("  --lqr  : Compare LQR vs swing-up performance")
        print("\nExample: python cartpole_analysis.py --sr")
        print("Example: python cartpole_analysis.py --lqr")


if __name__ == "__main__":
    main()

