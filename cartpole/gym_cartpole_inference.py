import os
import gym
import math
import time
# import imageio
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym_continuous_cartpole import ContinuousCartPoleEnv
from gym_cartpole_swingup_lqr import swingup_lqr_controller

# PYTHONPATH=/home/aidan/Code/Work/berkeley/final/cartpole


def run_single_system(masscart, masspole, length, state_data, ctrl_data, run_idx, save_dir, context=None):
    # env = ContinuousCartPoleEnv(
    #     masscart=masscart,
    #     masspole=masspole,
    #     length=length,
    #     render_mode="rgb_array"
    # )

    # import pdb; pdb.set_trace()
    # state_data = state_data.cpu().numpy()
    # ctrl_data = ctrl_data.cpu().numpy()

    # obs, _ = env.reset(options={"init_state": [0.0, 0.0, theta_init, thetadot_init]})
    # frames, states, actions = [], [obs], []
    # switched = False
    # max_steps = 560

    # for _ in range(max_steps):
    #     action, switched = swingup_lqr_controller(obs, switched, masscart, masspole, length)
    #     obs, reward, done, truncated, _, applied_action = env.step(action)
    #     frame = env.render()
    #     frames.append(frame)
    #     states.append(obs)
    #     actions.append(applied_action)
    #     if done or truncated:
    #         break

    # frames = []

    # for i, s in enumerate(state_data):
    #     # env.state = np.array(s, dtype=np.float32)
    #     env.state = s
    #     frame = env.render()
    #     frames.append(frame)


    # env.close()

    # Final state
    # states = np.array(states)
    # actions = np.array(actions).squeeze()

    # states = np.array(state_data)
    # actions = np.array(ctrl_data).squeeze()

    states = state_data.cpu() if isinstance(state_data, torch.Tensor) else state_data
    actions = ctrl_data.cpu() if isinstance(ctrl_data, torch.Tensor) else ctrl_data

    # states = torch.tensor(state_data, dtype=torch.float32)
    # actions = torch.tensor(ctrl_data, dtype=torch.float32)
    # final_theta = states[-1, 2]
    # final_theta_dot = states[-1, 3]
    # stabilized = abs((final_theta + np.pi) % (2*np.pi) - np.pi) < 0.2 and abs(final_theta_dot) < 0.5
    # theta_wrapped = (final_theta + np.pi) % (2 * np.pi) - np.pi
    # stabilized = abs(theta_wrapped) < 0.2 and abs(final_theta_dot) < 0.5
    # Output path and naming
    if context is None:
        folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}"
    else:
        folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_context_{context}"
    path = os.path.join(save_dir, f"run_{run_idx:03}_{folder_name}")
    os.makedirs(path, exist_ok=True)

    # Save video
    # imageio.mimsave(os.path.join(path, 'cartpole.mp4'), frames, fps=50)

    # Save control plot
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(actions)), actions, label='Control Actions', color='blue', s=10)
    plt.title('Control Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(path, 'controls.png'))
    plt.close()

    # Save control actions to CSV
    import csv
    csv_path = os.path.join(path, 'controls.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time_step', 'control_action'])
        for t, a in enumerate(actions):
            writer.writerow([t, float(a)])

    # Save state plots
    plt.figure(figsize=(15, 10))
    labels = ['Cart Position (x)', 'Cart Velocity (x_dot)', 'Pole Angle (theta)', 'Pole Angular Velocity (theta_dot)']
    colors = ['red', 'green', 'orange', 'purple']
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.scatter(range(len(states)), states[:, i], label=labels[i], color=colors[i], s=10)
        plt.title(labels[i] + ' Over Time')
        plt.xlabel('Time Step')
        plt.ylabel(labels[i].split('(')[-1].rstrip(')'))
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'states.png'))
    plt.close()

    csv_path = os.path.join(path, 'states.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time_step', 'x', 'x_dot', 'theta', 'theta_dot'])
        for t, s in enumerate(states):
            writer.writerow([t] + list(map(float, s)))

    # Generate video creation script
    video_script_path = os.path.join(path, 'generate_video.py')
    with open(video_script_path, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
Video generation script for cartpole simulation
Run this script to generate the video from saved state data
"""
import os
import imageio
import numpy as np
import pandas as pd
from gym_continuous_cartpole import ContinuousCartPoleEnv

# Parameters
masscart = {masscart}
masspole = {masspole}
length = {length}

# Load state data
states_csv = os.path.join(os.path.dirname(__file__), 'states.csv')
df = pd.read_csv(states_csv)
states = df[['x', 'x_dot', 'theta', 'theta_dot']].values

# Create environment
env = ContinuousCartPoleEnv(
    masscart=masscart,
    masspole=masspole,
    length=length,
    render_mode="rgb_array"
)

# Generate frames
frames = []
for state in states:
    env.state = np.array(state, dtype=np.float32)
    frame = env.render()
    frames.append(frame)

env.close()

# Save video
output_path = os.path.join(os.path.dirname(__file__), 'cartpole.mp4')
imageio.mimsave(output_path, frames, fps=50)
print(f"Video saved to {{output_path}}")
''')
    
    # Make script executable
    os.chmod(video_script_path, 0o755)

    return {
        "path": path,
        "masscart": masscart,
        "masspole": masspole,
        "length": length,
        # "theta_init": theta_init,
        # "thetadot_init": thetadot_init,
        # "stabilized": stabilized,
        "final_state": states[-1]
    }

def load_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    results = []
    base_save_dir = os.path.join(os.getcwd(), 'videos', 'cartpole_inference_gym_runs')
    model_run_id = "ebonye_models" # "b3725997-9aee-4578-b668-d33e7cb29c4e" #"2ca9672c-582e-43ef-85cf-8550f325947a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"cf756e46-3ddb-4df7-9a13-ba850f495257" #"5b73d6c9-b526-4bfe-bab3-005f5369cf5a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"f8211c69-7ae8-47b3-9bd2-e4f0edda1e82"#"de00c432-078f-44d1-8cc9-e6ab0dfdca88" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"c3af70ca-3733-4cec-a876-95db9bc9a593" #"32ea0675-5539-4d02-80fb-7bfe1f4c263e" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"57f687d9-9e41-48f5-83d4-559592ca762b" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"be268b82-d25e-4026-b303-91ba3c6d1e9f" #"eb14c6a6-d8eb-4b1c-8f37-a57c04a2d66b" #"a49e6137-5856-47ca-86d8-1d37f2f11e7b" #"a885c11b-dee1-472b-9577-46c8c03f70b3" #"9329819d-2d02-4ff0-8741-ed2959c98fdd" #"850d5e45-7a37-4e6d-8499-ac7b11076de3" #"0215ad56-8bf1-47e0-b509-db5af831dabe" #"ee0f5a22-9607-43e5-8cd9-db2f1be66a23"
    steps = [25000, 35000, 45000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000] #55000#274941 #300800
    numberpend = 100 #200 #5
    
    for step in tqdm(steps, desc="Processing steps", unit="step"):
        print(f"\n{'='*80}")
        print(f"Processing step {step}")
        print(f"{'='*80}\n")
        
        save_dir = os.path.join(base_save_dir, model_run_id, f"step_{step}")
        os.makedirs(save_dir, exist_ok=True)
        
        data_path = f'inference_run/mse_control_{step}_{model_run_id}/results_maxcontext50_numpends{numberpend}_indistr_alexcode.pkl'
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"⚠️  WARNING: Data file not found, skipping step {step}")
            print(f"   Expected: {data_path}")
            continue
        
        try:
            cartmasses, polemasses, polelengths, phase_data, controls_data, data_and_controls, pends, _ = load_data(data_path)
        except Exception as e:
            print(f"❌ ERROR loading data for step {step}: {e}")
            print(f"   Skipping step {step}")
            continue

        for cartpole_idx in tqdm(range(len(cartmasses)), desc=f"Step {step} cartpoles", leave=False):
            save_dir_inner = os.path.join(save_dir, f"run_{cartpole_idx:03}")
            masscart = cartmasses[cartpole_idx]
            masspole = polemasses[cartpole_idx]
            length = polelengths[cartpole_idx]

            ground_truth_data_controls = data_and_controls[0][cartpole_idx]
            state_data = ground_truth_data_controls[0]
            ctrl_data = ground_truth_data_controls[1][:, 0]

            run_single_system(masscart, masspole, length, 
                              state_data, ctrl_data, cartpole_idx, save_dir_inner)
            for context in phase_data.keys():
                context_state_data = phase_data[context][cartpole_idx]
                context_ctrl_data = controls_data[context][cartpole_idx][:, 0]
                run_single_system(masscart, masspole, length, 
                                  context_state_data, context_ctrl_data, cartpole_idx, save_dir_inner, context=context)
    
    print("\n" + "="*80)
    print("All steps completed!")
    print("="*80)
