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
# from gym_continuous_cartpole import ContinuousCartPoleEnv
# from gym_cartpole_swingup_lqr import swingup_lqr_controller
from gym_continuous_acrobot import AcrobotEnv

# STEPS = [5000, 10000, 15000, 20000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000, 125000, 130000, 135000, 140000, 150000, 155000, 160000, 170000, 175000, 180000, 185000, 190000, 195000, 200000, 205000, 210000, 215000, 220000, 225000, 230000, 235000, 240000, 245000, 250000, 255000, 260000, 265000, 270000, 275000, 280000, 284408]
STEPS = [75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000, 125000, 130000, 135000, 140000, 150000, 155000, 160000, 170000, 175000, 180000, 185000, 190000, 195000, 200000, 205000, 210000, 215000, 220000, 225000, 230000, 235000, 240000, 245000, 250000, 255000, 260000, 265000, 270000, 275000, 280000, 284408]
# def run_single_system(masscart, masspole, length, state_data, ctrl_data, run_idx, save_dir, context=None):
def run_single_system(link_length1, link_length2, link_mass1, link_mass2, state_data, ctrl_data, run_idx, save_dir, context=None):
    # link_com_pos_1 = 0.5 * link_length1
    # link_com_pos_2 = 0.5 * link_length2
    # link_moi_1 = (1/12) * link_mass1 * link_length1 **2
    # link_moi_2 = (1/12) * link_mass2 * link_length2 **2
    # env = AcrobotEnv(
    #     LINK_LENGTH_1=link_length1,
    #     LINK_LENGTH_2=link_length2,
    #     LINK_MASS_1=link_mass1,
    #     LINK_MASS_2=link_mass2,
    #     LINK_COM_POS_1=link_com_pos_1,
    #     LINK_COM_POS_2=link_com_pos_2,
    #     LINK_MOI1=link_moi_1,
    #     LINK_MOI2=link_moi_2,
    #     render_mode='rgb_array'
    # )


    # frames = []

    # for i, s in enumerate(state_data):
    #     # env.state = np.array(s, dtype=np.float32)
    #     s = s.cpu().numpy() if isinstance(s, torch.Tensor) else np.array(s, dtype=np.float32)
    #     env.state = s
    #     frame = env.render()
    #     frames.append(frame)


    # env.close()

    

    states = state_data.cpu() if isinstance(state_data, torch.Tensor) else state_data
    actions = ctrl_data.cpu() if isinstance(ctrl_data, torch.Tensor) else ctrl_data

    # print(f"actions shape: {actions.shape}, states shape: {states.shape}")
    # import pdb; pdb.set_trace()
    
    if context is None:
        # folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}"
        folder_name = f"l1_{link_length1:.2f}_l2_{link_length2:.2f}_m1_{link_mass1:.2f}_m2_{link_mass2:.2f}"
    else:
        # folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_context_{context}"
        folder_name = f"l1_{link_length1:.2f}_l2_{link_length2:.2f}_m1_{link_mass1:.2f}_m2_{link_mass2:.2f}_context_{context}"
    path = os.path.join(save_dir, f"run_{run_idx:03}_{folder_name}")
    os.makedirs(path, exist_ok=True)

    # Save video
    # imageio.mimsave(os.path.join(path, 'acrobot.mp4'), frames, fps=70)

    # Save control plot
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(actions[:, 0])), actions[:, 0], label='Control Actions', color='blue', s=10)
    plt.title('Control Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(range(len(actions[:, 1])), actions[:, 1], label='Control Label', color='orange', s=10)
    plt.title('Control Labels Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Label')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'controls.png'))
    plt.close()



    # Save state plots
    plt.figure(figsize=(15, 10))
    # labels = ['Cart Position (x)', 'Cart Velocity (x_dot)', 'Pole Angle (theta)', 'Pole Angular Velocity (theta_dot)']
    labels = ['Theta 1 (theta1)', 'Theta 2 (theta2)', 'Theta Dot 1 (dtheta1)', 'Theta Dot 2 (dtheta2)']
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

    # Save states to CSV
    import csv
    csv_path = os.path.join(path, 'states.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time_step', 'theta1', 'theta2', 'dtheta1', 'dtheta2'])
        for t, s in enumerate(states):
            writer.writerow([t] + list(map(float, s)))

    # Generate video creation script
    video_script_path = os.path.join(path, 'generate_video.py')
    with open(video_script_path, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
"""
Video generation script for acrobot simulation
Run this script to generate the video from saved state data
"""
import os
import imageio
import numpy as np
import pandas as pd
from gym_continuous_acrobot import AcrobotEnv

# Parameters
link_length1 = {link_length1}
link_length2 = {link_length2}
link_mass1 = {link_mass1}
link_mass2 = {link_mass2}
link_com_pos_1 = 0.5 * link_length1
link_com_pos_2 = 0.5 * link_length2
link_moi_1 = (1/12) * link_mass1 * link_length1 **2
link_moi_2 = (1/12) * link_mass2 * link_length2 **2

# Load state data
states_csv = os.path.join(os.path.dirname(__file__), 'states.csv')
df = pd.read_csv(states_csv)
states = df[['theta1', 'theta2', 'dtheta1', 'dtheta2']].values

# Create environment
env = AcrobotEnv(
    LINK_LENGTH_1=link_length1,
    LINK_LENGTH_2=link_length2,
    LINK_MASS_1=link_mass1,
    LINK_MASS_2=link_mass2,
    LINK_COM_POS_1=link_com_pos_1,
    LINK_COM_POS_2=link_com_pos_2,
    LINK_MOI1=link_moi_1,
    LINK_MOI2=link_moi_2,
    render_mode='rgb_array'
)

# Generate frames
frames = []
for state in states:
    env.state = np.array(state, dtype=np.float32)
    frame = env.render()
    frames.append(frame)

env.close()

# Save video
output_path = os.path.join(os.path.dirname(__file__), 'acrobot.mp4')
imageio.mimsave(output_path, frames, fps=70)
print(f"Video saved to {{output_path}}")
''')
    
    # Make script executable
    os.chmod(video_script_path, 0o755)

    return {
        "path": path,
        # "masscart": masscart,
        # "masspole": masspole,
        # "length": length,
        "link_length1": link_length1,
        "link_length2": link_length2,
        "link_mass1": link_mass1,
        "link_mass2": link_mass2,
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
    base_save_dir = os.path.join(os.getcwd(), 'videos', 'acrobot_inference_gym_runs')
    model_run_id =  "15bf641c-dbc0-4f2f-b62f-fe04f568aacb" #"c953cb49-31b2-4829-8d1e-d9e2b1c99dce" #"056764e2-f56a-4e25-8019-3ce5098c388c" #"b3725997-9aee-4578-b668-d33e7cb29c4e" #"2ca9672c-582e-43ef-85cf-8550f325947a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"cf756e46-3ddb-4df7-9a13-ba850f495257" #"5b73d6c9-b526-4bfe-bab3-005f5369cf5a" #"a1d5f223-6768-4134-934b-4879031f7ea1" #"f8211c69-7ae8-47b3-9bd2-e4f0edda1e82"#"de00c432-078f-44d1-8cc9-e6ab0dfdca88" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"c3af70ca-3733-4cec-a876-95db9bc9a593" #"32ea0675-5539-4d02-80fb-7bfe1f4c263e" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"57f687d9-9e41-48f5-83d4-559592ca762b" #"457c45df-8c2f-4ac1-9b4e-e77eeed90f3a" #"be268b82-d25e-4026-b303-91ba3c6d1e9f" #"eb14c6a6-d8eb-4b1c-8f37-a57c04a2d66b" #"a49e6137-5856-47ca-86d8-1d37f2f11e7b" #"a885c11b-dee1-472b-9577-46c8c03f70b3" #"9329819d-2d02-4ff0-8741-ed2959c98fdd" #"850d5e45-7a37-4e6d-8499-ac7b11076de3" #"0215ad56-8bf1-47e0-b509-db5af831dabe" #"ee0f5a22-9607-43e5-8cd9-db2f1be66a23"
    numberpend = 100 #200 #5
    
    for step in tqdm(STEPS, desc="Processing steps", unit="step"):
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
            # cartmasses, polemasses, polelengths, phase_data, controls_data, data_and_controls, pends = load_data(data_path)
            link_lengths1, link_lengths2, link_masses1, link_masses2, phase_data, controls_data, data_and_controls, pends, _ = load_data(data_path)
        except Exception as e:
            print(f"❌ ERROR loading data for step {step}: {e}")
            print(f"   Skipping step {step}")
            continue

        # for cartpole_idx in range(len(cartmasses)):
        for acrobot_idx in tqdm(range(len(link_lengths1)), desc=f"Step {step} acrobots", leave=False):
            # save_dir_inner = os.path.join(save_dir, f"run_{cartpole_idx:03}")
            save_dir_inner = os.path.join(save_dir, f"run_{acrobot_idx:03}")
            link_length1 = link_lengths1[acrobot_idx]
            link_length2 = link_lengths2[acrobot_idx]
            link_mass1 = link_masses1[acrobot_idx]
            link_mass2 = link_masses2[acrobot_idx]

            ground_truth_data_controls = data_and_controls[0][acrobot_idx]
            state_data = ground_truth_data_controls[0]
            ctrl_data = ground_truth_data_controls[1]

            run_single_system(link_mass1, link_mass2, link_length1, link_length2,
                              state_data, ctrl_data, acrobot_idx, save_dir_inner)
            for context in phase_data.keys():
                context_state_data = phase_data[context][acrobot_idx]
                context_ctrl_data = controls_data[context][acrobot_idx]
                run_single_system(link_mass1, link_mass2, link_length1, link_length2,
                                  context_state_data, context_ctrl_data, acrobot_idx, save_dir_inner, context=context)
    
    print("\n" + "="*80)
    print("All steps completed!")
    print("="*80)
