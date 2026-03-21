import os
import gym
import math
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import solve_continuous_are
from gym_continuous_acrobot import AcrobotEnv
# from gym_acrobot_swingup_lqr import swingup_lqr_controller, swingup_lqr_controller2
from gym_acrobot_swingup_lqr import swingup_lqr_controller3
import itertools
import csv

# def wrap(a):
#     return (a + np.pi) % (2*np.pi) - np.pi

def wrap(x, m=-np.pi, M=np.pi):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x
    

# def run_single_system(run_idx, save_dir):
def run_single_system(LINK_LENGTH_1=1.0, LINK_LENGTH_2=1.0, LINK_MASS_1=1.0, LINK_MASS_2=1.0, LINK_COM_POS_1=0.5, LINK_COM_POS_2=0.5, LINK_MOI1=1.0, LINK_MOI2=1.0):
    # env = AcrobotEnv(
    #     render_mode="rgb_array"
    # )
    env = AcrobotEnv(
        LINK_LENGTH_1=LINK_LENGTH_1,
        LINK_LENGTH_2=LINK_LENGTH_2,
        LINK_MASS_1=LINK_MASS_1,
        LINK_MASS_2=LINK_MASS_2,
        LINK_COM_POS_1=LINK_COM_POS_1,
        LINK_COM_POS_2=LINK_COM_POS_2,
        LINK_MOI1=LINK_MOI1,
        LINK_MOI2=LINK_MOI2, 
        render_mode="rgb_array"
    )


    obs, _, obs_absolute = env.reset()  # state: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    print(f"Initial observation: {obs}")
    print(f"Initial absolute observation: {obs_absolute}")
    # goal_state = np.array([-1.0, 0.0, -1.0, 0.0, 0.0, 0.0])  # Upright position
    # goal_state_absolute = np.array([np.pi, 0, 0, 0])
    frames, states, actions, modes = [], [obs_absolute], [], []
    max_steps = 800 #3500
    # kp = 17  # 16
    # kd = 4.5 # 4.5
    # alpha = 0.8 # 0.8
    mode = 0

    for _ in range(max_steps): 
        
        # action, mode = swingup_lqr_controller(obs_absolute, LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2, LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2)
        # action, mode = swingup_lqr_controller2(obs_absolute, LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2, LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2, mode)
        action, mode = swingup_lqr_controller3(obs_absolute, LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2, LINK_COM_POS_1, LINK_COM_POS_2, LINK_MOI1, LINK_MOI2, mode)
        obs, reward, done, truncated, obs_absolute, torque = env.step(action)
        if mode == 1:
            noise_absolute = np.random.normal(0, 1, size=obs_absolute.shape) * 0.005
            obs_absolute = obs_absolute + noise_absolute
            obs_absolute[0] = wrap(obs_absolute[0])
            obs_absolute[1] = wrap(obs_absolute[1])
            obs = np.array([np.cos(obs_absolute[0]), np.sin(obs_absolute[0]), np.cos(obs_absolute[1]), np.sin(obs_absolute[1]), obs_absolute[2], obs_absolute[3]])

        frame = env.render()
        frames.append(frame)
        states.append(obs_absolute)
        actions.append(torque)
        modes.append(mode)
        # if done or truncated:
        #     break

    # print(f"A: {A}")
    # print(f"B: {B}")
    env.close()
    print(f"Final observation: {obs}")
    print(f"Final absolute observation: {obs_absolute}")


    # Final state
    print(f"len states: {len(states)}")
    states = np.array(states)
    actions = np.array(actions).squeeze()
    modes = np.array(modes).squeeze()
    print(f"max action: {np.max(np.abs(actions))}")
    # print(f"states: {states}")
    final_theta1 = states[-1, 0]
    final_theta2 = states[-1, 1]

    # stabilized = (abs(wrap(final_theta1)  - np.pi) < 0.2 and abs(wrap(final_theta2)) < 0.2 and abs(states[-1, 2]) < 0.5 and abs(states[-1, 3]) < 0.5)
    stabilized = (abs(np.cos(final_theta1) + 1) < 0.3 and abs(np.sin(final_theta1)) < 0.3 and abs(np.cos(final_theta2) - 1) < 0.3 and abs(np.sin(final_theta2)) < 0.3) #and abs(states[-1, 2]) < 0.5 and abs(states[-1, 3]) < 0.5)
    # final_theta = states[-1, 2]
    # final_theta_dot = states[-1, 3]
    # stabilized = abs((final_theta + np.pi) % (2*np.pi) - np.pi) < 0.2 and abs(final_theta_dot) < 0.5
    # theta_wrapped = (final_theta + np.pi) % (2 * np.pi) - np.pi
    # stabilized = abs(theta_wrapped) < 0.2 and abs(final_theta_dot) < 0.5
    # Output path and naming
    # folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_th{theta_init:.2f}_dth{thetadot_init:.2f}"
    # folder_name = f"run_{run_idx:03}"
    folder_name = f"run_{run_idx:03}_L1_{LINK_LENGTH_1:.2f}_L2_{LINK_LENGTH_2:.2f}_M1_{LINK_MASS_1:.2f}_M2_{LINK_MASS_2:.2f}"
    path = os.path.join(save_dir, folder_name)
    os.makedirs(path, exist_ok=True)

    # Save video
    imageio.mimsave(os.path.join(path, 'acrobot.mp4'), frames, fps=120)

    # Save control plot
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(actions)), actions, label='Control Actions', color='blue', s=10)
    plt.title('Control Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Control Action')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.scatter(range(len(modes)), modes, label='Controller Mode (0=Energy, 1=LQR)', color='orange', s=10)
    plt.title('Controller Mode Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Controller Mode')
    plt.yticks([0, 1], ['Energy', 'LQR'])
    plt.grid()
    # plt.legend()
    plt.savefig(os.path.join(path, 'controls.png'))
    plt.close()

    # Save state plots
    plt.figure(figsize=(15, 10))
    labels = ['Theta1 (rad)', 'Theta2 (rad)', 'Theta1 Dot (rad/s)', 'Theta2 Dot (rad/s)']
    # labels = ['Cart Position (x)', 'Cart Velocity (x_dot)', 'Pole Angle (theta)', 'Pole Angular Velocity (theta_dot)']
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

    return {
        "path": path,
        # "masscart": masscart,
        # "masspole": masspole,
        # "length": length,
        # "theta_init": theta_init,
        # "thetadot_init": thetadot_init,
        "stabilized": stabilized,
        # "final_state": states[-1]
    }


if __name__ == "__main__":
    num_runs = 1
    results = []
    save_dir = os.path.join(os.getcwd(), 'videos', 'acrobot_gym_runs_14_gptctrlr')
    os.makedirs(save_dir, exist_ok=True)
    # mass_poles = np.array([0.90, 0.83])
    # length_poles = np.array([1.30, 1.14])
    # theta_inits = np.array([2.39, 1.93])
    # thetadot_inits = np.array([0.3, 0.68])

    # mass1 = [1.5, 2.0]
    # mass2 = [1.0, 1.5]
    # # length1 = [1.0, 1.5]
    # length1 = [0.5, 1.0]
    # length2 = [1.5, 2.0]

    # all_combos = itertools.product(mass1, mass2, length1, length2)
    # valid_combos = [(m1, m2, l1, l2) for (m1, m2, l1, l2) in all_combos if m1 >= m2]
    
    # num_runs = len(valid_combos)

    # num_runs = 1

    for run_idx in range(num_runs):
    # for run_idx, (mass1, mass2, length1, length2) in enumerate(valid_combos):

        # LINK_LENGTH_1 = np.random.uniform(0.5, 1)
        # LINK_LENGTH_2 = np.random.uniform(1, 1.5)
        # LINK_MASS_1 = np.random.uniform(0.5, 1)
        # LINK_MASS_2 = np.random.uniform(1, 1.5)

        LINK_LENGTH_1 = np.random.uniform(0.5, 1.0)
        # LINK_LENGTH_1 = np.random.uniform(0.85, 1.35)
        LINK_LENGTH_2 = np.random.uniform(1.0, 1.5)
        LINK_MASS_1 = np.random.uniform(1.5, 2.0)
        LINK_MASS_2 = np.random.uniform(1.0, 1.5)

        # LINK_LENGTH_1 = length1
        # LINK_LENGTH_2 = length2
        # LINK_MASS_1 = mass1
        # LINK_MASS_2 = mass2

        # LINK_LENGTH_1 = 1.0
        # LINK_LENGTH_2 = 2.0
        # LINK_MASS_1 = 1.0
        # LINK_MASS_2 = 1.0
        LINK_COM_POS_1 = LINK_LENGTH_1 / 2
        LINK_COM_POS_2 = LINK_LENGTH_2 / 2
        LINK_MOI1 = LINK_MASS_1 * LINK_LENGTH_1**2 / 12
        LINK_MOI2 = LINK_MASS_2 * LINK_LENGTH_2**2 / 12
        # LINK_MOI1 = 1.0
        # LINK_MOI2 = 1.0
        print(f"Run {run_idx+1}/{num_runs}: LINK_LENGTH_1={LINK_LENGTH_1}, LINK_LENGTH_2={LINK_LENGTH_2}, LINK_MASS_1={LINK_MASS_1}, LINK_MASS_2={LINK_MASS_2}, LINK_COM_POS_1={LINK_COM_POS_1}, LINK_COM_POS_2={LINK_COM_POS_2}, LINK_MOI1={LINK_MOI1}, LINK_MOI2={LINK_MOI2}")
        result = run_single_system(
            LINK_LENGTH_1=LINK_LENGTH_1,
            LINK_LENGTH_2=LINK_LENGTH_2,
            LINK_MASS_1=LINK_MASS_1,
            LINK_MASS_2=LINK_MASS_2,
            LINK_COM_POS_1=LINK_COM_POS_1,
            LINK_COM_POS_2=LINK_COM_POS_2,
            LINK_MOI1=LINK_MOI1,
            LINK_MOI2=LINK_MOI2
        )
        # results.append(result)
        results.append(((LINK_LENGTH_1, LINK_LENGTH_2, LINK_MASS_1, LINK_MASS_2), result['stabilized']))
        print(f"[{run_idx+1}/{num_runs}] Stabilized: {result['stabilized']} ")
        
        # Write CSV
        with open(os.path.join(save_dir, "swingup_batch_results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            header = ["LINK_LENGTH_1", "LINK_LENGTH_2", "LINK_MASS_1", "LINK_MASS_2", "STABILIZED"]
            writer.writerow(header)
            for (l1, l2, m1, m2), stabilized in results:
                writer.writerow([l1, l2, m1, m2, stabilized])
    # Summary
    # total_stabilized = sum(r['stabilized'] for r in results)
    total_stabilized = sum(1 for _, stabilized in results if stabilized)
    print(f"\n✅ {total_stabilized} / {num_runs} systems stabilized.")




