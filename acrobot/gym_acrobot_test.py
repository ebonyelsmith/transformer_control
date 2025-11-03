import os
import gym
import math
import time
import imageio
import numpy as np
import matplotlib.pyplot as plt
from gym_continuous_acrobot import AcrobotEnv

def run_single_system(run_idx, save_dir):
    env = AcrobotEnv(
        render_mode="rgb_array"
    )


    obs, _, d22_bar, h2_bar, phi2_bar, obs_absolute = env.reset()  # state: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1_dot, theta2_dot]
    print(f"Initial observation: {obs}")
    goal_state = np.array([-1.0, 0.0, -1.0, 0.0, 0.0, 0.0])  # Upright position
    frames, states, actions = [], [obs], []
    switched = False
    max_steps = 700
    kp = 16
    kd = 4.5
    alpha = 0.8

    for _ in range(max_steps):
        # action, switched = swingup_lqr_controller(obs, switched, masscart, masspole, length)
        # action = np.random.uniform(-env.max_torque, env.max_torque, size=(1,))

        # q2d = 2 * alpha / math.pi * math.atan(obs[2])
        # q2d = 2 * alpha / math.pi * math.atan(obs[4])   # desired angle based on theta1_dot
        q2d = 2 * alpha / math.pi * math.atan(obs_absolute[2])  # desired angle based on theta1_dot
        # action = kp * (q2d - obs[1]) - kd * obs[3]
        # v2 = kp * (q2d - (math.atan2(obs[3], obs[2]))) - kd * obs[5]
        v2 = kp * (q2d - obs_absolute[1]) - kd * obs_absolute[3]
        action = d22_bar * v2 + h2_bar + phi2_bar

        # import pdb; pdb.set_trace()

        obs, reward, done, truncated, d22_bar, h2_bar, phi2_bar, obs_absolute = env.step(action)
        # add a small perturbation to the state
        # noise = np.random.normal(0, 0.03, size=obs.shape)
        # obs += noise
        frame = env.render()
        frames.append(frame)
        states.append(obs)
        actions.append(action)
        # if done or truncated:
        #     break

    env.close()
    print(f"Final observation: {obs}")


    # Final state
    states = np.array(states)
    actions = np.array(actions).squeeze()
    print(f"max action: {np.max(np.abs(actions))}")
    # print(f"states: {states}")
    # final_theta = states[-1, 2]
    # final_theta_dot = states[-1, 3]
    # stabilized = abs((final_theta + np.pi) % (2*np.pi) - np.pi) < 0.2 and abs(final_theta_dot) < 0.5
    # theta_wrapped = (final_theta + np.pi) % (2 * np.pi) - np.pi
    # stabilized = abs(theta_wrapped) < 0.2 and abs(final_theta_dot) < 0.5
    # Output path and naming
    # folder_name = f"mC{masscart:.2f}_mP{masspole:.2f}_L{length:.2f}_th{theta_init:.2f}_dth{thetadot_init:.2f}"
    folder_name = f"run_{run_idx:03}"
    path = os.path.join(save_dir, folder_name)
    os.makedirs(path, exist_ok=True)

    # Save video
    imageio.mimsave(os.path.join(path, 'acrobot.mp4'), frames, fps=15)

    # Save control plot
    # plt.figure(figsize=(10, 5))
    # plt.scatter(range(len(actions)), actions, label='Control Actions', color='blue', s=10)
    # plt.title('Control Actions Over Time')
    # plt.xlabel('Time Step')
    # plt.ylabel('Control Action')
    # plt.grid()
    # plt.legend()
    # plt.savefig(os.path.join(path, 'controls.png'))
    # plt.close()

    # # Save state plots
    # plt.figure(figsize=(15, 10))
    # labels = ['Cart Position (x)', 'Cart Velocity (x_dot)', 'Pole Angle (theta)', 'Pole Angular Velocity (theta_dot)']
    # colors = ['red', 'green', 'orange', 'purple']
    # for i in range(4):
    #     plt.subplot(2, 2, i+1)
    #     plt.scatter(range(len(states)), states[:, i], label=labels[i], color=colors[i], s=10)
    #     plt.title(labels[i] + ' Over Time')
    #     plt.xlabel('Time Step')
    #     plt.ylabel(labels[i].split('(')[-1].rstrip(')'))
    #     plt.grid()
    #     plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(path, 'states.png'))
    # plt.close()

    return {
        "path": path,
        # "masscart": masscart,
        # "masspole": masspole,
        # "length": length,
        # "theta_init": theta_init,
        # "thetadot_init": thetadot_init,
        # "stabilized": stabilized,
        # "final_state": states[-1]
    }


if __name__ == "__main__":
    num_runs = 1
    results = []
    save_dir = os.path.join(os.getcwd(), 'videos', 'acrobot_gym_runs')
    os.makedirs(save_dir, exist_ok=True)
    # mass_poles = np.array([0.90, 0.83])
    # length_poles = np.array([1.30, 1.14])
    # theta_inits = np.array([2.39, 1.93])
    # thetadot_inits = np.array([0.3, 0.68])


    for run_idx in range(num_runs):
        # masscart = 2.0
        # masspole = np.random.uniform(0.5, 1.0)
        # length = np.random.uniform(1.0, 1.5)
        # masspole = mass_poles[run_idx]
        # length = length_poles[run_idx]
        # theta_init = np.random.uniform(np.pi - np.pi/2, np.pi + np.pi/2)
        # thetadot_init = np.random.uniform(-1.0, 1.0)
        # theta_init = theta_inits[run_idx]
        # thetadot_init = thetadot_inits[run_idx]

        # result = run_single_system(masscart, masspole, length, theta_init, thetadot_init, run_idx, save_dir)
        result = run_single_system(run_idx, save_dir)
        # results.append(result)
        # print(f"[{run_idx+1}/{num_runs}] Stabilized: {result['stabilized']} — Final State: {result['final_state']}")

    # Summary
    # total_stabilized = sum(r['stabilized'] for r in results)
    # print(f"\n✅ {total_stabilized} / {num_runs} systems stabilized.")




