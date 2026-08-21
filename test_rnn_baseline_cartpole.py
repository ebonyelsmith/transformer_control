import os
import numpy as np
import torch
import matplotlib.pyplot as plt
# import workCon
# import workCon_linearsys as workCon
# from eval import get_model_from_run
from tqdm import tqdm
import ipdb
import traceback
import random
import math
from scipy.integrate import solve_ivp
import pickle
import re

# asadas
from models_cartpole import RNNModel
from munch import Munch
import yaml
import importlib.util


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

plot_label = 'rnn_zerodyn' #'mse_control'
phase_plot_label = 'mse_control_phaseplot'
mse_plot_label = 'mse_control_mseplot'
save_results = "trainsteps_test_mse_control.txt"
save_phase_plot = "trainsteps_test_mse_control.txt"
log_info = "trainsteps_log_mse_control.txt"
model_name= "cartpole_rnn"
model_run_id= "bf5733fe-98c0-4c2d-aa18-2701b723de57" #"three_layer_lstm_v4_h1024_e256_dropout0.2" 

model_config = {'n_dims': {'state': 5, 'distance': 1, 'mode': 3,'control': 1},
                'hidden_size': 1024,
                'num_layers': 3,
                'cell_type': 'lstm',
                'n_embd': 256}

model = RNNModel(n_dims=model_config['n_dims'], 
                    hidden_size=model_config['hidden_size'],
                    num_layers=model_config['num_layers'], 
                    cell_type=model_config['cell_type'],
                    n_embd=model_config['n_embd'])

model_checkpoint_epoch = 1 #25 #59 #125 #14 #38 #60 #125
mode = 'ood' # 'indistr', 'ood', 'train' 
    

total_time = 14 #4 #5 #1.5
dt = 0.025
# Num_of_context = 30 #50 #150
Num_of_pendulums = 100 #5 #100 #200 #1 #10 #20 #40 #10
# start_index_num = [Num_of_context]




random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1000)

def mse(xs_pred, xs_true, device):
    """
    Calculates the Mean Squared Error (MSE) between the predicted and true states.
    """
    xs_pred = xs_pred.to(device)
    xs_true = xs_true.to(device)
    return (xs_true - xs_pred).pow(2).mean().item()

def mse_controls(control_values_model, control_values_rk4, device):
    """
    Calculates the Mean Absolute Error (MAE) between the predicted and true control values.
    """
    control_pred = control_values_model.to(device)
    control_true = control_values_rk4.to(device)
    # return (control_true - control_pred).pow(2).mean().item()
    #mean absolute error instead
    return torch.abs(control_true - control_pred).mean().item()

def get_model_from_run(model, run_path, epoch, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf
    
    print(f"Loading model from {run_path}, epoch {epoch}, step {step}")

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path, map_location="cpu")  
        state_dict = state["model_state_dict"]
    else:
        model_path = os.path.join(run_path, f"checkpoint_epoch{epoch}_step{step}.pt") 
        state = torch.load(model_path, map_location="cpu") 
        state_dict = state["model_state_dict"]

    # consume_prefix_in_state_dict_if_present(checkpoint['model_state_dict'], prefix='module.')
    # model.load_state_dict(checkpoint['model_state_dict'])
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    return model, conf

def load_model(model, run_dir, name, run_id, step, epoch):
    """
    Loads a pre-trained model and its configuration from a specified run directory.
    """
    # e.g. load_model(./models, cartpole_rnn, two_layer_lstm_v3_h1024_e256_dropout, 1000, 1)
    # -> loads checpoint_epoch1_step1000.pt
    # e.g. load_model(./models, cartpole_rnn, two_layer_lstm_v3_h1024_e256_dropout, -1, 1)
    # -> loads state.pt
    run_path = os.path.join(run_dir, name, run_id)
    model, conf = get_model_from_run(model, run_path, epoch= epoch, step=step)
    return model, conf

def cartpole_dynamics(state, u, cartmass, polemass, polelength, g=9.81):
    """
    Computes the dynamics of the cartpole system.
    Inputs:
        state: (4,) tensor representing [x, x_dot, theta, theta_dot]
        u: (1,) tensor representing the control input (force applied to the cart)
        cartmass: float, mass of the cart
        polemass: float, mass of the pole
        polelength: float, length of the pole
        g: float, acceleration due to gravity (default 9.81 m/s^2)
    Returns:
        dydt: (4,) tensor representing the derivatives [x_dot, xacc, theta_dot, thetaacc]
    """
    x, x_dot, theta, theta_dot = state
    force = u.item()  

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (force + sintheta * polelength * polemass * theta_dot**2) / (cartmass + polemass)
    thetaacc = (g * sintheta - costheta * temp) / (polelength * (4.0 / 3.0 - polemass * costheta**2 / (cartmass + polemass)))
    xacc = temp - (polemass * polelength * thetaacc * costheta) / (cartmass + polemass)

    return torch.tensor([x_dot, xacc, theta_dot, thetaacc], dtype=torch.float32).to(state.device)

def rk4_step(state, u, dt, dynamics, cartmass, polemass, polelength):
    """
    Perform a single RK4 step.
    Inputs:
        state: (batch, 4)
        u: (batch,)
        dt: float
        dynamics: function that takes (state, u) and returns derivative
    Returns:
        state_next: (batch, 4)
    """
    k1 = dynamics(state, u, cartmass, polemass, polelength)
    k2 = dynamics(state + 0.5 * dt * k1, u, cartmass, polemass, polelength)
    k3 = dynamics(state + 0.5 * dt * k2, u, cartmass, polemass, polelength)
    k4 = dynamics(state + dt * k3, u, cartmass, polemass, polelength)
    state_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return state_next

def run_inference_on_model(model, XData, YS, total_time, device, dt=0.01, context=1, start_index=1, cartmass=1.0, polemass=0.1, polelength=0.5):
    """
    Runs inference on a trained model to simulate the dynamics of a pendulum system over time, given an initial state and context data.

    Args:
        model (torch.nn.Module): The trained model used for predicting control inputs.
        XData (torch.Tensor): The input state  (e.g., [theta, thetadot])
        YS (torch.Tensor): The ground truth control input data
        total_time (float): Total duration of the simulation in seconds.
        dt (float, optional): Time step for the simulation. Defaults to 0.01.
        context (int, optional): The number of previous time steps used as context for the model. Defaults to 1.
        start_index (int, optional): The starting index for inference. Must be at least equal to `context`. Defaults to 1.
        device (torch.device): The device on which the model and data are located (e.g., 'cpu' or 'cuda').
        cartmass (float, optional): Mass of the cart. Defaults to 1.0.
        polemass (float, optional): Mass of the pendulum. Defaults to 0.1.
        polelength (float, optional): Length of the pendulum. Defaults to 0.5.
    Returns:
        tuple: A tuple containing:
            - T (np.ndarray): Array of time steps during the simulation.
            - theta_model (np.ndarray): Array of predicted theta over time.
            - thetadot_model (np.ndarray): Array of predicted thetadot over time.
            - YS_context (np.ndarray): Array of control inputs over time.

    Raises:
        AssertionError: If `start_index` is less than `context`.
    """
    assert start_index >= context, "start_index must be at least equal to context"
    model = model.to(device)
    model.eval()

    """
    XData : (x1, x.1, t1, t.1), ..., (x560, x.560, t560, t.560) [560, 4]
    YS : (u1, m1), ..., (u559, m559), [559, 2]
    """

    cartmass = torch.tensor(cartmass).to(device) 
    polemass = torch.tensor(polemass).to(device) 
    polelength = torch.tensor(polelength).to(device) 
    
    # (0, dt, 2dt, 3dt, ..., (total_time // dt) dt)
    T = np.arange(0, total_time, dt)
    n_steps = len(T)

    XData_context = XData[:context].to(device) 
    YS_context = YS[:context-1].to(device)
    # XData_context : (x1, x.1, t1, t.1), ..., (xk, x.k, tk, t.k)
    # YS_context : (u1, m1), ..., (uk-1, mk-1) or empty if k = 1

    XData_final = XData_context.clone()
    YS_final = YS_context.clone() 

    states_scale = [7.0, 8.0, 1.0, 1.0, 5.0] 
    control_scale = 15.0  

    for i in range(start_index, n_steps):
        with torch.no_grad():
            # Zero-pad for inference 
            # --- model expects (s1, ..., sk), (m1, ..., mk), (u1, ..., uk)
            # --- no harm in doing this. hidden state for sk conditioned on (s1, m1, u1, ..., sk-1, mk-1, uk-1)
            YS_context = torch.cat((YS_context, torch.zeros((1, 2), device=YS_context.device)), dim=0)


            xs = XData_context
            ys = YS_context
            # xs : (x1, x.1, t1, t.1), ..., (xk, x.k, tk, t.k)
            # ys : (u1, m1), ..., (uk-1, mk-1), (0, 0)

            # Processing
            cos_theta = torch.cos(xs[:, 2]) 
            sin_theta = torch.sin(xs[:, 2]) 
            xs = torch.cat((xs[:, :2], cos_theta.unsqueeze(1), sin_theta.unsqueeze(1), xs[:, 3:]), dim=-1)
            xs = xs / torch.tensor(states_scale, device=device) 
            ys = ys / torch.tensor([control_scale, 1.0], device=device)  

            ys_for_model = ys.clone()
            ys_for_model[:, 1] = ys_for_model[:, 1] + 1.0 

            # xs : (x1, x.1, cos t1, sin t1, t.1), ..., (xk, x.k, cos tk, sin tk, t.k) normalized
            # ys_for_model : (u1, m1), ..., (uk-1, mk-1), (0, 0) normalized and shifted

            

            # Add batch dimension for inference
            s, m, a = xs.unsqueeze(0), ys_for_model[:, 1].unsqueeze(0).unsqueeze(-1), ys_for_model[:, 0].unsqueeze(0).unsqueeze(-1)
            # s : s1, s2, ..., sk-1, sk
            # m : m1, m2, ..., mk-1, 0
            # a : u1, u2, ..., uk-1, 0

            _, flag_logits, u_pred = model(s, m, a, inf=True) 
            u = u_pred.squeeze()
            flag_logits = flag_logits.squeeze(0) 
            flag_pred = torch.argmax(flag_logits) - 1 # -1, 0, 1
            # u : uk (still normalized)
            # m : mk (shifted back)

            if i >= context:
                if flag_pred == -1:
                    actual_mode_to_use = torch.tensor(0, device=device)
                else:
                    actual_mode_to_use = flag_pred
            else:
                actual_mode_to_use = torch.tensor(-1, device=device)

            u_with_label = torch.cat(((u * control_scale).unsqueeze(0), actual_mode_to_use.unsqueeze(0)), dim=0) 
            u_with_label = u_with_label.squeeze(-1) 
            # u_with_label : (uk, mk) , both in the same units / scale as ground truth    
        
        u_unscaled = u * control_scale  
        previous_state_unscaled = XData_context[-1] 
        # u_unscaled : uk (scaled to ground truth scale)
        # previous_state_unscaled : (xk, x.k, tk, t.k)

        next_state = rk4_step(
            previous_state_unscaled,
            u_unscaled,
            dt,
            cartpole_dynamics,
            cartmass=cartmass,
            polemass=polemass,
            polelength=polelength
        )
        # next_state : (xk+1, x.k+1, tk+1, t.k+1) [4]

        new_X = next_state
        new_X = new_X.unsqueeze(0)
        # new_X : [1, 4]

        XData_context = torch.cat((XData_context, new_X), dim=0) 
        XData_final = torch.cat((XData_final, new_X), dim=0) 
        # XData_context : (x1, x.1, t1, t.1), ..., (xk+1, x.k+1, tk+1, t.k+1)
        # XData_final : (x1, x.1, t1, t.1), ..., (xk+1, x.k+1, tk+1, t.k+1)

        # sliding window
        XData_context = XData_context[1:] 
        # XData_context : (x2, x.2, t2, t.2), ..., (xk+1, x.k+1, tk+1, t.k+1)

        new_Y = torch.tensor(u_with_label, dtype=torch.float32, device=device)
        YS_context[-1, :] = new_Y
        YS_final = torch.cat((YS_final, new_Y.unsqueeze(0)), dim=0) 
        # YS_context : (u1, m1), ..., (uk, mk)
        # YS_final : (u1, m1), ..., (uk, mk)

        if context == 1:
            YS_context = YS_context[:0] # set it back to being empty
        else:
            YS_context = YS_context[1:] # otherwise drop the first term
        # YS_context : (u2, m2), ..., (uk, mk) or empty if k = 1

    YS_context = YS_final[1:] 
    x_model = XData_final[:, 0] 
    theta_model = XData_final[:, 2] 
    xdot_model = XData_final[:, 1] 
    thetadot_model = XData_final[:, 3] 
    
    return T, x_model, theta_model, xdot_model, thetadot_model, YS_context




def plot_mse_vs_context_length(mean_mse, std_mse, save_results_path, folder_name, plot_label, loss_type= "state"):
    """
    Plots the mean MSE vs. context length graph.

    Args:
        mean_mse (dict): A dictionary containing the mean MSE values for each context length.
        std_mse (dict): A dictionary containing the standard deviation of MSE values for each context length.
        save_results_path (str): The path to save the results log.
        folder_name (str): The directory where the plot will be saved.
        plot_label (str): The label for the plot.
        loss_type (str): The type of loss used for the MSE calculation. Defaults to "state mse".

    Returns:
        None
    """
    # with open(save_results_path, "a") as f:
    #     f.write(f"mean_mse: {mean_mse}\n")
    #     f.write(f"std_mse: {std_mse}\n")

    x_axis = list(mean_mse.keys())
    y_axis = list(mean_mse.values())
    y_err = list(std_mse.values())

    plt.figure(figsize=(10, 6))
    plt.errorbar(x_axis, y_axis, yerr=y_err, fmt='o-', capsize=8, label=plot_label)
    plt.xticks(x_axis)
    plt.xlabel('Context Length')
    plt.ylabel('Mean MSE')
    plt.title(f'Mean MSE vs Context Length {Num_of_pendulums} Pendulums')
    plt.legend()
    plt.grid(True)


    plot_path = os.path.join(folder_name, f"{loss_type}_mean_mse_vs_context_length({Num_of_pendulums} Pendulums).png")
    plt.savefig(plot_path, dpi=600, bbox_inches='tight')





import os
import numpy as np
import matplotlib.pyplot as plt

def phase_plot(theta_plot, thetadot_plot, theta_rk4_unscaled, save_results_path, folder_name):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import random

    # Create save directory if it doesn't exist
    save_dir = os.path.join(save_results_path, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 7))

    # Plot ground truth
    plt.plot(theta_rk4_unscaled[:-1], np.diff(theta_rk4_unscaled)/0.01, 
             label='Ground Truth', linewidth=3, color='black', alpha=0.8)

    # Plot every 10th sequence from theta_plot and thetadot_plot
    for idx in range(0, len(theta_plot), 10):
        plt.plot(theta_plot[idx], thetadot_plot[idx], linestyle='--', alpha=0.7, label=f'Context {idx+1}')

    # Labels and title
    plt.xlabel("Theta (rad)")
    plt.ylabel("Theta dot (rad/s)")
    plt.title("Phase Plot: Theta vs Theta dot")
    plt.legend(fontsize='small', ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Generate random number to ensure unique filename
    random_index = random.randint(1000, 9999)
    plot_path = os.path.join(save_dir, f"phase_plot_{random_index}.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return plot_path

def main(model,
        chkpt_step, folder_name,
        number_of_context,
        phase_data,
        controls_data,
        data_and_controls,
        ):

    model, _ = load_model(
        model=model,
        run_dir="./models",
        name= model_name,
        run_id= model_run_id,
        step=chkpt_step,
        epoch=model_checkpoint_epoch 
    )

    os.makedirs(folder_name, exist_ok=True)
    save_results_path = os.path.join(folder_name, save_results)
    save_phase_path = os.path.join(folder_name, save_phase_plot)
    log_info_path = os.path.join(folder_name, log_info)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if mode == 'ood':
        base_dir = f"/data/shared/Control_ICL/Dataset_Cartpole_AIGymWithNoise_ICL_new_ranges_zero_dyn/picklefolder_test_outofdistr"
        pickle_file = f"batch_test_0_{number_of_context}.pkl"
    elif mode == 'indistr':
        base_dir = f"/data/shared/Control_ICL/Dataset_Cartpole_AIGymWithNoise_ICL_new_ranges_zero_dyn/picklefolder_test_indistr"
        pickle_file = f"batch_test_0_{number_of_context}.pkl"
    elif mode == 'train':
        base_dir = f"/data/shared/Control_ICL/Dataset_Cartpole_AIGymWithNoise_ICL_new_ranges/picklefolder"
        pickle_file = "batch_0.pkl"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    file_path_test_data = os.path.join(base_dir, pickle_file)
    with open(file_path_test_data, "rb") as f:
        xs, ys, cartmasses, polemasses, polelenghs = pickle.load(f)
        # xs : (x, x., t, t.) length 560
        # ys : (u, m) length 559

    pends = np.arange(Num_of_pendulums) # Grab n trajectories
    cartmasses = [cartmasses[i] for i in pends]
    polemasses = [polemasses[i] for i in pends]
    polelenghs = [polelenghs[i] for i in pends]
    states = [xs[i].cpu().detach().numpy() for i in pends]
    controls = [ys[i].cpu().detach().numpy() for i in pends]
    data_and_controls.append([(xs[i].cpu().numpy(), ys[i].cpu().numpy()) for i in pends])
    # states : list of trajectories (x1, x.1, t1, t.1), ..., (x560, x.560, t560, t.560)
    # controls : list of trajectories (u1, m1), ..., (u559, m559)
    # data_and_controls : list of trajectory tuples ((x, x., t, t.), (u, m))
    
    with open(log_info_path, "w") as file:
        counter = 0
        for cartmass, polemass, polelength in tqdm(zip(cartmasses, polemasses, polelenghs), desc="Cartpole", total=len(cartmasses), leave=False):
            mse_per_context = []
            mse_control_per_context = []
            
            # Grab x, x., t, t., (u, m) values from counter-th trajectory.
            x_rk4 = states[counter][:, 0]
            theta_rk4 = states[counter][:, 2]
            xdot_rk4 = states[counter][:, 1]
            thetadot_rk4 = states[counter][:, 3]
            control_values_rk4 = controls[counter]

            # Stack rk4 data into (560, 4) matrix with row i = (x_rk4[i], xdot_rk4[i], ..., thetadot_rk4[i])
            xs_dataset = np.column_stack((x_rk4, xdot_rk4, theta_rk4, thetadot_rk4))
            xs_dataset = torch.tensor(xs_dataset).float().cuda()
            control_values_rk4 = torch.tensor(control_values_rk4).float().cuda()
            # xs_dataset : (x1, x.1, t1, t.1), ..., (x560, x.560, t560, t.560)
            # control_values_rk4 : (u1, m1), ..., (u559, m559)
            
            x_plot = []
            theta_plot = []
            xdot_plot = []
            thetadot_plot = []

            control_values_scaled = control_values_rk4

            start_index = number_of_context

            T_model, x_model2, theta_model2, xdot_model2, thetadot_model2, controls_model2 = run_inference_on_model(
                model, xs_dataset, control_values_rk4, total_time, device, dt,
                context=number_of_context, start_index=start_index, cartmass=cartmass, polemass=polemass, polelength=polelength
            )   

            # x_model2 : x1, x2, ..., xN+1
            # theta_model2 : t1, t2, ..., tN+1
            # xdot_model2 : x.1, x.2, ..., x.N+1
            # thetadot_model2 : t.1, t.2, ..., t.N+1
            # controls_model2 : (u1, m1), ..., (uN, mN)

            trajectory = torch.stack([x_model2, xdot_model2, theta_model2, thetadot_model2], axis=1) 
            controls_for_trajectory = controls_model2
            phase_data[number_of_context].append(trajectory)
            controls_data[number_of_context].append(controls_for_trajectory)

            x_plot.append(x_model2)
            theta_plot.append(theta_model2)
            xdot_plot.append(xdot_model2)
            thetadot_plot.append(thetadot_model2)
            
            # state_pred_context = torch.stack([x_model2[number_of_context:], xdot_model2[number_of_context:], theta_model2[number_of_context:], thetadot_model2[number_of_context:]], dim=1)  # 7/26/2025
            # state_rk4_context = torch.stack([xs_dataset[number_of_context:, 0], xs_dataset[number_of_context:, 1], xs_dataset[number_of_context:, 2], xs_dataset[number_of_context:, 3]], dim=1)
            # mse_loss = mse(state_pred_context, state_rk4_context, device)
            # print("mse_loss",mse_loss)
            # mse_per_context.append(mse_loss)


            # mse_control_loss = mse_controls(controls_model2[number_of_context-1:], control_values_scaled[number_of_context-1:], device)
            # print("mse_control_loss",mse_control_loss)
            # mse_control_per_context.append(mse_control_loss)
                
            counter += 1
            
        # mse_mean = {context_length: np.mean(mse_results[context_length]) for context_length in contexts}
        # mse_std = {context_length: np.std(mse_results[context_length]) for context_length in contexts}
        # mse_control_mean = {context_length: np.mean(mse_control_results[context_length]) for context_length in contexts}
        # mse_control_std = {context_length: np.std(mse_control_results[context_length]) for context_length in contexts}
        # print(f"Mean MSE: {mse_mean}")
        # print(f"Std MSE: {mse_std}")

    # plot_mse_vs_context_length(mse_mean, mse_std, save_results_path, folder_name, mse_plot_label, loss_type="state")
    # plot_mse_vs_context_length(mse_control_mean, mse_control_std, save_results_path, folder_name, mse_plot_label, loss_type="control")

    # phase_data :  Predicted state trajectories (x1, x.1, t1, t.1), ..., (xN+1, x.N+1, tN+1, t.N+1)
    # controls_data : Predicted control trajectories (u1, m1), ..., (uN, mN)
    # data_and_controls : True trajectory tuples ({(x1, x.1, t1, t.1), ..., (x560, x.560, t560, t.560)}, {(u1, m1), ..., (u559, m559)})
    return cartmasses, polemasses, polelenghs, phase_data, controls_data, data_and_controls, pends

try:
    # model_checkpoint_step_list = [5000, 10000, 12000, 15000, 17000, 20000]
    model_checkpoint_step_list = [225547]
    Num_of_contexts = [1, 5, 10, 25, 50]

    for step in tqdm(model_checkpoint_step_list, desc="Checkpoint Steps"):
        model_checkpoint_step = int(step)
        folder_name = f"inference_run/{plot_label}_{model_checkpoint_step}_{model_run_id}"

        phase_data = {context_length: [] for context_length in Num_of_contexts}
        controls_data = {context_length: [] for context_length in Num_of_contexts}
        data_and_controls = []

        for Num_of_context in Num_of_contexts:
            print(f"Running inference for checkpoint step {model_checkpoint_step} with context length {Num_of_context}...")
            
            results = main(model,
                           model_checkpoint_step,
                            folder_name,
                            Num_of_context,
                            # mse_results,
                            # mse_control_results,
                            phase_data,
                            controls_data,
                            data_and_controls,                          
                           ) 
            _, _, _, phase_data, controls_data, data_and_controls, pends = results

        
        save_results = os.path.join(folder_name, f"results_maxcontext{Num_of_context}_numpends{Num_of_pendulums}_{mode}_alexcode.pkl")

        with open(save_results, "wb") as f:
            pickle.dump(results, f)


    print("done")   
except Exception:
    print("exception starting debugger")
    traceback.print_exc()
    ipdb.post_mortem()