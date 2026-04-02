### set cuda visible device to gpu 2 and 3
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import matplotlib.pyplot as plt
# import workCon
# import workCon_linearsys as workCon
import workCon_acrobot_aigym as workCon
# import os
# rom eval import get_model_from_run
from tqdm import tqdm
import ipdb
import traceback
import random
import math
# import generate_dataset
from scipy.integrate import solve_ivp
import pickle
import re

from models_acrobot_new import RNNModel
from munch import Munch
import yaml
import importlib.util

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


plot_label = 'mse_control2'
phase_plot_label = 'mse_control_phaseplot'
mse_plot_label = 'mse_control_mseplot'
save_results = "trainsteps_test_mse_control.txt"
save_phase_plot = "trainsteps_test_mse_control.txt"
log_info = "trainsteps_log_mse_control.txt"
model_name= "acrobot_rnn"


model_run_id= "lstm_v4_h1024_e256_dropout0.2" #"aa880853-841e-4b61-a7a7-9a3720482be2" #"e6ca8305-a383-4bc2-9f18-bd258dcc0183" #"15bf641c-dbc0-4f2f-b62f-fe04f568aacb" #"ec03ac2f-4708-4295-a44d-c14d439f7335" ### finetuned last 2 layers #"d9d1d44a-9942-40b9-a2d4-bfba4177f2ce" ### finetuned chkpt #"15bf641c-dbc0-4f2f-b62f-fe04f568aacb" #"c953cb49-31b2-4829-8d1e-d9e2b1c99dce" #"056764e2-f56a-4e25-8019-3ce5098c388c"

model_config = {'n_dims': {'state': 6, 'distance': 1, 'mode': 3,'control': 1},
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
mode = 'ood' # 'train', 'ood', 'indistr'

    

total_time = 70 
dt = 0.02
Num_of_pendulums = 100 




random.seed(1000)
np.random.seed(1000)
torch.manual_seed(1000)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1000)

def mse(xs_pred, xs_true, device):
    xs_pred = xs_pred.to(device)
    xs_true = xs_true.to(device)
    return (xs_true - xs_pred).pow(2).mean().item()

def mse_controls(control_values_model, control_values_rk4, device):
    control_pred = control_values_model.to(device)
    control_true = control_values_rk4.to(device)
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
    run_path = os.path.join(run_dir, name, run_id)
    model, conf = get_model_from_run(model, run_path, epoch= epoch, step=step)
    return model, conf


def debug_model(model, XData, YS, device):
    model.eval()
    XData = XData.to(device)
    YS = YS.to(device)
    with torch.no_grad():
        model = model.to(device)
        u_pred, state_pred = model(XData, YS, inf="yes")
    # import pdb; pdb.set_trace()
    return u_pred, state_pred
        
def acrobot_dynamics(state, a, l1, l2, m1, m2, g=9.81):
    """
    Computes the dynamics of the acrobot system.
    Inputs:
        state: (4,) tensor representing [theta1, theta2, dtheta1, dtheta2]
        a: (1,) tensor representing the control input (torque applied to the second joint)
        l1: float, length of the first link
        l2: float, length of the second link
        m1: float, mass of the first link
        m2: float, mass of the second link
        g: float, acceleration due to gravity (default 9.81 m/s^2)
    """
    lc1 = l1 / 2.0
    lc2 = l2 / 2.0
    I1 = (1/12) * m1 * l1**2
    I2 = (1/12) * m2 * l2**2
    cos = torch.cos
    sin = torch.sin
    pi = torch.pi
    theta1, theta2, dtheta1, dtheta2 = state
    d1 = (
        m1 * lc1**2
        + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2))
        + I1
        + I2
    )
    d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
    phi1 = (
        -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
        - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
        + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
        + phi2
    )
        
    ddtheta2 = (
        a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
    ) / (m2 * lc2**2 + I2 - d2**2 / d1)
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    # return dtheta1, dtheta2, ddtheta1, ddtheta2
    return torch.tensor([dtheta1, dtheta2, ddtheta1, ddtheta2], dtype=torch.float32).to(state.device)

def rk4_step(state, u, dt, dynamics, l1, l2, m1, m2):
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
    k1 = dynamics(state, u, l1, l2, m1, m2)
    k2 = dynamics(state + 0.5 * dt * k1, u, l1, l2, m1, m2)
    k3 = dynamics(state + 0.5 * dt * k2, u, l1, l2, m1, m2)
    k4 = dynamics(state + dt * k3, u, l1, l2, m1, m2)
    state_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return state_next





def run_inference_on_model(model, XData, YS, total_time, device, dt=0.01, context=1, start_index=1, l1=1.0, l2=1.0, m1=1.0, m2=1.0):
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

    # XData : (t1_1, t2_1, t1._1, t2._1), ..., (t1_120, t2_120, t1._120, t2._120) [120, 4]
    # YS : (u1, m1), ..., (u120, u120) [120, 2]

    l1 = torch.tensor(l1, dtype=torch.float32, device=device)
    l2 = torch.tensor(l2, dtype=torch.float32, device=device)
    m1 = torch.tensor(m1, dtype=torch.float32, device=device)
    m2 = torch.tensor(m2, dtype=torch.float32, device=device)
    
    # (0, dt, 2dt, 3dt, ..., (total_time // dt) dt)
    T = np.arange(0, total_time, dt)
    n_steps = len(T)

    XData_context = XData[:context].to(device) #6/25/2025
    YS_context = YS[:context-1].to(device)
    # XData_context : (t1_1, t2_1, t1._1, t2._1), ..., (t1_k, t2_k, t1._k, t2._k)
    # YS_context : (u1, m1), ..., (uk-1, mk-1) or empty if k = 1

    XData_final = XData_context.clone()
    YS_final = YS_context.clone()

    states_scale = [1.0, 1.0, 1.0, 1.0, 7.0, 7.0] # for acrobot cos(theta1), sin(theta1), cos(theta2), sin(theta2), theta1dot, theta2dot
    control_scale = 10.0

    for i in range(start_index, n_steps):
        with torch.no_grad():
            # Zero-pad for inference 
            # --- model expects (s1, ..., sk), (m1, ..., mk), (u1, ..., uk)
            # --- no harm in doing this. hidden state for sk conditioned on (s1, m1, u1, ..., sk-1, mk-1, uk-1)
            YS_context = torch.cat((YS_context, torch.zeros((1, 2), device=YS_context.device)), dim=0)

            xs = XData_context
            ys = YS_context
            # xs : (x1, x.1, t1, t.1), ..., (xk, x.k, tk, t.k) [k, 4]
            # ys : (u1, m1), ..., (uk-1, mk-1), (0, 0) [k, 2]

            cos_theta1 = torch.cos(xs[:, 0]) 
            sin_theta1 = torch.sin(xs[:, 0]) 
            cos_theta2 = torch.cos(xs[:, 1]) 
            sin_theta2 = torch.sin(xs[:, 1]) 
            xs = torch.cat((cos_theta1.unsqueeze(1), sin_theta1.unsqueeze(1), cos_theta2.unsqueeze(1), sin_theta2.unsqueeze(1), xs[:, 2:]), dim=-1) #9/16/2025
            xs = xs / torch.tensor(states_scale, device=device)
            ys = ys / torch.tensor([control_scale, 1.0], device=device)

            ys_for_model = ys.clone()
            ys_for_model[:, 1] = ys_for_model[:, 1] + 1.0
            # xs : (cos t1 _1, sin t1 _1, cos t2 _ 1, sin t2 _1, t1._1, t2._2), ..., (cos t1 _k, sin t1 _k, cos t2 _k, sin t2 _k, t1._k, t2._k) [k, 6]
            # ys_for_model : (u1, m1), ..., (uk-1, mk-1), (0, 1) normalized and shifted [k, 2]

            s, m, a = xs.unsqueeze(0), ys_for_model[:, 1].unsqueeze(0).unsqueeze(-1), ys_for_model[:, 0].unsqueeze(0).unsqueeze(-1)
            # s : s1, s2, ..., sk-1, sk [1, k, 6]
            # m : m1, m2, ..., mk-1, 1  [1, k, 1]
            # a : u1, u2, ..., uk-1, 0  [1, k, 1]

            _, flag_logits, u_pred = model(s, m, a, inf=True) 

            u = u_pred.squeeze()
            flag_logits = flag_logits.squeeze(0) 
            flag_pred = torch.argmax(flag_logits) - 1 # -1, 0, 1
            # u : uk (still normalized) []
            # flag_pred : mk (shifted back) []

            if i >= context:
                if flag_pred == -1:
                    actual_mode_to_use = torch.tensor(0, device=device)
                else:
                    actual_mode_to_use = flag_pred
            else:
                actual_mode_to_use = torch.tensor(-1, device=device)

            u_with_label = torch.cat(((u * control_scale).unsqueeze(0), actual_mode_to_use.unsqueeze(0)), dim=0) 
            u_with_label = u_with_label.squeeze(-1)
            # u_with_label : (uk, mk) , both in the same units / scale as ground truth  [2]  

        u_unscaled = u * control_scale  
        previous_state_unscaled = XData_context[-1]
        # u_unscaled : uk (scaled to ground truth scale) []
        # previous_state_unscaled : (xk, x.k, tk, t.k) [4]

        next_state = rk4_step(
            previous_state_unscaled,
            u_unscaled,
            dt,
            acrobot_dynamics,
            l1, l2, m1, m2
        )

        new_X = next_state
        new_X = new_X.unsqueeze(0)
        # new_X : [1, 4]

        XData_context = torch.cat((XData_context, new_X), dim=0) 
        XData_final = torch.cat((XData_final, new_X), dim=0)

        # sliding window
        XData_context = XData_context[1:]
        

        new_Y = torch.tensor(u_with_label, dtype=torch.float32, device=device) # [2]
        YS_context[-1, :] = new_Y
        YS_final = torch.cat((YS_final, new_Y.unsqueeze(0)), dim=0) 

        if context == 1:
            YS_context = YS_context[:0] # set it back to being empty
        else:
            YS_context = YS_context[1:] # otherwise drop the first term
        


    YS_context = YS_final
    theta1_model = XData_final[:, 0] 
    theta2_model = XData_final[:, 1] 
    thetadot1_model = XData_final[:, 2]
    thetadot2_model = XData_final[:, 3]
    
    return T, theta1_model, theta2_model, thetadot1_model, thetadot2_model, YS_context

import os
import numpy as np
import matplotlib.pyplot as plt

def main(model,
        chkpt_step, folder_name,
        number_of_context,
        phase_data,
        controls_data,
        data_and_controls,
        ):
    """_summary_
    """
    model, _ = load_model(
        model,
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

    data_and_controls = []
    
    if mode == 'ood':
        base_dir = f"/data/shared/Control_ICL/Dataset_Acrobot_AIGymWithNoiseShorter_ICL/picklefolder_test_outofdistr"
        pickle_file = f"batch_test_0_{number_of_context}.pkl"
    elif mode == 'indistr':
        base_dir = f"/data/shared/Control_ICL/Dataset_Acrobot_AIGymWithNoiseShorter_ICL/picklefolder_test_indistr"
        pickle_file = f"batch_test_0_{number_of_context}.pkl"
    elif mode == 'train':
        base_dir = f"/data/shared/Control_ICL/Dataset_Acrobot_AIGymWithNoiseShorter_ICL/picklefolder"
        pickle_file = "batch_0.pkl"
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    file_path_test_data = os.path.join(base_dir, pickle_file)
    with open(file_path_test_data, "rb") as f:
        xs, ys, link_lengths1, link_lengths2, link_masses1, link_masses2, true_xs = pickle.load(f)
        # xs : (t1, t2, t1., t2.)     [B, 120, 4]
        # ys : (u, m)                 [B, 120, 2]
        # link_lengths1: (l1)         [B]
        # link_lengths2: (l2)         [B]
        # link_masses1 : (m1)         [B]
        # link_masses2 : (m2)         [B]
    
    pends = np.arange(Num_of_pendulums)
    link_lengths1 = [link_lengths1[i] for i in pends]
    link_lengths2 = [link_lengths2[i] for i in pends]
    link_masses1 = [link_masses1[i] for i in pends]
    link_masses2 = [link_masses2[i] for i in pends]
    states = [xs[i].cpu().detach().numpy() for i in pends]
    controls = [ys[i].cpu().detach().numpy() for i in pends]
    data_and_controls.append([(xs[i].cpu().numpy(), ys[i].cpu().numpy()) for i in pends])
    # states : list of trajectories (t1_1, t2_1, t1._1, t2._1), ..., (t1_120, t2_120, t1._120, t2._120)
    # controls : list of trajectories (u1, m1), ..., (u120, m120)
    # data_and_controls : list of trajectory tuples ({(t1_1, t2_1, t1._1, t2._1), ..., (t1_120, t2_120, t1._120, t2._120)}, {(u1, m1), ..., (u120, m120)})

    with open(log_info_path, "w") as file:
        counter = 0
        for linklength1, linklength2, linkmass1, linkmass2 in tqdm(zip(link_lengths1, link_lengths2, link_masses1, link_masses2), desc="Acrobot", total=len(link_lengths1), leave=False):
            
            # Grab t1, t2, t1., t2., (u, m) values from counter-th trajectory.
            theta1_rk4 = states[counter][:, 0]
            theta2_rk4 = states[counter][:, 1]
            thetadot1_rk4 = states[counter][:, 2]
            thetadot2_rk4 = states[counter][:, 3]
            control_values_rk4 = controls[counter]

            
            xs_dataset = np.column_stack((theta1_rk4, theta2_rk4, thetadot1_rk4, thetadot2_rk4)) 
            xs_dataset = torch.tensor(xs_dataset).float().to(device)
            control_values_rk4 = torch.tensor(control_values_rk4).float().to(device)
            # xs_dataset : (t1_1, t2_1, t1._1, t2._1), ..., (t1_120, t2_120, t1._120, t2._120)
            # control_values_rk4 : (u1, m1), ..., (u120, u120)

            theta1_plot = []
            theta2_plot = []
            thetadot1_plot = []
            thetadot2_plot = []

            control_values_scaled = control_values_rk4
            states_scaled = xs_dataset
            
            start_index = number_of_context
            
            T_model, theta1_model2, theta2_model2, thetadot1_model2, thetadot2_model2, controls_model2 = run_inference_on_model(
                model, xs_dataset, control_values_rk4, total_time, device, dt,
                context=number_of_context, start_index=start_index, l1=linklength1, l2=linklength2, m1=linkmass1, m2=linkmass2
            )

            # theta1_model2 : t1_1, t1_2, ..., t1_N+1
            # theta2_model2 : t2_1, t2_2, ..., t2_N+1
            # thetadot1_model2 : ---
            # thetadot2_model2 : ---
            # controls_model2 : (u1, m1), ..., (uN, mN)


            trajectory = torch.stack([theta1_model2, theta2_model2, thetadot1_model2, thetadot2_model2], axis=1) 
            controls_for_trajectory = controls_model2
            phase_data[number_of_context].append(trajectory)
            controls_data[number_of_context].append(controls_for_trajectory)
            

            theta1_plot.append(theta1_model2)
            theta2_plot.append(theta2_model2)
            thetadot1_plot.append(thetadot1_model2)
            thetadot2_plot.append(thetadot2_model2)
            
            counter += 1
            
        # mse_mean = {context_length: np.mean(mse_results[context_length]) for context_length in contexts}
        # mse_std = {context_length: np.std(mse_results[context_length]) for context_length in contexts}
        # mse_control_mean = {context_length: np.mean(mse_control_results[context_length]) for context_length in contexts}
        # mse_control_std = {context_length: np.std(mse_control_results[context_length]) for context_length in contexts}
        # print(f"Mean MSE: {mse_mean}")
        # print(f"Std MSE: {mse_std}")

    # plot_mse_vs_context_length(mse_mean, mse_std, save_results_path, folder_name, mse_plot_label, loss_type="state")
    # plot_mse_vs_context_length(mse_control_mean, mse_control_std, save_results_path, folder_name, mse_plot_label, loss_type="control")


                

    # return cartmasses, polemasses, polelenghs, phase_data, controls_data, data_and_controls, pends
    return link_lengths1, link_lengths2, link_masses1, link_masses2, phase_data, controls_data, data_and_controls, pends

try:
    # results = main()

    # model_checkpoint_step_list = [5000, 10000, 15000, 20000, 25000, 30000, 40000, 60000,
    #                               165000, 250000, 255000, 260000, 275000, 284408]

    # model_checkpoint_step_list = [395000]
    # model_checkpoint_step_list = [50000, 135000, 300000]
    # model_checkpoint_step_list = [10000, 20000, 30000, 60000, 100000]
    model_checkpoint_step_list = [300000] #[50000] #, 60000, 10000, 20000, 30000, 100000, 135000, 200000, 300000]
    Num_of_contexts = [1, 5, 10, 25, 50] # [1, 5, 10, 25, 50, 75, 100] , phoenix

    for step in tqdm(model_checkpoint_step_list, desc="Model Checkpoint Steps"):
        model_checkpoint_step = int(step)
        folder_name = f"inference_run/acrobot_{plot_label}_{model_checkpoint_step}_{model_run_id}"

        phase_data = {context_length: [] for context_length in Num_of_contexts}
        controls_data = {context_length: [] for context_length in Num_of_contexts}
        data_and_controls = []

        for Num_of_context in Num_of_contexts:
            print(f"Running inference for checkpoint step {model_checkpoint_step} with context length {Num_of_context}...")

            results = main(model,model_checkpoint_step,
                            folder_name,
                            Num_of_context,
                            # mse_results,
                            # mse_control_results,
                            phase_data,
                            controls_data,
                            data_and_controls,                          
                           ) 
            _, _, _, _, phase_data, controls_data, data_and_controls, pends = results
        
        
        save_results = os.path.join(folder_name, f"results_maxcontext{Num_of_context}_numpends{Num_of_pendulums}_{mode}_alexcode.pkl")    

        with open(save_results, "wb") as f:
            pickle.dump(results, f)
    


    print("done")   
except Exception:
    print("exception starting debugger")
    traceback.print_exc()
    ipdb.post_mortem()